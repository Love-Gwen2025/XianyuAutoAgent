import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict
from collections import Counter
import math

import chromadb
import jieba
from chromadb import EmbeddingFunction, Embeddings, Documents
from loguru import logger

# 中文停用词（高频无意义词）
_STOPWORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "怎么", "如何", "可以", "能", "吗", "呢", "吧", "啊", "哦",
    "被", "把", "让", "用", "从", "为", "对", "与", "而", "但", "或",
    "如果", "因为", "所以", "虽然", "这个", "那个", "还是", "已经",
    "之", "其", "及", "等", "以", "于", "中", "个", "来", "下",
})

# 加载自定义词典（放在项目根目录 data/user_dict.txt）
_USER_DICT = os.path.join(os.path.dirname(__file__), "data", "user_dict.txt")
if os.path.exists(_USER_DICT):
    jieba.load_userdict(_USER_DICT)
    logger.info(f"已加载 jieba 自定义词典: {_USER_DICT}")


class DashScopeEmbedding(EmbeddingFunction):
    """基于阿里 DashScope 的 Embedding 函数，复用 OpenAI 兼容接口"""

    def __init__(self, client, model: str = "text-embedding-v3", dimensions: int = 1024):
        self.client = client
        self.model = model
        self.dimensions = dimensions
        # DashScope 兼容接口对 batch size 有限制（<=10）
        self.max_batch_size = int(os.getenv("EMBEDDING_MAX_BATCH_SIZE", "10"))

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        if isinstance(input, str):
            input = [input]

        all_embeddings: Embeddings = []
        for i in range(0, len(input), self.max_batch_size):
            batch = input[i:i + self.max_batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                logger.error(f"Embedding API 调用失败 (batch {i//self.max_batch_size}): {e}")
                # 填充零向量保持索引对齐
                all_embeddings.extend([[0.0] * self.dimensions for _ in batch])
        return all_embeddings


class RAGService:
    """RAG 知识库服务：文档加载、向量化、检索"""

    # 跳过的文件（纯 frontmatter 导航页，无实际内容）
    SKIP_FILES = {"index.md", "contact.md", "main-guide.md"}
    # 最小 chunk 长度，过短的片段没有检索价值
    MIN_CHUNK_LENGTH = 20
    # 最大 chunk 长度，超过后按段落边界二次分割
    MAX_CHUNK_LENGTH = 1500
    def __init__(self, openai_client):
        self.enabled = os.getenv("RAG_ENABLED", "True").lower() == "true"
        if not self.enabled:
            logger.info("RAG 服务已禁用")
            return

        self.openai_client = openai_client
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
        self.top_k = int(os.getenv("RAG_TOP_K", "5"))
        self.threshold = float(os.getenv("RAG_THRESHOLD", "0.5"))
        self.strict = os.getenv("RAG_STRICT", "True").lower() == "true"
        self.bm25_enabled = os.getenv("RAG_BM25", "True").lower() == "true"
        self.bm25_k1 = float(os.getenv("RAG_BM25_K1", "1.2"))
        self.bm25_b = float(os.getenv("RAG_BM25_B", "0.75"))
        self.bm25_weight = float(os.getenv("RAG_BM25_WEIGHT", "0.2"))
        self.vector_weight = float(os.getenv("RAG_VECTOR_WEIGHT", "0.8"))
        self.bm25_index_ready = False
        self.bm25_docs: List[str] = []
        self.bm25_metas: List[Dict] = []
        self.bm25_doc_freqs: List[Counter] = []
        self.bm25_df: Counter = Counter()
        self.bm25_doc_len: List[int] = []
        self.bm25_avgdl: float = 0.0
        # 初始化 ChromaDB
        db_path = os.getenv("CHROMA_DB_PATH", "data/chroma_db")
        self.embedding_fn = DashScopeEmbedding(openai_client, self.embedding_model, self.dimensions)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="gwenapi_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"RAG 服务初始化完成，当前索引包含 {self.collection.count()} 个文档片段")
        if self.bm25_enabled and self.collection.count() > 0:
            self._build_bm25_from_collection()

    def build_index(self, docs_dir: str):
        """全量重建索引"""
        try:
            self.chroma_client.delete_collection("gwenapi_docs")
        except Exception as e:
            logger.warning(f"删除旧 collection 失败（可能不存在）: {e}")
        self.collection = self.chroma_client.create_collection(
            name="gwenapi_docs",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

        chunks = self._load_and_split(docs_dir)
        if not chunks:
            logger.warning("未加载到任何文档片段")
            return
        # 批量写入（ChromaDB 单次最多 5461 条，这里远不到）
        ids = [hashlib.md5(f"{c['source']}:{c['title']}:{c['text']}".encode()).hexdigest() for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "title": c["title"]} for c in chunks]

        # 去重：相同 source/title/text 会产生相同 ID，保留第一条
        seen = set()
        unique_ids = []
        unique_documents = []
        unique_metadatas = []
        for id_, doc, meta in zip(ids, documents, metadatas):
            if id_ in seen:
                continue
            seen.add(id_)
            unique_ids.append(id_)
            unique_documents.append(doc)
            unique_metadatas.append(meta)

        self.collection.add(ids=unique_ids, documents=unique_documents, metadatas=unique_metadatas)
        if len(unique_ids) != len(ids):
            logger.info(f"检测到重复片段，已去重 {len(ids) - len(unique_ids)} 条")
        if self.bm25_enabled:
            self._build_bm25_index(unique_documents, unique_metadatas)
        logger.info(f"索引构建完成，共 {len(unique_ids)} 个文档片段")

    def _load_and_split(self, docs_dir: str) -> List[Dict]:
        """加载并切分 markdown 文档"""
        chunks = []
        docs_path = Path(docs_dir)

        for md_file in sorted(docs_path.rglob("*.md")):
            # 跳过 node_modules 和 .vitepress
            if "node_modules" in md_file.parts or ".vitepress" in md_file.parts:
                continue
            # 跳过纯导航页
            if md_file.name in self.SKIP_FILES:
                continue

            relative_path = md_file.relative_to(docs_path).as_posix()
            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"读取文件失败，跳过: {relative_path}, 错误: {e}")
                continue

            # 去除 frontmatter
            content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content, flags=re.DOTALL)
            content = content.strip()
            if not content:
                continue

            # 按一级/二级标题切分
            sections = re.split(r'\n(?=#{1,2}\s)', content)
            for section in sections:
                section = section.strip()
                if len(section) < self.MIN_CHUNK_LENGTH:
                    continue
                # 提取标题
                title_match = re.match(r'^(#{1,2})\s+(.+)', section)
                title = title_match.group(2).strip() if title_match else relative_path

                # 对超长 chunk 按段落边界二次分割
                sub_chunks = self._split_long_chunk(section)
                for i, sub in enumerate(sub_chunks):
                    chunk_title = f"{title} (续{i+1})" if len(sub_chunks) > 1 and i > 0 else title
                    chunks.append({
                        "text": sub,
                        "source": relative_path,
                        "title": chunk_title,
                    })

        logger.info(f"从 {docs_dir} 加载了 {len(chunks)} 个文档片段")
        return chunks

    def _split_long_chunk(self, text: str) -> List[str]:
        """对超过 MAX_CHUNK_LENGTH 的 chunk 按段落边界分割"""
        if len(text) <= self.MAX_CHUNK_LENGTH:
            return [text]

        paragraphs = re.split(r'\n\n+', text)
        result = []
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > self.MAX_CHUNK_LENGTH:
                result.append(current.strip())
                current = para
            else:
                current = current + "\n\n" + para if current else para
        if current.strip():
            result.append(current.strip())

        # 过滤太短的片段
        return [c for c in result if len(c) >= self.MIN_CHUNK_LENGTH]

    def retrieve(self, query: str) -> str:
        """检索相关文档片段，返回拼接后的文本"""
        if not self.enabled:
            return ""
        if self.collection.count() == 0:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=self.top_k
            )
        except Exception as e:
            logger.error(f"RAG 检索失败: {e}")
            return ""

        # 1) 向量检索候选
        vector_candidates = []
        for doc, distance, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):
            if distance < self.threshold:
                vector_candidates.append((doc, metadata, distance))

        # 2) BM25 候选
        bm25_candidates = []
        if self.bm25_enabled and self.bm25_index_ready:
            bm25_candidates = self._bm25_query(query)

        # 3) 混合排序（按加权分）
        combined = {}
        # vector score: 1 - distance (cosine distance)
        for doc, meta, dist in vector_candidates:
            score = max(0.0, 1.0 - dist)
            key = (meta.get("source", ""), doc)
            combined[key] = {"doc": doc, "meta": meta, "v": score, "b": 0.0}
        if bm25_candidates:
            max_bm25 = max(s for _, _, s in bm25_candidates) or 1.0
            for doc, meta, score in bm25_candidates:
                key = (meta.get("source", ""), doc)
                if key not in combined:
                    combined[key] = {"doc": doc, "meta": meta, "v": 0.0, "b": 0.0}
                combined[key]["b"] = score / max_bm25

        if not combined:
            return ""

        ranked = sorted(
            combined.values(),
            key=lambda x: (self.vector_weight * x["v"] + self.bm25_weight * x["b"]),
            reverse=True
        )
        ranked = ranked[: self.top_k]
        return "\n---\n".join([f"[来源: {r['meta']['source']}]\n{r['doc']}" for r in ranked])

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = None,
        threshold: float = None,
        bm25_weight: float = None,
        vector_weight: float = None,
    ) -> List[Dict]:
        """检索相关文档片段，返回结构化结果（用于评估）

        Args:
            query: 查询文本
            top_k: 覆盖默认 top_k
            threshold: 覆盖默认阈值
            bm25_weight: 覆盖默认 BM25 权重
            vector_weight: 覆盖默认向量权重

        Returns:
            list[dict]: 每个元素包含 doc, source, title, score
        """
        if not self.enabled or self.collection.count() == 0:
            return []

        _top_k = top_k if top_k is not None else self.top_k
        _threshold = threshold if threshold is not None else self.threshold
        _bm25_w = bm25_weight if bm25_weight is not None else self.bm25_weight
        _vector_w = vector_weight if vector_weight is not None else self.vector_weight

        try:
            results = self.collection.query(query_texts=[query], n_results=_top_k)
        except Exception as e:
            logger.error(f"RAG 检索失败: {e}")
            return []

        vector_candidates = []
        for doc, distance, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            if distance < _threshold:
                vector_candidates.append((doc, metadata, distance))

        bm25_candidates = []
        if self.bm25_enabled and self.bm25_index_ready:
            bm25_candidates = self._bm25_query(query)

        combined = {}
        for doc, meta, dist in vector_candidates:
            score = max(0.0, 1.0 - dist)
            key = (meta.get("source", ""), doc)
            combined[key] = {"doc": doc, "meta": meta, "v": score, "b": 0.0}
        if bm25_candidates:
            max_bm25 = max(s for _, _, s in bm25_candidates) or 1.0
            for doc, meta, score in bm25_candidates:
                key = (meta.get("source", ""), doc)
                if key not in combined:
                    combined[key] = {"doc": doc, "meta": meta, "v": 0.0, "b": 0.0}
                combined[key]["b"] = score / max_bm25

        if not combined:
            return []

        ranked = sorted(
            combined.values(),
            key=lambda x: (_vector_w * x["v"] + _bm25_w * x["b"]),
            reverse=True,
        )
        ranked = ranked[:_top_k]

        return [
            {
                "doc": r["doc"],
                "source": r["meta"].get("source", ""),
                "title": r["meta"].get("title", ""),
                "score": round(_vector_w * r["v"] + _bm25_w * r["b"], 4),
            }
            for r in ranked
        ]

    def _build_bm25_from_collection(self) -> None:
        """从 ChromaDB 现有集合构建 BM25 索引"""
        try:
            data = self.collection.get(include=["documents", "metadatas"])
            documents = data.get("documents", []) or []
            metadatas = data.get("metadatas", []) or []
            if documents:
                self._build_bm25_index(documents, metadatas)
        except Exception as e:
            logger.warning(f"BM25 索引构建失败，将仅使用向量检索: {e}")

    def _build_bm25_index(self, documents: List[str], metadatas: List[Dict]) -> None:
        self.bm25_docs = documents
        self.bm25_metas = metadatas
        self.bm25_doc_freqs = []
        self.bm25_df = Counter()
        self.bm25_doc_len = []

        for doc in documents:
            tokens = self._bm25_tokenize(doc)
            freqs = Counter(tokens)
            self.bm25_doc_freqs.append(freqs)
            self.bm25_doc_len.append(len(tokens))
            for t in freqs.keys():
                self.bm25_df[t] += 1

        n_docs = len(documents)
        self.bm25_avgdl = (sum(self.bm25_doc_len) / n_docs) if n_docs else 0.0
        self.bm25_index_ready = n_docs > 0
        logger.info(f"BM25 索引构建完成，共 {n_docs} 个文档片段")

    def _bm25_tokenize(self, text: str) -> List[str]:
        """jieba 精确分词 + 停用词过滤"""
        if not text:
            return []
        text = text.lower()
        # 英文/数字 token
        en_tokens = [t for t in re.findall(r"[a-z0-9_.-]{2,}", text) if t not in _STOPWORDS]
        # 中文用 jieba 精确模式分词，过滤单字和停用词
        zh_tokens = [w for w in jieba.cut(text) if len(w) >= 2 and re.match(r"[\u4e00-\u9fa5]{2,}", w) and w not in _STOPWORDS]
        return en_tokens + zh_tokens

    def _bm25_query(self, query: str):
        tokens = self._bm25_tokenize(query)
        if not tokens:
            return []
        n_docs = len(self.bm25_docs)
        if n_docs == 0:
            return []
        scores = []
        for i, freqs in enumerate(self.bm25_doc_freqs):
            dl = self.bm25_doc_len[i]
            if dl == 0:
                continue
            score = 0.0
            for t in tokens:
                df = self.bm25_df.get(t, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                tf = freqs.get(t, 0)
                if tf == 0:
                    continue
                denom = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * dl / (self.bm25_avgdl or 1.0))
                score += idf * (tf * (self.bm25_k1 + 1)) / (denom or 1.0)

            if score > 0:
                doc = self.bm25_docs[i]
                meta = self.bm25_metas[i] if i < len(self.bm25_metas) else {"source": ""}
                scores.append((doc, meta, score))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[: max(self.top_k, 5)]
