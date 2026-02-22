"""重建 RAG 知识库索引

用法: python build_index.py
"""
import os
from rag_service import RAGService
import sys
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")

client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

rag = RAGService(client)
docs_dir = os.getenv("DOCS_DIR", "")

if not docs_dir:
    logger.error("请在 .env 中配置 DOCS_DIR（文档目录路径）")
    sys.exit(1)

if not os.path.isdir(docs_dir):
    logger.error(f"文档目录不存在: {docs_dir}")
    sys.exit(1)

try:
    rag.build_index(docs_dir)
    logger.info(f"索引构建完成，共 {rag.collection.count()} 个文档片段")
except Exception as e:
    logger.error(f"索引构建失败: {e}")
    sys.exit(1)
