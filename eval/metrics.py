"""RAG 评估指标模块

包含检索层、回答层和 LLM-as-Judge 三层指标。
"""

import json
import os
from typing import List, Dict, Any

from loguru import logger


# ---------------------------------------------------------------------------
# 检索层指标（零成本）
# ---------------------------------------------------------------------------

def hit_rate(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """至少命中一个期望来源则为 1，否则 0"""
    if not expected_sources:
        return 1.0
    return 1.0 if any(
        any(exp in src for src in retrieved_sources)
        for exp in expected_sources
    ) else 0.0


def mrr(retrieved_sources: List[str], expected_sources: List[str]) -> float:
    """Mean Reciprocal Rank：第一个命中的期望来源的排名倒数"""
    if not expected_sources:
        return 1.0
    for i, source in enumerate(retrieved_sources):
        if any(exp in source for exp in expected_sources):
            return 1.0 / (i + 1)
    return 0.0


def keyword_coverage(text: str, expected_keywords: List[str]) -> float:
    """期望关键词在文本中的覆盖率"""
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw in text)
    return hits / len(expected_keywords)


# ---------------------------------------------------------------------------
# 回答层指标（需 LLM 生成回答）
# ---------------------------------------------------------------------------

def answer_keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """回答中期望关键词的覆盖率"""
    return keyword_coverage(answer, expected_keywords)


def answer_length(answer: str) -> int:
    """回答长度（字符数）"""
    return len(answer.strip())


# ---------------------------------------------------------------------------
# LLM-as-Judge（每题额外 1 次 LLM 调用）
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """你是一个严格的事实核查员。你将收到两段内容：
1. <documents> 标签内的参考文档（检索结果）
2. <answer> 标签内的待评估回答

你的任务：逐句分析回答内容，判断每个事实性声明是否有参考文档的支撑。

输出要求：严格输出 JSON，不要输出其他内容。格式如下：
{
  "supported": ["有文档支撑的声明1", "有文档支撑的声明2"],
  "unsupported": ["无文档支撑的声明1"],
  "score": 0.8
}

其中 score = len(supported) / (len(supported) + len(unsupported))，范围 0-1。
如果回答明确说"不确定"或"文档未覆盖"，这不算编造，score 给 1.0。
如果回答为空或仅包含礼貌用语，score 给 1.0。"""


def judge_faithfulness(
    client,
    retrieved_text: str,
    answer: str,
    model: str = None,
) -> Dict[str, Any]:
    """用 LLM 判定回答是否忠于检索文档

    Args:
        client: OpenAI 兼容客户端
        retrieved_text: 检索到的文档内容
        answer: 生成的回答
        model: LLM 模型名

    Returns:
        dict: {"supported": [...], "unsupported": [...], "score": float}
    """
    if not model:
        model = os.getenv("MODEL_NAME", "qwen-max")

    user_msg = (
        f"<documents>\n{retrieved_text}\n</documents>\n\n"
        f"<answer>\n{answer}\n</answer>"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        # 提取 JSON（处理可能的 markdown 包裹）
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        # 确保字段存在
        result.setdefault("supported", [])
        result.setdefault("unsupported", [])
        result.setdefault("score", 1.0)
        return result
    except Exception as e:
        logger.warning(f"LLM-as-Judge 解析失败: {e}")
        return {"supported": [], "unsupported": [], "score": -1.0, "error": str(e)}


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------

def evaluate_single(
    question: Dict,
    retrieved_results: List[Dict],
    answer: str = None,
    client=None,
    model: str = None,
    mode: str = "full",
) -> Dict[str, Any]:
    """评估单个问题

    Args:
        question: 数据集中的问题条目
        retrieved_results: retrieve_with_metadata() 返回的结果列表
        answer: LLM 生成的回答（mode=answer/full 时需要）
        client: OpenAI 客户端（mode=full 时需要）
        model: LLM 模型名
        mode: "retrieval" / "answer" / "full"

    Returns:
        dict: 各项指标结果
    """
    sources = [r["source"] for r in retrieved_results]
    retrieved_text = "\n".join(r["doc"] for r in retrieved_results)
    expected_src = question.get("expected_sources", [])
    expected_kw = question.get("expected_keywords", [])

    result = {
        "id": question["id"],
        "query": question["query"],
        "category": question.get("category", ""),
        # 检索层
        "hit_rate": hit_rate(sources, expected_src),
        "mrr": mrr(sources, expected_src),
        "retrieval_keyword_coverage": keyword_coverage(retrieved_text, expected_kw),
        "retrieved_sources": sources,
    }

    if mode in ("answer", "full") and answer is not None:
        result["answer_keyword_coverage"] = answer_keyword_coverage(answer, expected_kw)
        result["answer_length"] = answer_length(answer)
        result["answer"] = answer

    if mode == "full" and answer is not None and client is not None:
        judge_result = judge_faithfulness(client, retrieved_text, answer, model)
        result["faithfulness"] = judge_result.get("score", -1.0)
        result["hallucination_rate"] = max(0.0, 1.0 - judge_result.get("score", 1.0))
        result["judge_detail"] = judge_result

    return result


def aggregate_results(results: List[Dict], mode: str = "full") -> Dict[str, Any]:
    """汇总所有问题的评估结果"""
    n = len(results)
    if n == 0:
        return {}

    summary = {
        "total_questions": n,
        "avg_hit_rate": sum(r["hit_rate"] for r in results) / n,
        "avg_mrr": sum(r["mrr"] for r in results) / n,
        "avg_retrieval_keyword_coverage": sum(r["retrieval_keyword_coverage"] for r in results) / n,
    }

    if mode in ("answer", "full"):
        answered = [r for r in results if "answer_keyword_coverage" in r]
        if answered:
            summary["avg_answer_keyword_coverage"] = sum(r["answer_keyword_coverage"] for r in answered) / len(answered)
            summary["avg_answer_length"] = sum(r["answer_length"] for r in answered) / len(answered)

    if mode == "full":
        judged = [r for r in results if "faithfulness" in r and r["faithfulness"] >= 0]
        if judged:
            summary["avg_faithfulness"] = sum(r["faithfulness"] for r in judged) / len(judged)
            summary["avg_hallucination_rate"] = sum(r["hallucination_rate"] for r in judged) / len(judged)

    # 按类别汇总
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    summary["by_category"] = {}
    for cat, cat_results in categories.items():
        cat_n = len(cat_results)
        cat_summary = {
            "count": cat_n,
            "avg_hit_rate": sum(r["hit_rate"] for r in cat_results) / cat_n,
            "avg_mrr": sum(r["mrr"] for r in cat_results) / cat_n,
        }
        if mode == "full":
            judged = [r for r in cat_results if "faithfulness" in r and r["faithfulness"] >= 0]
            if judged:
                cat_summary["avg_faithfulness"] = sum(r["faithfulness"] for r in judged) / len(judged)
        summary["by_category"][cat] = cat_summary

    return summary
