"""RAG 评估主脚本

用法:
    # 仅检索评估（零成本）
    uv run python eval/run_eval.py --mode retrieval

    # 检索 + 回答（21 次 LLM 调用）
    uv run python eval/run_eval.py --mode answer

    # 完整评估含 LLM-as-Judge（42 次 LLM 调用）
    uv run python eval/run_eval.py --mode full

    # 参数扫描（仅检索层，零成本）
    uv run python eval/run_eval.py --mode retrieval --sweep
"""

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag_service import RAGService
from eval.metrics import evaluate_single, aggregate_results


def load_dataset(path: str = None) -> list:
    if path is None:
        path = str(PROJECT_ROOT / "eval" / "dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def init_services():
    """初始化 OpenAI 客户端和 RAG 服务"""
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )
    rag = RAGService(client)
    return client, rag


def generate_answer(client, rag: RAGService, query: str, retrieved_text: str) -> str:
    """调用 LLM 生成回答（简化版，不走完整 ReAct 流程）"""
    model = os.getenv("MODEL_NAME", "qwen-max")

    system = (
        "你是 GwenAPI 客服助手。请仅基于以下检索到的文档回答用户问题。\n"
        "如果文档未覆盖该问题，请如实说明。\n\n"
        f"<retrieved_documents>\n{retrieved_text}\n</retrieved_documents>"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.3")),
        max_tokens=600,
    )
    return response.choices[0].message.content or ""


def run_eval(
    questions: list,
    client,
    rag: RAGService,
    mode: str = "retrieval",
    top_k: int = None,
    threshold: float = None,
    bm25_weight: float = None,
    vector_weight: float = None,
    verbose: bool = True,
) -> dict:
    """运行一轮评估"""
    model = os.getenv("MODEL_NAME", "qwen-max")
    results = []

    for i, q in enumerate(questions):
        if verbose:
            logger.info(f"[{i+1}/{len(questions)}] {q['id']}: {q['query']}")

        # 检索
        retrieved = rag.retrieve_with_metadata(
            q["query"],
            top_k=top_k,
            threshold=threshold,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )

        # 生成回答（如果需要）
        answer = None
        if mode in ("answer", "full") and retrieved:
            retrieved_text = "\n---\n".join(
                f"[来源: {r['source']}]\n{r['doc']}" for r in retrieved
            )
            try:
                answer = generate_answer(client, rag, q["query"], retrieved_text)
            except Exception as e:
                logger.error(f"生成回答失败: {e}")
                answer = ""
        elif mode in ("answer", "full"):
            answer = ""

        # 评估
        result = evaluate_single(
            question=q,
            retrieved_results=retrieved,
            answer=answer,
            client=client if mode == "full" else None,
            model=model,
            mode=mode,
        )
        results.append(result)

        if verbose:
            status = "HIT" if result["hit_rate"] > 0 else "MISS"
            logger.info(f"  -> {status} | MRR={result['mrr']:.2f} | KW={result['retrieval_keyword_coverage']:.2f}")
            if "faithfulness" in result:
                logger.info(f"  -> Faithfulness={result['faithfulness']:.2f}")

    summary = aggregate_results(results, mode=mode)
    return {"summary": summary, "details": results}


def run_sweep(questions: list, client, rag: RAGService):
    """参数扫描（仅检索层）"""
    top_k_values = [3, 5, 7]
    threshold_values = [0.3, 0.4, 0.5, 0.6]
    bm25_weight_values = [0.0, 0.2, 0.4, 0.6]

    best_score = -1
    best_params = {}
    all_results = []

    total = len(top_k_values) * len(threshold_values) * len(bm25_weight_values)
    count = 0

    for tk, th, bw in product(top_k_values, threshold_values, bm25_weight_values):
        vw = round(1.0 - bw, 1)
        count += 1
        logger.info(f"[{count}/{total}] top_k={tk}, threshold={th}, bm25_weight={bw}, vector_weight={vw}")

        result = run_eval(
            questions, client, rag,
            mode="retrieval",
            top_k=tk,
            threshold=th,
            bm25_weight=bw,
            vector_weight=vw,
            verbose=False,
        )

        s = result["summary"]
        # 综合评分：Hit Rate * 0.4 + MRR * 0.3 + Keyword Coverage * 0.3
        composite = (
            s["avg_hit_rate"] * 0.4
            + s["avg_mrr"] * 0.3
            + s["avg_retrieval_keyword_coverage"] * 0.3
        )

        entry = {
            "top_k": tk,
            "threshold": th,
            "bm25_weight": bw,
            "vector_weight": vw,
            "hit_rate": round(s["avg_hit_rate"], 4),
            "mrr": round(s["avg_mrr"], 4),
            "keyword_coverage": round(s["avg_retrieval_keyword_coverage"], 4),
            "composite": round(composite, 4),
        }
        all_results.append(entry)

        if composite > best_score:
            best_score = composite
            best_params = entry

        logger.info(
            f"  HR={entry['hit_rate']:.3f} MRR={entry['mrr']:.3f} "
            f"KC={entry['keyword_coverage']:.3f} Composite={entry['composite']:.3f}"
        )

    return {"best": best_params, "all_results": all_results}


def print_report(report: dict, mode: str):
    """打印评估报告"""
    s = report["summary"]
    print("\n" + "=" * 60)
    print("RAG 评估报告")
    print("=" * 60)
    print(f"评估模式: {mode}")
    print(f"问题总数: {s['total_questions']}")
    print()
    print("--- 检索层指标 ---")
    print(f"  Hit Rate:              {s['avg_hit_rate']:.3f}")
    print(f"  MRR:                   {s['avg_mrr']:.3f}")
    print(f"  Keyword Coverage:      {s['avg_retrieval_keyword_coverage']:.3f}")

    if "avg_answer_keyword_coverage" in s:
        print()
        print("--- 回答层指标 ---")
        print(f"  Answer KW Coverage:    {s['avg_answer_keyword_coverage']:.3f}")
        print(f"  Avg Answer Length:     {s['avg_answer_length']:.0f} chars")

    if "avg_faithfulness" in s:
        print()
        print("--- LLM-as-Judge ---")
        print(f"  Faithfulness:          {s['avg_faithfulness']:.3f}")
        print(f"  Hallucination Rate:    {s['avg_hallucination_rate']:.3f}")

    if "by_category" in s:
        print()
        print("--- 分类别 ---")
        for cat, cs in s["by_category"].items():
            faith_str = ""
            if "avg_faithfulness" in cs:
                faith_str = f" | Faith={cs['avg_faithfulness']:.2f}"
            print(f"  [{cat}] (n={cs['count']}) HR={cs['avg_hit_rate']:.2f} MRR={cs['avg_mrr']:.2f}{faith_str}")

    # 打印未命中的问题
    missed = [d for d in report["details"] if d["hit_rate"] == 0]
    if missed:
        print()
        print(f"--- 未命中问题 ({len(missed)}) ---")
        for d in missed:
            print(f"  {d['id']}: {d['query']}")
            print(f"    期望来源: {d.get('expected_sources', [])}")
            print(f"    实际来源: {d.get('retrieved_sources', [])}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 评估脚本")
    parser.add_argument("--mode", choices=["retrieval", "answer", "full"], default="retrieval")
    parser.add_argument("--sweep", action="store_true", help="参数扫描模式")
    parser.add_argument("--dataset", type=str, default=None, help="数据集路径")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 加载环境变量
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))
    env_example = PROJECT_ROOT / ".env.example"
    if env_example.exists():
        load_dotenv(str(env_example))

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    questions = load_dataset(args.dataset)
    logger.info(f"加载了 {len(questions)} 个测试问题")

    client, rag = init_services()
    logger.info(f"RAG 索引包含 {rag.collection.count()} 个文档片段")

    if args.sweep:
        logger.info("开始参数扫描...")
        sweep_result = run_sweep(questions, client, rag)
        print("\n" + "=" * 60)
        print("参数扫描结果")
        print("=" * 60)
        best = sweep_result["best"]
        print(f"最优参数: top_k={best['top_k']}, threshold={best['threshold']}, "
              f"bm25_weight={best['bm25_weight']}, vector_weight={best['vector_weight']}")
        print(f"最优评分: HR={best['hit_rate']:.3f} MRR={best['mrr']:.3f} "
              f"KC={best['keyword_coverage']:.3f} Composite={best['composite']:.3f}")
        print("=" * 60)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(sweep_result, f, ensure_ascii=False, indent=2)
            logger.info(f"扫描结果已保存到: {args.output}")
    else:
        logger.info(f"开始评估 (mode={args.mode})...")
        start = time.time()
        report = run_eval(questions, client, rag, mode=args.mode)
        elapsed = time.time() - start
        print_report(report, args.mode)
        logger.info(f"评估完成，耗时 {elapsed:.1f}s")

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
