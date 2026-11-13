"""Independent retrieval benchmark for the LlamaIndex RAG project.

This script does not call an LLM and never writes Chroma or DocStore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from llama_index.core import Settings
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.bridge.pydantic import PrivateAttr

from src.config import get_settings
from src.retriever import (
    _chinese_tokenizer,
    _load_chroma_vector_store,
    _load_persisted_nodes,
)
from src.settings import _build_embedding

from evaluation.retrieval_metrics import (
    STRATEGIES,
    load_jsonl,
    question_level_differences,
    write_per_question_csv,
    write_summary_csv,
)


DATASET_PATH = ROOT / "evaluation" / "datasets" / "rag_eval_50.jsonl"
RESULTS_ROOT = ROOT / "evaluation" / "results"
EXPECTED_CATEGORIES = {
    "参数查询",
    "安装操作",
    "故障排查",
    "跨产品型号归属辨析",
    "跨语言检索",
}
FINAL_TOP_K = 3
RRF_K = 60.0


class ReusableResultRetriever(BaseRetriever):
    """Return precomputed candidates so B/C share identical component results."""

    _results: list[NodeWithScore] = PrivateAttr(default_factory=list)

    def __init__(self, results: list[NodeWithScore]) -> None:
        super().__init__()
        self._results = list(results)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._results


@dataclass
class CallCounter:
    embedding_calls: int = 0
    vector_calls: int = 0
    bm25_calls: int = 0
    rrf_calls: int = 0
    rerank_calls: int = 0
    api_failures: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def record_error(
        self, question_id: str, round_number: int, stage: str, exc: BaseException
    ) -> None:
        self.errors.append(
            {
                "question_id": question_id,
                "round": str(round_number),
                "stage": stage,
                "error_type": type(exc).__name__,
                "message": sanitize_error(exc),
            }
        )
        if stage in {"query_embedding", "rerank"}:
            self.api_failures += 1


def sanitize_error(exc: BaseException, secrets: list[str] | None = None) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for secret in secrets or []:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    return message[:1000]


@dataclass
class Runtime:
    cfg: Any
    records: list[dict[str, Any]]
    vector_store: Any
    embedding_model: Any
    bm25_retriever: BM25Retriever
    reranker: DashScopeRerank
    call_counter: CallCounter
    parameters: dict[str, Any]
    preflight: dict[str, Any]
    secrets: list[str]


def timed_call(callable_obj, *args, **kwargs):
    started = time.perf_counter()
    try:
        result = callable_obj(*args, **kwargs)
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    return result, elapsed_ms


def make_run_dir() -> Path:
    base = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = RESULTS_ROOT / base
    suffix = 1
    while candidate.exists():
        candidate = RESULTS_ROOT / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def validate_dataset(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) != 50:
        raise ValueError(f"dataset must contain 50 records, got {len(records)}")
    ids = [record["question_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("question_id values are not unique")
    categories = Counter(record["category"] for record in records)
    if set(categories) != EXPECTED_CATEGORIES:
        raise ValueError(f"unexpected categories: {sorted(categories)}")
    if any(count != 10 for count in categories.values()):
        raise ValueError(f"category counts are not all 10: {dict(categories)}")
    return {
        "question_count": len(records),
        "unique_question_ids": len(set(ids)),
        "category_counts": dict(sorted(categories.items())),
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
    }


def build_runtime(records: list[dict[str, Any]]) -> Runtime:
    cfg = get_settings()
    if cfg.embedding_provider.strip().lower() != "dashscope":
        raise ValueError("benchmark requires EMBEDDING_PROVIDER=dashscope")
    if cfg.embedding_model.strip() != "text-embedding-v3":
        raise ValueError(f"unexpected embedding model: {cfg.embedding_model}")
    if cfg.rerank_model.strip() != "gte-rerank-v2":
        raise ValueError(f"unexpected reranker model: {cfg.rerank_model}")

    collection = chromadb.PersistentClient(path=str(cfg.chroma_dir)).get_collection(
        name=cfg.chroma_collection
    )
    chroma_ids = set(collection.get(include=[])["ids"])
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(cfg.docstore_dir)
    )
    docstore_ids = set(docstore.docs)
    if chroma_ids != docstore_ids:
        raise ValueError(
            f"Chroma/DocStore mismatch: chroma={len(chroma_ids)}, "
            f"docstore={len(docstore_ids)}"
        )

    expected_chunks = set()
    for record in records:
        chunks = record.get("expected_chunks") or [record["expected_chunk"]]
        if not chunks:
            raise ValueError(f"{record['question_id']} has no expected chunk")
        expected_chunks.update(chunks)
    missing = expected_chunks - chroma_ids
    if missing:
        raise ValueError(f"expected chunks missing from index: {sorted(missing)}")

    vector_store = _load_chroma_vector_store()
    nodes = _load_persisted_nodes()
    if len(nodes) != len(docstore_ids):
        raise ValueError(f"BM25 node count mismatch: {len(nodes)}")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=max(cfg.bm25_top_k * 3, 15),
        tokenizer=_chinese_tokenizer,
    )
    embedding_model = _build_embedding(cfg)
    Settings.embed_model = embedding_model
    # QueryFusionRetriever resolves the global LLM even when num_queries=1.
    # This benchmark is retrieval-only, so disable the LLM in this process.
    Settings.llm = None
    reranker = DashScopeRerank(
        model=cfg.rerank_model,
        top_n=FINAL_TOP_K,
        api_key=cfg.dashscope_api_key,
    )

    parameters = {
        "embedding_model": cfg.embedding_model,
        "rerank_model": cfg.rerank_model,
        "chroma_collection": cfg.chroma_collection,
        "vector_candidate_k": max(cfg.vector_top_k * 3, 15),
        "bm25_candidate_k": max(cfg.bm25_top_k * 3, 15),
        "rrf_candidate_pool_k": cfg.vector_top_k + cfg.bm25_top_k,
        "rrf_mode": "reciprocal_rerank",
        "rrf_k": RRF_K,
        "rrf_num_queries": 1,
        "rrf_use_async": False,
        "final_top_k": FINAL_TOP_K,
        "production_rerank_top_n": cfg.rerank_top_n,
        "evaluation_rerank_top_n": FINAL_TOP_K,
    }
    preflight = {
        **validate_dataset(records),
        "chroma_count": len(chroma_ids),
        "docstore_count": len(docstore_ids),
        "expected_chunk_count": len(expected_chunks),
        "bm25_node_count": len(nodes),
    }
    secrets = [cfg.dashscope_api_key]
    if cfg.openai_api_key:
        secrets.append(cfg.openai_api_key.get_secret_value())
    return Runtime(
        cfg=cfg,
        records=records,
        vector_store=vector_store,
        embedding_model=embedding_model,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        call_counter=CallCounter(),
        parameters=parameters,
        preflight=preflight,
        secrets=secrets,
    )


def node_hit(node_with_score: NodeWithScore) -> dict[str, Any]:
    node = node_with_score.node
    metadata = node.metadata or {}
    score = node_with_score.score
    return {
        "node_id": node.node_id,
        "document": metadata.get("original_name") or metadata.get("file_name"),
        "page_label": metadata.get("page_label"),
        "score": float(score) if score is not None else None,
        "text": node.get_content(),
    }


def hits(node_with_scores: list[NodeWithScore]) -> list[dict[str, Any]]:
    return [
        {"rank": rank, **node_hit(item)}
        for rank, item in enumerate(node_with_scores, start=1)
    ]


def recall_at_3(
    top_hits: list[dict[str, Any]], expected_chunks: list[str]
) -> float:
    returned = {item["node_id"] for item in top_hits[:FINAL_TOP_K]}
    return 1.0 if returned.intersection(expected_chunks) else 0.0


def failed_row(
    runtime: Runtime,
    question: dict[str, Any],
    strategy: str,
    round_number: int,
    stage: str,
    exc: BaseException,
    embedding_ms: float | None,
    vector_ms: float | None,
    bm25_ms: float | None,
    rrf_ms: float | None,
    rerank_ms: float | None,
) -> dict[str, Any]:
    vector_calls = int(vector_ms is not None or stage == "vector")
    bm25_calls = int(bm25_ms is not None or stage == "bm25")
    rrf_calls = int(rrf_ms is not None or stage == "rrf")
    rerank_calls = int(rerank_ms is not None or stage == "rerank")
    return {
        "question_id": question["question_id"],
        "category": question["category"],
        "question": question["question"],
        "strategy": strategy,
        "strategy_name": STRATEGIES[strategy],
        "round": round_number,
        "expected_chunks": question.get("expected_chunks")
        or [question["expected_chunk"]],
        "top3": [],
        "recall_at_3": None,
        "error_stage": stage,
        "error_type": type(exc).__name__,
        "error_message": sanitize_error(exc, runtime.secrets),
        "embedding_ms": embedding_ms,
        "vector_ms": vector_ms,
        "bm25_ms": bm25_ms,
        "rrf_ms": rrf_ms,
        "rerank_ms": rerank_ms,
        "total_ms": None,
        "embedding_calls": 1,
        "vector_calls": vector_calls,
        "bm25_calls": bm25_calls,
        "rrf_calls": rrf_calls,
        "rerank_calls": rerank_calls,
        "vector_candidate_ids": [],
        "bm25_candidate_ids": [],
        "rrf_candidate_ids": [],
    }


def run_round(
    runtime: Runtime, question: dict[str, Any], round_number: int
) -> list[dict[str, Any]]:
    cfg = runtime.cfg
    query = question["question"]
    expected_chunks = question.get("expected_chunks") or [
        question["expected_chunk"]
    ]
    shared = {
        "embedding_ms": None,
        "vector_ms": None,
        "bm25_ms": None,
        "rrf_ms": None,
        "rerank_ms": None,
    }

    try:
        runtime.call_counter.embedding_calls += 1
        embedding, shared["embedding_ms"] = timed_call(
            runtime.embedding_model.get_query_embedding, query
        )
    except Exception as exc:
        runtime.call_counter.record_error(
            question["question_id"], round_number, "query_embedding", exc
        )
        return [
            failed_row(
                runtime,
                question,
                strategy,
                round_number,
                "query_embedding",
                exc,
                shared["embedding_ms"],
                None,
                None,
                None,
                None,
            )
            for strategy in STRATEGIES
        ]

    vector_error = None
    vector_results: list[NodeWithScore] = []
    try:
        runtime.call_counter.vector_calls += 1
        vector_query = VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=runtime.parameters["vector_candidate_k"],
        )
        vector_result, shared["vector_ms"] = timed_call(
            runtime.vector_store.query, vector_query
        )
        similarities = vector_result.similarities or []
        vector_results = [
            NodeWithScore(
                node=node,
                score=float(similarities[index])
                if index < len(similarities)
                else None,
            )
            for index, node in enumerate(vector_result.nodes or [])
        ]
    except Exception as exc:
        vector_error = exc
        runtime.call_counter.record_error(
            question["question_id"], round_number, "vector", exc
        )

    bm25_error = None
    bm25_results: list[NodeWithScore] = []
    try:
        runtime.call_counter.bm25_calls += 1
        bm25_results, shared["bm25_ms"] = timed_call(
            runtime.bm25_retriever.retrieve, QueryBundle(query_str=query)
        )
    except Exception as exc:
        bm25_error = exc
        runtime.call_counter.record_error(
            question["question_id"], round_number, "bm25", exc
        )

    rows: list[dict[str, Any]] = []
    vector_top = vector_results[:FINAL_TOP_K]
    if vector_error is None:
        rows.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "question": query,
                "strategy": "A",
                "strategy_name": STRATEGIES["A"],
                "round": round_number,
                "expected_chunks": expected_chunks,
                "top3": hits(vector_top),
                "recall_at_3": recall_at_3(hits(vector_top), expected_chunks),
                "error_stage": None,
                "error_type": None,
                "error_message": None,
                **shared,
                "total_ms": shared["embedding_ms"] + shared["vector_ms"],
                "embedding_calls": 1,
                "vector_calls": 1,
                "bm25_calls": int(shared["bm25_ms"] is not None),
                "rrf_calls": 0,
                "rerank_calls": 0,
                "vector_candidate_ids": [
                    result.node.node_id for result in vector_results
                ],
                "bm25_candidate_ids": [
                    result.node.node_id for result in bm25_results
                ],
                "rrf_candidate_ids": [],
            }
        )
    else:
        rows.append(
            failed_row(
                runtime,
                question,
                "A",
                round_number,
                "vector",
                vector_error,
                shared["embedding_ms"],
                shared["vector_ms"],
                shared["bm25_ms"],
                None,
                None,
            )
        )

    fusion_error = None
    fused_results: list[NodeWithScore] = []
    if vector_error is None and bm25_error is None:
        try:
            runtime.call_counter.rrf_calls += 1
            fused_retriever = QueryFusionRetriever(
                retrievers=[
                    ReusableResultRetriever(vector_results),
                    ReusableResultRetriever(bm25_results),
                ],
                similarity_top_k=runtime.parameters["rrf_candidate_pool_k"],
                num_queries=runtime.parameters["rrf_num_queries"],
                mode=runtime.parameters["rrf_mode"],
                use_async=runtime.parameters["rrf_use_async"],
                verbose=False,
            )
            fused_results, shared["rrf_ms"] = timed_call(
                fused_retriever.retrieve, QueryBundle(query_str=query)
            )
        except Exception as exc:
            fusion_error = exc
            runtime.call_counter.record_error(
                question["question_id"], round_number, "rrf", exc
            )
    elif vector_error is not None:
        fusion_error = vector_error
    elif bm25_error is not None:
        fusion_error = bm25_error

    if fusion_error is None:
        fused_top = fused_results[:FINAL_TOP_K]
        rows.append(
            {
                "question_id": question["question_id"],
                "category": question["category"],
                "question": query,
                "strategy": "B",
                "strategy_name": STRATEGIES["B"],
                "round": round_number,
                "expected_chunks": expected_chunks,
                "top3": hits(fused_top),
                "recall_at_3": recall_at_3(hits(fused_top), expected_chunks),
                "error_stage": None,
                "error_type": None,
                "error_message": None,
                **shared,
                "total_ms": (
                    shared["embedding_ms"]
                    + shared["vector_ms"]
                    + shared["bm25_ms"]
                    + shared["rrf_ms"]
                ),
                "embedding_calls": 1,
                "vector_calls": 1,
                "bm25_calls": 1,
                "rrf_calls": 1,
                "rerank_calls": 0,
                "vector_candidate_ids": [
                    result.node.node_id for result in vector_results
                ],
                "bm25_candidate_ids": [
                    result.node.node_id for result in bm25_results
                ],
                "rrf_candidate_ids": [
                    result.node.node_id for result in fused_results
                ],
            }
        )
    else:
        fusion_failure_stage = (
            "vector"
            if vector_error is not None
            else "bm25"
            if bm25_error is not None
            else "rrf"
        )
        for strategy in ("B", "C"):
            rows.append(
                failed_row(
                    runtime,
                    question,
                    strategy,
                    round_number,
                    fusion_failure_stage,
                    fusion_error,
                    shared["embedding_ms"],
                    shared["vector_ms"],
                    shared["bm25_ms"],
                    shared["rrf_ms"],
                    None,
                )
            )

    if fusion_error is None:
        try:
            runtime.call_counter.rerank_calls += 1
            reranked, shared["rerank_ms"] = timed_call(
                runtime.reranker.postprocess_nodes,
                fused_results,
                query_bundle=QueryBundle(query_str=query),
            )
            reranked_top = reranked[:FINAL_TOP_K]
            rows.append(
                {
                    "question_id": question["question_id"],
                    "category": question["category"],
                    "question": query,
                    "strategy": "C",
                    "strategy_name": STRATEGIES["C"],
                    "round": round_number,
                    "expected_chunks": expected_chunks,
                    "top3": hits(reranked_top),
                    "recall_at_3": recall_at_3(
                        hits(reranked_top), expected_chunks
                    ),
                    "error_stage": None,
                    "error_type": None,
                    "error_message": None,
                    **shared,
                    "total_ms": (
                        shared["embedding_ms"]
                        + shared["vector_ms"]
                        + shared["bm25_ms"]
                        + shared["rrf_ms"]
                        + shared["rerank_ms"]
                    ),
                    "embedding_calls": 1,
                    "vector_calls": 1,
                    "bm25_calls": 1,
                    "rrf_calls": 1,
                    "rerank_calls": 1,
                    "vector_candidate_ids": [
                        result.node.node_id for result in vector_results
                    ],
                    "bm25_candidate_ids": [
                        result.node.node_id for result in bm25_results
                    ],
                    "rrf_candidate_ids": [
                        result.node.node_id for result in fused_results
                    ],
                }
            )
        except Exception as exc:
            runtime.call_counter.record_error(
                question["question_id"], round_number, "rerank", exc
            )
            rows.append(
                failed_row(
                    runtime,
                    question,
                    "C",
                    round_number,
                    "rerank",
                    exc,
                    shared["embedding_ms"],
                    shared["vector_ms"],
                    shared["bm25_ms"],
                    shared["rrf_ms"],
                    shared["rerank_ms"],
                )
            )
    return sorted(rows, key=lambda row: row["strategy"])


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_ms(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def format_recall(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def build_report(
    runtime: Runtime,
    run_dir: Path,
    formal_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    warmup_calls: dict[str, int],
    formal_calls: dict[str, int],
    elapsed_seconds: float,
) -> Path:
    overall = {
        row["strategy"]: row
        for row in summary_rows
        if row["category"] == "ALL"
    }
    categories = sorted({row["category"] for row in summary_rows if row["category"] != "ALL"})
    category_lookup = {
        (row["strategy"], row["category"]): row for row in summary_rows
    }
    differences = question_level_differences(formal_rows)
    failures = [
        row
        for row in formal_rows
        if row.get("recall_at_3") is None
    ]
    report = []
    report.append("# RAG 检索策略对比实验报告")
    report.append("")
    report.append("## 实验环境")
    report.append("")
    report.append(f"- 实验编号：`{run_dir.name}`")
    report.append(f"- 完成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}")
    report.append(f"- 正式运行总耗时：{elapsed_seconds:.2f} 秒")
    report.append(f"- Python：{platform.python_version()}")
    report.append(f"- 平台：{platform.platform()}")
    report.append(f"- LlamaIndex：`{__import__('importlib.metadata', fromlist=['version']).version('llama-index-core')}` / Python 3.11")
    report.append("")
    report.append("## 数据集与索引")
    report.append("")
    report.append(f"- 问题数：{runtime.preflight['question_count']}")
    report.append(f"- Dataset SHA256：`{runtime.preflight['dataset_sha256']}`")
    report.append(f"- Chroma 节点数：{runtime.preflight['chroma_count']}")
    report.append(f"- DocStore 节点数：{runtime.preflight['docstore_count']}")
    report.append(f"- 有效证据节点数：{runtime.preflight['expected_chunk_count']}")
    report.append("- Chroma 与 DocStore node_id 完全相同。")
    report.append("")
    report.append("## 固定参数")
    report.append("")
    report.append("| 参数 | 值 |")
    report.append("|---|---|")
    for key, value in runtime.parameters.items():
        report.append(f"| `{key}` | `{value}` |")
    report.append("")
    report.append("生产配置中的 `rerank_top_n` 仍为 4，未修改；本评测按任务要求将 Reranker 输出 Top 3。")
    report.append("")
    report.append("## 三组实现")
    report.append("")
    report.append("- A：Chroma 向量检索候选 15，按向量相似度取 Top 3。")
    report.append("- B：相同 Vector 15 + BM25 15，使用现有 `QueryFusionRetriever` 的 reciprocal rank fusion（k=60、双路等权），融合候选池 10，取 Top 3。")
    report.append("- C：与 B 完全复用同一份 Vector、BM25、RRF 候选池，再调用 `gte-rerank-v2`，取 Top 3。")
    report.append("- A/B/C 共用每轮一次 Query Embedding、向量检索、BM25 检索和 RRF 结果；C 额外调用一次 Rerank。")
    report.append("- `QueryFusionRetriever` 内部使用的 MockLLM 仅用于组件解析；`num_queries=1`，没有执行 LLM 请求或查询改写。")
    report.append("- 正式数据逐条验证：B、C 在全部 question-round 组合中的 RRF 候选 node_id 完全一致。")
    report.append("")
    report.append("## 整体 Recall@3")
    report.append("")
    report.append("| 策略 | 成功记录 | 失败记录 | Recall@3 |")
    report.append("|---|---:|---:|---:|")
    for strategy in STRATEGIES:
        row = overall[strategy]
        report.append(
            f"| {strategy} ({STRATEGIES[strategy]}) | {row['successful_records']} | "
            f"{row['failed_records']} | {format_recall(row['recall_at_3'])} |"
        )
    report.append("")
    report.append("## 五类问题 Recall@3")
    report.append("")
    report.append("| 类别 | A | B | C |")
    report.append("|---|---:|---:|---:|")
    for category in categories:
        report.append(
            f"| {category} | "
            f"{format_recall(category_lookup[('A', category)]['recall_at_3'])} | "
            f"{format_recall(category_lookup[('B', category)]['recall_at_3'])} | "
            f"{format_recall(category_lookup[('C', category)]['recall_at_3'])} |"
        )
    report.append("")
    report.append("## 题目级差异")
    report.append("")
    for name, values in differences.items():
        report.append(
            f"- `{name}`：提升 {values['win']} 题，下降 {values['loss']} 题，"
            f"持平 {values['tie']} 题，数据不完整 {values['incomplete']} 题。"
        )
    report.append("")
    report.append("## 耗时（ms）")
    report.append("")
    report.append("| 策略 | 平均 Total | P50 Total | P95 Total | Max Total | 平均 Embedding | 平均 Vector | 平均 BM25 | 平均 RRF | 平均 Rerank |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for strategy in STRATEGIES:
        row = overall[strategy]
        report.append(
            f"| {strategy} | {format_ms(row['avg_total_ms'])} | "
            f"{format_ms(row['p50_total_ms'])} | {format_ms(row['p95_total_ms'])} | "
            f"{format_ms(row['max_total_ms'])} | "
            f"{format_ms(row['avg_embedding_ms'])} | "
            f"{format_ms(row['avg_vector_ms'])} | "
            f"{format_ms(row['avg_bm25_ms'])} | "
            f"{format_ms(row['avg_rrf_ms'])} | "
            f"{format_ms(row['avg_rerank_ms'])} |"
        )
    report.append("")
    report.append("A 的 Total 为 Embedding + Vector；B 为 Embedding + Vector + BM25 + RRF；C 在 B 基础上增加 Rerank。")
    report.append("共享阶段在每次运行中只执行一次，未重复调用 API；这些组合总耗时用于策略级比较。")
    report.append("")
    report.append("## API 调用与失败")
    report.append("")
    report.append("| 阶段 | Warmup | Formal | Total |")
    report.append("|---|---:|---:|---:|")
    report.append(
        f"| Query Embedding | {warmup_calls.get('embedding', 0)} | "
        f"{formal_calls['embedding']} | "
        f"{warmup_calls.get('embedding', 0) + formal_calls['embedding']} |"
    )
    report.append(
        f"| DashScope Rerank | {warmup_calls.get('rerank', 0)} | "
        f"{formal_calls['rerank']} | "
        f"{warmup_calls.get('rerank', 0) + formal_calls['rerank']} |"
    )
    report.append(
        f"| Vector Chroma | {warmup_calls.get('vector', 0)} | "
        f"{formal_calls['vector']} | "
        f"{warmup_calls.get('vector', 0) + formal_calls['vector']} |"
    )
    report.append(
        f"| BM25 | {warmup_calls.get('bm25', 0)} | {formal_calls['bm25']} | "
        f"{warmup_calls.get('bm25', 0) + formal_calls['bm25']} |"
    )
    report.append(
        f"| RRF | {warmup_calls.get('rrf', 0)} | {formal_calls['rrf']} | "
        f"{warmup_calls.get('rrf', 0) + formal_calls['rrf']} |"
    )
    report.append("")
    report.append(f"- 正式运行失败记录：{len(failures)}")
    report.append(f"- API 失败次数：{runtime.call_counter.api_failures}")
    report.append("- LLM API 调用次数：0")
    warmup_api_errors = sum(
        error["round"] == "0"
        and error["stage"] in {"query_embedding", "rerank"}
        for error in runtime.call_counter.errors
    )
    formal_api_errors = sum(
        error["round"] != "0"
        and error["stage"] in {"query_embedding", "rerank"}
        for error in runtime.call_counter.errors
    )
    report.append(f"- Warmup API 失败次数：{warmup_api_errors}")
    report.append(f"- Formal API 失败次数：{formal_api_errors}")
    report.append("- 成本：当前任务未提供 Embedding/Reranker 价格，成本记为未知；未虚构金额。")
    report.append("- 索引构建成本：未重建索引，因此本次无法测得构建成本。")
    report.append("")
    report.append("## Reranker 收益与代价")
    report.append("")
    b = overall["B"]
    c = overall["C"]
    if b["recall_at_3"] is not None and c["recall_at_3"] is not None:
        report.append(
            f"- Rerank 后整体 Recall@3 变化：{c['recall_at_3'] - b['recall_at_3']:+.4f}。"
        )
    if c["avg_rerank_ms"] is not None:
        report.append(
            f"- 平均 Rerank 耗时：{c['avg_rerank_ms']:.3f} ms；"
            f"每次 C 记录额外一次 API 调用。"
        )
    report.append("")
    report.append("## 实验局限性")
    report.append("")
    report.append("- 只评估检索，不生成答案，不计算答案正确率或引用准确率。")
    report.append("- 远程 Embedding/Rerank 耗时受网络、限流和服务端波动影响。")
    report.append("- 当前只有五份产品说明书和 50 条问题，结论不应外推到其他领域。")
    report.append("- 生产配置 `rerank_top_n=4`，评测按任务要求使用 Top 3。")
    report.append("- API 错误不会按 Recall=0 计入；失败在原始记录和汇总中单独列出。")
    report.append("")
    report.append("## 结论与下一步")
    report.append("")
    report.append("- 以 `retrieval_summary.csv` 与 `retrieval_results.jsonl` 为最终数据源。")
    report.append("- 如 B/C 失败记录非零，应先补测相同题号和轮次，再比较 Reranker 收益。")
    report.append("- 后续答案评估应使用独立人工评分，不使用待测 LLM 自评。")
    report_path = run_dir / "retrieval_comparison_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Development-only question limit")
    parser.add_argument("--rounds", type=int, default=3, help="Formal rounds per question")
    parser.add_argument("--skip-warmup", action="store_true", help="Development-only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory; must be unused",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_jsonl(DATASET_PATH)
    runtime = build_runtime(records)
    print("PREFLIGHT", json.dumps(runtime.preflight, ensure_ascii=False))
    print("PARAMETERS", json.dumps(runtime.parameters, ensure_ascii=False))
    if args.preflight_only:
        print("preflight_only: ok")
        return 0

    selected = runtime.records[: args.limit] if args.limit else runtime.records
    if args.rounds < 1:
        raise ValueError("--rounds must be >= 1")
    if args.output_dir:
        run_dir = args.output_dir
        if run_dir.exists():
            raise FileExistsError(f"output directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    else:
        run_dir = make_run_dir()

    results_path = run_dir / "retrieval_results.jsonl"
    warmup_path = run_dir / "retrieval_warmup.jsonl"
    results_path.touch()
    warmup_path.touch()
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "created_at": datetime.now().astimezone().isoformat(),
                "preflight": runtime.preflight,
                "parameters": runtime.parameters,
                "question_limit": args.limit,
                "formal_rounds": args.rounds,
                "warmup_enabled": not args.skip_warmup,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    warmup_start = runtime.call_counter.__dict__.copy()
    if not args.skip_warmup:
        for index, question in enumerate(selected, start=1):
            rows = run_round(runtime, question, 0)
            append_jsonl(warmup_path, rows)
            print(f"WARMUP {index}/{len(selected)} {question['question_id']}")
    warmup_end = runtime.call_counter.__dict__.copy()
    warmup_calls = {
        key: warmup_end.get(key, 0) - warmup_start.get(key, 0)
        for key in ("embedding_calls", "vector_calls", "bm25_calls", "rrf_calls", "rerank_calls")
    }
    warmup_calls = {
        {
            "embedding_calls": "embedding",
            "vector_calls": "vector",
            "bm25_calls": "bm25",
            "rrf_calls": "rrf",
            "rerank_calls": "rerank",
        }[key]: value
        for key, value in warmup_calls.items()
    }

    started = time.perf_counter()
    formal_count_start = asdict(runtime.call_counter)
    formal_rows: list[dict[str, Any]] = []
    for round_number in range(1, args.rounds + 1):
        for index, question in enumerate(selected, start=1):
            rows = run_round(runtime, question, round_number)
            formal_rows.extend(rows)
            append_jsonl(results_path, rows)
            print(
                f"ROUND {round_number}/{args.rounds} "
                f"{index}/{len(selected)} {question['question_id']}"
            )
    elapsed_seconds = time.perf_counter() - started
    formal_count_end = asdict(runtime.call_counter)
    formal_calls = {
        "embedding": formal_count_end["embedding_calls"]
        - formal_count_start["embedding_calls"],
        "vector": formal_count_end["vector_calls"]
        - formal_count_start["vector_calls"],
        "bm25": formal_count_end["bm25_calls"]
        - formal_count_start["bm25_calls"],
        "rrf": formal_count_end["rrf_calls"]
        - formal_count_start["rrf_calls"],
        "rerank": formal_count_end["rerank_calls"]
        - formal_count_start["rerank_calls"],
    }
    shared_pool_verified = True
    by_question_round: dict[tuple[str, int], dict[str, list[str]]] = {}
    for row in formal_rows:
        key = (row["question_id"], row["round"])
        by_question_round.setdefault(key, {})[row["strategy"]] = row[
            "rrf_candidate_ids"
        ]
    for pools in by_question_round.values():
        if "B" in pools and "C" in pools:
            shared_pool_verified &= pools["B"] == pools["C"]
    if not shared_pool_verified:
        raise AssertionError("B and C RRF candidate pools differ")

    summary_rows = write_summary_csv(
        formal_rows, run_dir / "retrieval_summary.csv"
    )
    write_per_question_csv(
        formal_rows, run_dir / "retrieval_per_question.csv"
    )
    (run_dir / "retrieval_errors.jsonl").touch()
    errors = [row for row in formal_rows if row.get("recall_at_3") is None]
    append_jsonl(run_dir / "retrieval_errors.jsonl", errors)
    report_path = build_report(
        runtime,
        run_dir,
        formal_rows,
        summary_rows,
        warmup_calls,
        formal_calls,
        elapsed_seconds,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "created_at": datetime.now().astimezone().isoformat(),
                "preflight": runtime.preflight,
                "parameters": runtime.parameters,
                "question_limit": args.limit,
                "formal_rounds": args.rounds,
                "warmup_enabled": not args.skip_warmup,
                "formal_record_count": len(formal_rows),
                "formal_failure_count": len(errors),
                "warmup_calls": warmup_calls,
                "formal_calls": formal_calls,
                "shared_rrf_candidate_pool_verified": shared_pool_verified,
                "total_call_counter": asdict(runtime.call_counter),
                "elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("RESULTS_DIR", run_dir)
    print("FORMAL_RECORDS", len(formal_rows))
    print("FORMAL_FAILURES", len(errors))
    print("REPORT", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
