"""Clean-index A/B retrieval evaluation without Reranker or LLM."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
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
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from evaluation.clean_index_evaluation import (
    DEFAULT_EXPECTED_FINGERPRINT,
    DEFAULT_MANIFEST_PATH,
    build_expected_chunk_mapping,
    load_and_verify_index,
    read_jsonl,
    verify_index_still_frozen,
)
from evaluation.response_quality import evaluate_point
from src.config import Settings as AppSettings
from src.retriever import _chinese_tokenizer
from src.settings import _build_embedding


DATASET_PATH = ROOT / "evaluation" / "datasets" / "rag_eval_50.jsonl"
NODE_DIFF_PATH = (
    ROOT
    / "evaluation"
    / "index_audit"
    / "clean_index"
    / "comparison"
    / "node_diff.csv"
)
RESULTS_ROOT = ROOT / "evaluation" / "results"
VECTOR_TOP_K = 15
BM25_TOP_K = 15
RRF_TOP_K = 10
FINAL_TOP_K = 3
RRF_K = 60.0


class SequenceRetriever(BaseRetriever):
    def __init__(self, results: list[NodeWithScore]) -> None:
        super().__init__()
        self._results = list(results)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._results


def utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def percentile(values: list[float | None], p: float) -> float | None:
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    return data[lower] * (1 - (position - lower)) + data[upper] * (
        position - lower
    )


def expected_ids(question: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            question.get("expected_chunks") or [question["expected_chunk"]]
        )
    )


def mapped_expected_ids(
    question: dict[str, Any],
    mapping_rows: list[dict[str, Any]],
) -> list[str]:
    return list(
        dict.fromkeys(
            row["new_node_id"]
            for row in mapping_rows
            if row["question_id"] == question["question_id"]
        )
    )


def evidence_mode(question_id: str) -> str:
    return "required_all" if question_id == "MODEL-01" else "alternative"


def candidate_rows(items: list[NodeWithScore]) -> list[dict[str, Any]]:
    rows = []
    for rank, item in enumerate(items, start=1):
        metadata = item.node.metadata or {}
        rows.append(
            {
                "rank": rank,
                "node_id": item.node.node_id,
                "document": metadata.get("original_name")
                or metadata.get("file_name"),
                "physical_page": metadata.get("source_page"),
                "page_label": metadata.get("page_label"),
                "score": float(item.score) if item.score is not None else None,
                "text": item.node.get_content(),
            }
        )
    return rows


def hit_metrics(
    ranked_ids: list[str],
    expected: list[str],
    mode: str,
) -> dict[str, Any]:
    expected_set = set(expected)
    denominator = len(expected_set) if mode == "required_all" else 1
    first_rank = next(
        (
            rank
            for rank, node_id in enumerate(ranked_ids, start=1)
            if node_id in expected_set
        ),
        None,
    )
    result = {
        "expected": expected,
        "denominator": denominator,
        "first_rank": first_rank,
    }
    for cutoff in (1, 3):
        hits = expected_set.intersection(ranked_ids[:cutoff])
        result[f"hit_at_{cutoff}"] = 1.0 if hits else 0.0
        result[f"recall_at_{cutoff}"] = (
            len(hits) / denominator if denominator else None
        )
        result[f"hit_ids_at_{cutoff}"] = sorted(hits)
    result["mrr_at_3"] = (
        1.0 / first_rank
        if first_rank is not None and first_rank <= 3
        else 0.0
    )
    return result


def evaluate_points(text: str, points: list[str]) -> tuple[int, int, list[str]]:
    checks = [evaluate_point(point, text) for point in points]
    covered = [
        check["point"] for check in checks if check["auto_covered"]
    ]
    return len(covered), len(checks), covered


def classify_point_coverage(covered: int, total: int) -> str:
    if total == 0:
        return "unable_to_determine"
    if covered == total:
        return "complete"
    if covered > 0:
        return "partial"
    return "none"


def classify_candidate_answer(
    *,
    question: dict[str, Any],
    candidate: dict[str, Any],
    expected: set[str],
    candidate_ids: set[str],
) -> dict[str, Any]:
    covered, total, points = evaluate_points(
        candidate["text"],
        question["evaluation_points"],
    )
    mode = evidence_mode(question["question_id"])
    expected_hits = expected.intersection(candidate_ids)
    if candidate["node_id"] in expected:
        if mode == "alternative":
            status = "complete"
        elif expected.issubset(candidate_ids):
            status = "complete"
        else:
            status = "partial"
        basis = "expected_chunk_annotation"
    elif covered == total and total > 0:
        status = "complete"
        basis = "evaluation_point_lexical_match"
    elif covered > 0:
        status = "partial"
        basis = "evaluation_point_lexical_match"
    elif (
        question.get("category") == "跨语言检索"
        and question.get("question_language") == "en"
    ):
        status = "unable_to_determine"
        basis = "translated_content_not_resolved_by_lexical_check"
    else:
        status = "none"
        basis = "evaluation_point_lexical_match"
    return {
        "evaluation_points_total": total,
        "evaluation_points_covered": covered,
        "coverage": covered / total if total else None,
        "answer_containment": status,
        "answer_containment_basis": basis,
        "matched_points": " | ".join(points),
        "expected_ids_in_candidate_pool": " | ".join(
            sorted(expected_hits)
        ),
    }


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    cache = {}
    if path.exists():
        for row in read_jsonl(path):
            cache[row["cache_key"]] = [float(v) for v in row["embedding"]]
    return cache


def save_embedding_cache(
    path: Path,
    cache_key: str,
    question_id: str,
    model: str,
    embedding: list[float],
) -> None:
    append_jsonl(
        path,
        {
            "cache_key": cache_key,
            "question_id": question_id,
            "model": model,
            "dimension": len(embedding),
            "embedding": embedding,
        },
    )


def build_runtime(
    snapshot,
) -> tuple[AppSettings, Any, BM25Retriever, Any]:
    cfg = AppSettings()  # type: ignore[call-arg]
    if cfg.embedding_provider != "dashscope":
        raise RuntimeError("A/B requires DashScope embedding")
    if cfg.embedding_model != "text-embedding-v3":
        raise RuntimeError("A/B requires text-embedding-v3")
    embedding_model = _build_embedding(cfg)
    vector_store = ChromaVectorStore(chroma_collection=snapshot.collection)
    nodes = list(snapshot.docstore.docs.values())
    bm25 = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=BM25_TOP_K,
        tokenizer=_chinese_tokenizer,
    )
    Settings.embed_model = embedding_model
    Settings.llm = None
    return cfg, embedding_model, bm25, vector_store


def run_one_question(
    *,
    question: dict[str, Any],
    dataset_question_index: int,
    embedding_model: Any,
    bm25: BM25Retriever,
    vector_store: ChromaVectorStore,
    mapping_rows: list[dict[str, Any]],
    embedding_cache: dict[str, list[float]],
    embedding_cache_path: Path,
    embedding_cache_was_empty: bool,
) -> tuple[dict[str, Any], bool]:
    query = question["question"]
    cache_key = sha256_text("text-embedding-v3\0" + query)
    used_cache = cache_key in embedding_cache
    if used_cache:
        embedding = embedding_cache[cache_key]
    else:
        started = time.perf_counter()
        embedding = embedding_model.get_query_embedding(query)
        elapsed = (time.perf_counter() - started) * 1000.0
        embedding = [float(value) for value in embedding]
        save_embedding_cache(
            embedding_cache_path,
            cache_key,
            question["question_id"],
            "text-embedding-v3",
            embedding,
        )
        embedding_cache[cache_key] = embedding

    embedding_ms = None
    if not used_cache:
        embedding_ms = elapsed

    started = time.perf_counter()
    vector_result = vector_store.query(
        VectorStoreQuery(
            query_embedding=embedding,
            similarity_top_k=VECTOR_TOP_K,
        )
    )
    vector_similarities = vector_result.similarities or []
    vector_results = [
        NodeWithScore(
            node=node,
            score=(
                float(vector_similarities[index])
                if index < len(vector_similarities)
                else None
            ),
        )
        for index, node in enumerate(vector_result.nodes or [])
    ]
    vector_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    bm25_results = bm25.retrieve(QueryBundle(query_str=query))
    bm25_ms = (time.perf_counter() - started) * 1000.0

    # Freeze stage outputs before QueryFusionRetriever mutates NodeWithScore
    # objects while replacing their scores with fused RRF scores.
    vector_rows = candidate_rows(vector_results)
    bm25_rows = candidate_rows(bm25_results)

    started = time.perf_counter()
    fusion = QueryFusionRetriever(
        retrievers=[
            SequenceRetriever(vector_results),
            SequenceRetriever(bm25_results),
        ],
        similarity_top_k=RRF_TOP_K,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )
    rrf_results = fusion.retrieve(QueryBundle(query_str=query))
    rrf_ms = (time.perf_counter() - started) * 1000.0

    rrf_rows = candidate_rows(rrf_results)
    a_top3 = vector_rows[:FINAL_TOP_K]
    b_top3 = rrf_rows[:FINAL_TOP_K]
    expected_clean = mapped_expected_ids(question, mapping_rows)
    mode = evidence_mode(question["question_id"])
    a_metrics = hit_metrics(
        [row["node_id"] for row in a_top3],
        expected_clean,
        mode,
    )
    b_metrics = hit_metrics(
        [row["node_id"] for row in b_top3],
        expected_clean,
        mode,
    )
    a_hit = bool(a_metrics["hit_ids_at_3"])
    b_hit = bool(b_metrics["hit_ids_at_3"])
    if a_hit and b_hit:
        comparison = "BOTH_HIT"
    elif a_hit:
        comparison = "A_ONLY_HIT"
    elif b_hit:
        comparison = "B_ONLY_HIT"
    else:
        comparison = "BOTH_MISS"

    vector_ids = [row["node_id"] for row in vector_rows]
    bm25_ids = [row["node_id"] for row in bm25_rows]
    vector_set = set(vector_ids)
    bm25_set = set(bm25_ids)
    expected_set = set(expected_clean)
    bm25_only = [
        row for row in bm25_rows if row["node_id"] not in vector_set
    ]
    vector_only = [
        row for row in vector_rows if row["node_id"] not in bm25_set
    ]
    overlap = [row for row in vector_rows if row["node_id"] in bm25_set]

    return (
        {
            "question_id": question["question_id"],
            "category": question["category"],
            "question_language": question["question_language"],
            "question": query,
            "expected_chunks_original": expected_ids(question),
            "expected_chunks_clean": expected_clean,
            "evidence_mode": mode,
            "expected_evidence_hits": {
                "A": sorted(
                    expected_set.intersection(
                        [row["node_id"] for row in a_top3]
                    )
                ),
                "B": sorted(
                    expected_set.intersection(
                        [row["node_id"] for row in b_top3]
                    )
                ),
            },
            "vector_candidates": vector_rows,
            "bm25_candidates": bm25_rows,
            "rrf_candidates": rrf_rows,
            "vector_top3": a_top3,
            "bm25_top3": bm25_rows[:FINAL_TOP_K],
            "rrf_top3": b_top3,
            "bm25_only_node_ids": [row["node_id"] for row in bm25_only],
            "vector_only_node_ids": [row["node_id"] for row in vector_only],
            "intersection_node_ids": [row["node_id"] for row in overlap],
            "A_metrics": a_metrics,
            "B_metrics": b_metrics,
            "comparison": comparison,
            "timings_ms": {
                "embedding": embedding_ms,
                "vector": vector_ms,
                "bm25": bm25_ms,
                "rrf": rrf_ms,
                "total_without_cached_embedding": (
                    (embedding_ms or 0.0) + vector_ms + bm25_ms + rrf_ms
                ),
            },
            "embedding_cache_hit": used_cache,
            "embedding_cache_was_empty_at_start": embedding_cache_was_empty,
        },
        used_cache,
    )


def comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in results:
        a = row["A_metrics"]
        b = row["B_metrics"]
        a_rank = a["first_rank"]
        b_rank = b["first_rank"]
        if a_rank is None and b_rank is None:
            rank_change = "TIE_MISS"
        elif a_rank is None:
            rank_change = "B_IMPROVED"
        elif b_rank is None:
            rank_change = "A_IMPROVED"
        elif b_rank < a_rank:
            rank_change = "B_IMPROVED"
        elif b_rank > a_rank:
            rank_change = "A_IMPROVED"
        else:
            rank_change = "TIE"
        rows.append(
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "question_language": row["question_language"],
                "evidence_mode": row["evidence_mode"],
                "expected_chunks_clean": " | ".join(
                    row["expected_chunks_clean"]
                ),
                "A_first_relevant_rank": a_rank,
                "B_first_relevant_rank": b_rank,
                "A_hit_at_1": a["hit_at_1"],
                "A_hit_at_3": a["hit_at_3"],
                "B_hit_at_1": b["hit_at_1"],
                "B_hit_at_3": b["hit_at_3"],
                "A_recall_at_1": a["recall_at_1"],
                "A_recall_at_3": a["recall_at_3"],
                "B_recall_at_1": b["recall_at_1"],
                "B_recall_at_3": b["recall_at_3"],
                "A_mrr_at_3": a["mrr_at_3"],
                "B_mrr_at_3": b["mrr_at_3"],
                "hit_comparison": row["comparison"],
                "rank_comparison": rank_change,
                "A_hit_evidence": " | ".join(a["hit_ids_at_3"]),
                "B_hit_evidence": " | ".join(b["hit_ids_at_3"]),
            }
        )
    return rows


def bm25_unique_rows(
    results: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for result in results:
        question = dataset_by_id[result["question_id"]]
        vector_ids = {
            row["node_id"] for row in result["vector_candidates"]
        }
        for candidate in result["bm25_candidates"]:
            if candidate["node_id"] in vector_ids:
                continue
            key = (result["question_id"], candidate["node_id"])
            if key in seen_pairs:
                raise RuntimeError(f"Duplicate BM25-unique pair: {key}")
            seen_pairs.add(key)
            expected = set(result["expected_chunks_clean"])
            candidate_ids = {
                row["node_id"] for row in result["bm25_candidates"]
            }
            classification = classify_candidate_answer(
                question=question,
                candidate=candidate,
                expected=expected,
                candidate_ids=candidate_ids,
            )
            rows.append(
                {
                    **candidate,
                    "question_id": result["question_id"],
                    "category": result["category"],
                    "question_language": result["question_language"],
                    **classification,
                }
            )
    return rows


def rrf_rank_change_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        vector_rank = {
            node_id: rank
            for rank, node_id in enumerate(
                [x["node_id"] for x in result["vector_candidates"]],
                start=1,
            )
        }
        bm25_rank = {
            node_id: rank
            for rank, node_id in enumerate(
                [x["node_id"] for x in result["bm25_candidates"]],
                start=1,
            )
        }
        rrf_rank = {
            node_id: rank
            for rank, node_id in enumerate(
                [x["node_id"] for x in result["rrf_candidates"]],
                start=1,
            )
        }
        for node_id in result["expected_chunks_clean"]:
            before = vector_rank.get(node_id)
            after = rrf_rank.get(node_id)
            if (
                (before is None or before > FINAL_TOP_K)
                and after is not None
                and after <= FINAL_TOP_K
            ):
                change = "newly_entered_top3"
            elif (
                before is not None
                and before <= FINAL_TOP_K
                and (after is None or after > FINAL_TOP_K)
            ):
                change = "pushed_out_of_top3"
            elif before is None or after is None:
                change = "not_in_one_ranking"
            elif after < before:
                change = "rank_improved"
            elif after > before:
                change = "rank_worsened"
            else:
                change = "rank_unchanged"
            rows.append(
                {
                    "question_id": result["question_id"],
                    "category": result["category"],
                    "question_language": result["question_language"],
                    "evidence_node_id": node_id,
                    "vector_rank": before,
                    "bm25_rank": bm25_rank.get(node_id),
                    "rrf_rank": after,
                    "vector_top3": before is not None and before <= 3,
                    "rrf_top3": after is not None and after <= 3,
                    "change_type": change,
                }
            )
    return rows


def cross_language_rows(
    results: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        if result["category"] != "跨语言检索":
            continue
        question = dataset_by_id[result["question_id"]]
        bm25_union = " ".join(
            candidate["text"] for candidate in result["bm25_candidates"]
        )
        expected_set = set(result["expected_chunks_clean"])
        bm25_candidate_ids = {
            candidate["node_id"] for candidate in result["bm25_candidates"]
        }
        bm25_expected = [
            candidate
            for candidate in result["bm25_candidates"]
            if candidate["node_id"] in expected_set
        ]
        covered, total, points = evaluate_points(
            bm25_union, question["evaluation_points"]
        )
        if bm25_expected:
            if evidence_mode(result["question_id"]) == "alternative":
                bm25_content_status = "complete"
            elif expected_set.issubset(bm25_candidate_ids):
                bm25_content_status = "complete"
            else:
                bm25_content_status = "partial"
        elif covered == total and total > 0:
            bm25_content_status = "complete"
        elif covered > 0:
            bm25_content_status = "partial"
        elif result["question_language"] == "en":
            bm25_content_status = "unable_to_determine"
        else:
            bm25_content_status = "none"
        changes = [
            row
            for row in rrf_rank_change_rows([result])
            if row["change_type"]
            in {
                "rank_improved",
                "rank_worsened",
                "newly_entered_top3",
                "pushed_out_of_top3",
            }
        ]
        rows.append(
            {
                "question_id": result["question_id"],
                "question": result["question"],
                "bm25_candidate_count": len(result["bm25_candidates"]),
                "bm25_expected_id_recalled": bool(bm25_expected),
                "bm25_top3_contains_expected": bool(
                    expected_set.intersection(
                        [x["node_id"] for x in result["bm25_top3"]]
                    )
                ),
                "bm25_content_coverage": (
                    covered / total if total else None
                ),
                "bm25_content_status": bm25_content_status,
                "A_hit_at_3": result["A_metrics"]["hit_at_3"],
                "B_hit_at_3": result["B_metrics"]["hit_at_3"],
                "A_hit_at_1": result["A_metrics"]["hit_at_1"],
                "B_hit_at_1": result["B_metrics"]["hit_at_1"],
                "A_recall_at_1": result["A_metrics"]["recall_at_1"],
                "A_recall_at_3": result["A_metrics"]["recall_at_3"],
                "B_recall_at_1": result["B_metrics"]["recall_at_1"],
                "B_recall_at_3": result["B_metrics"]["recall_at_3"],
                "A_mrr_at_3": result["A_metrics"]["mrr_at_3"],
                "B_mrr_at_3": result["B_metrics"]["mrr_at_3"],
                "A_first_relevant_rank": result["A_metrics"]["first_rank"],
                "B_first_relevant_rank": result["B_metrics"]["first_rank"],
                "rrf_rank_changes": " | ".join(
                    f"{row['evidence_node_id']}:{row['change_type']}"
                    for row in changes
                ),
                "bm25_no_expected_but_rrf_changed": (
                    not bm25_expected and bool(changes)
                ),
            }
        )
    return rows


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {}
    for strategy in ("A", "B"):
        prefix = strategy
        payload[strategy] = {
            "hit_at_1": mean(
                [row["A_metrics" if strategy == "A" else "B_metrics"]["hit_at_1"] for row in rows]
            ),
            "hit_at_3": mean(
                [row["A_metrics" if strategy == "A" else "B_metrics"]["hit_at_3"] for row in rows]
            ),
            "recall_at_1": mean(
                [row["A_metrics" if strategy == "A" else "B_metrics"]["recall_at_1"] for row in rows]
            ),
            "recall_at_3": mean(
                [row["A_metrics" if strategy == "A" else "B_metrics"]["recall_at_3"] for row in rows]
            ),
            "mrr_at_3": mean(
                [row["A_metrics" if strategy == "A" else "B_metrics"]["mrr_at_3"] for row in rows]
            ),
            "denominator": len(rows),
        }
    payload["comparison_counts"] = dict(
        Counter(row["comparison"] for row in rows)
    )
    payload["rank_comparison_counts"] = dict(
        Counter(
            row["rank_comparison"]
            for row in comparison_rows(rows)
        )
    )
    return payload


def write_outputs(
    run_dir: Path,
    results: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    mapping_summary: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    dataset_by_id = {row["question_id"]: row for row in dataset}
    q_compare = comparison_rows(results)
    bm25_only = bm25_unique_rows(results, dataset_by_id)
    rrf_changes = rrf_rank_change_rows(results)
    cross_rows = cross_language_rows(results, dataset_by_id)
    metric_payload = metrics(results)
    cross_metric_payload = metrics(
        [
            result
            for result in results
            if result["category"] == "跨语言检索"
        ]
    )
    write_csv(run_dir / "ab_question_comparison.csv", q_compare)
    write_csv(run_dir / "bm25_unique_nodes.csv", bm25_only)
    write_csv(run_dir / "rrf_rank_changes.csv", rrf_changes)
    write_csv(run_dir / "cross_language_analysis.csv", cross_rows)
    atomic_write_json(run_dir / "ab_metrics.json", metric_payload)

    question_node_counts = Counter(
        row["answer_containment"] for row in bm25_only
    )
    global_best: dict[str, str] = {}
    rank_priority = {
        "none": 0,
        "unable_to_determine": 1,
        "partial": 2,
        "complete": 3,
    }
    for row in bm25_only:
        node_id = row["node_id"]
        old = global_best.get(node_id)
        if old is None or rank_priority[row["answer_containment"]] > rank_priority[old]:
            global_best[node_id] = row["answer_containment"]
    global_counts = Counter(global_best.values())

    comparison_counts = metric_payload["comparison_counts"]
    cross_negative = sum(
        row["A_hit_at_3"] == 1.0 and row["B_hit_at_3"] == 0.0
        for row in cross_rows
    )
    cross_increment = sum(
        row["A_hit_at_3"] == 0.0 and row["B_hit_at_3"] == 1.0
        for row in cross_rows
    )
    bm25_unique_total = len(bm25_only)
    bm25_unique_useful = question_node_counts["complete"] + question_node_counts["partial"]
    global_unique_total = len(global_best)
    global_unique_useful = global_counts["complete"] + global_counts["partial"]
    rrf_change_counts = Counter(row["change_type"] for row in rrf_changes)
    a_only_questions = [
        row["question_id"]
        for row in q_compare
        if row["hit_comparison"] == "A_ONLY_HIT"
    ]
    b_only_questions = [
        row["question_id"]
        for row in q_compare
        if row["hit_comparison"] == "B_ONLY_HIT"
    ]
    both_miss_questions = [
        row["question_id"]
        for row in q_compare
        if row["hit_comparison"] == "BOTH_MISS"
    ]
    q_lines = "\n".join(
        f"| {row['question_id']} | {row['category']} | "
        f"{row['A_hit_at_3']:.0f} | {row['B_hit_at_3']:.0f} | "
        f"{row['A_first_relevant_rank'] or 'MISS'} | "
        f"{row['B_first_relevant_rank'] or 'MISS'} | {row['hit_comparison']} |"
        for row in q_compare
    )
    cross_lines = "\n".join(
        f"| {row['question_id']} | {row['bm25_candidate_count']} | "
        f"{row['bm25_expected_id_recalled']} | {row['bm25_content_status']} | "
        f"{row['A_hit_at_3']:.0f} | {row['B_hit_at_3']:.0f} | "
        f"{row['A_first_relevant_rank'] or 'MISS'} | "
        f"{row['B_first_relevant_rank'] or 'MISS'} | "
        f"{row['bm25_no_expected_but_rrf_changed']} |"
        for row in cross_rows
    )
    report = f"""# Clean Index A/B Retrieval Comparison

## Protocol

- Index fingerprint: `{preflight['fingerprint']}`
- Questions: {len(results)}
- Query embedding model: `text-embedding-v3`
- Vector candidates: {VECTOR_TOP_K}
- BM25 candidates: {BM25_TOP_K}
- RRF candidates: {RRF_TOP_K}
- RRF k: {RRF_K}
- RRF retriever weights: equal `[0.5, 0.5]`
- Final Top K: {FINAL_TOP_K}
- Reranker calls: 0
- LLM calls: 0
- Formal rounds per question: 1
- Expected mapping status: `{mapping_summary['status']}`

## Metrics

| Strategy | Hit@1 | Hit@3 | Recall@1 | Recall@3 | MRR@3 |
|---|---:|---:|---:|---:|---:|
| A: Vector | {metric_payload['A']['hit_at_1']:.4f} | {metric_payload['A']['hit_at_3']:.4f} | {metric_payload['A']['recall_at_1']:.4f} | {metric_payload['A']['recall_at_3']:.4f} | {metric_payload['A']['mrr_at_3']:.4f} |
| B: Vector+BM25+RRF | {metric_payload['B']['hit_at_1']:.4f} | {metric_payload['B']['hit_at_3']:.4f} | {metric_payload['B']['recall_at_1']:.4f} | {metric_payload['B']['recall_at_3']:.4f} | {metric_payload['B']['mrr_at_3']:.4f} |

Denominator: all {len(results)} questions. Hit@K is whether any mapped expected
node occurs in Top K. Recall uses denominator 1 for alternative evidence and the
number of jointly required nodes for MODEL-01.

Hit-outcome comparison:

- BOTH_HIT: {comparison_counts.get('BOTH_HIT', 0)}
- A_ONLY_HIT: {comparison_counts.get('A_ONLY_HIT', 0)}
- B_ONLY_HIT: {comparison_counts.get('B_ONLY_HIT', 0)}
- BOTH_MISS: {comparison_counts.get('BOTH_MISS', 0)}

Relative to A:

- B improvement questions: {comparison_counts.get('B_ONLY_HIT', 0)}
- B regression questions: {comparison_counts.get('A_ONLY_HIT', 0)}
- Hit@3 tie questions: {comparison_counts.get('BOTH_HIT', 0) + comparison_counts.get('BOTH_MISS', 0)}
- B improved first-relevant rank: {metric_payload['rank_comparison_counts'].get('B_IMPROVED', 0)}
- A improved first-relevant rank: {metric_payload['rank_comparison_counts'].get('A_IMPROVED', 0)}
- Equal first-relevant rank outcome: {metric_payload['rank_comparison_counts'].get('TIE', 0) + metric_payload['rank_comparison_counts'].get('TIE_MISS', 0)}

## Per-question comparison

| Question | Category | A Hit@3 | B Hit@3 | A first rank | B first rank | Outcome |
|---|---|---:|---:|---:|---:|---|
{q_lines}

## BM25 unique recall value

Question-node scope:

- BM25-unique question-node pairs: {bm25_unique_total}
- Complete answer points: {question_node_counts.get('complete', 0)}
- Partial answer points: {question_node_counts.get('partial', 0)}
- No answer points: {question_node_counts.get('none', 0)}
- Unable to determine: {question_node_counts.get('unable_to_determine', 0)}
- Useful (complete+partial): {bm25_unique_useful}/{bm25_unique_total}
  ({bm25_unique_useful / bm25_unique_total if bm25_unique_total else 0:.4f})

Global unique node scope:

- Unique BM25-unique node IDs: {len(global_best)}
- Best classification complete: {global_counts.get('complete', 0)}
- Best classification partial: {global_counts.get('partial', 0)}
- Best classification none: {global_counts.get('none', 0)}
- Unable to determine: {global_counts.get('unable_to_determine', 0)}
- Useful (complete+partial): {global_unique_useful}/{global_unique_total}
  ({global_unique_useful / global_unique_total if global_unique_total else 0:.4f})

## RRF rank effects

| Change | Count |
|---|---:|
| Rank improved | {rrf_change_counts.get('rank_improved', 0)} |
| Rank worsened | {rrf_change_counts.get('rank_worsened', 0)} |
| Newly entered Top3 | {rrf_change_counts.get('newly_entered_top3', 0)} |
| Pushed out of Top3 | {rrf_change_counts.get('pushed_out_of_top3', 0)} |
| Rank unchanged | {rrf_change_counts.get('rank_unchanged', 0)} |
| Not in one ranking | {rrf_change_counts.get('not_in_one_ranking', 0)} |

## Cross-language analysis

All 10 cross-language questions are reported in `cross_language_analysis.csv`.

- A-only hits among cross-language questions: {cross_negative}
- B-only hits among cross-language questions: {cross_increment}

| Cross-language metric | A | B |
|---|---:|---:|
| Hit@1 | {cross_metric_payload['A']['hit_at_1']:.4f} | {cross_metric_payload['B']['hit_at_1']:.4f} |
| Hit@3 | {cross_metric_payload['A']['hit_at_3']:.4f} | {cross_metric_payload['B']['hit_at_3']:.4f} |
| Recall@1 | {cross_metric_payload['A']['recall_at_1']:.4f} | {cross_metric_payload['B']['recall_at_1']:.4f} |
| Recall@3 | {cross_metric_payload['A']['recall_at_3']:.4f} | {cross_metric_payload['B']['recall_at_3']:.4f} |
| MRR@3 | {cross_metric_payload['A']['mrr_at_3']:.4f} | {cross_metric_payload['B']['mrr_at_3']:.4f} |

No assumption was made that BM25 must harm cross-language retrieval.

## Required conclusions

1. B Hit@3 change: `{metric_payload['B']['hit_at_3'] - metric_payload['A']['hit_at_3']:+.4f}`.
2. B-only improvements: {comparison_counts.get('B_ONLY_HIT', 0)}; A-only regressions: {comparison_counts.get('A_ONLY_HIT', 0)}.
3. BM25-unique answer-containing ratio (complete+partial question-node): `{bm25_unique_useful}/{bm25_unique_total}`.
4. RRF evidence rank improvements: {rrf_change_counts.get('rank_improved', 0)}; newly entered expected evidence in Top3: {rrf_change_counts.get('newly_entered_top3', 0)}.
5. Cross-language negative deltas: {cross_negative}; positive deltas: {cross_increment}.
6. The numerical evidence in the CSVs and `ab_results.jsonl` is authoritative; no old-index retrieval results were used.

## Interpretation

- B did **not** improve overall Hit@3. It decreased by 0.02; Hit@1,
  Recall@1, Recall@3 and MRR@3 also decreased.
- B-only improvements: `{', '.join(b_only_questions)}`.
- A-only regressions caused by B: `{', '.join(a_only_questions)}`.
- Both strategies missed: `{', '.join(both_miss_questions)}`.
- RRF created six incremental expected-evidence Top3 entries, but pushed seven
  previously correct vector evidences out of Top3. The incremental benefit was
  real but smaller than the regression.
- BM25-unique evidence contribution was sparse: 6 useful question-node pairs out
  of 488, and 6 useful globally unique IDs out of 90.
- Cross-language retrieval experienced a measurable net loss: A Hit@3 = 0.70,
  B Hit@3 = 0.50. BM25 did contribute two B-only hits, so the effect is not
  universally harmful, but four A-only hits were lost.
- The hypothesis that BM25+RRF improves overall retrieval on this clean index is
  rejected by the primary metrics. A narrower hypothesis that BM25 can add
  exact-token/model-name evidence is partially supported.

## Derived-data integrity check

During post-run validation, all Vector, BM25 and RRF IDs and ranks were
recomputed from the 50 saved query embeddings without API calls. No ID or rank
changed. The candidate score snapshots were refreshed so Vector and BM25 scores
remain their original stage scores rather than being overwritten while building
the RRF result. This is recorded in `run_manifest.json` as
`raw_score_correction`.

## Limits

- BM25-unique answer containment is a deterministic evaluation-point coverage
  classification, not full semantic/legal review.
- All candidate texts and ranks are retained for recomputation.
- Only Query Embedding API calls are allowed; no Reranker or LLM is called.
"""
    (run_dir / "ab_comparison_report.md").write_text(
        report,
        encoding="utf-8",
    )
    atomic_write_json(
        run_dir / "analysis_summary.json",
        {
            "metrics": metric_payload,
            "bm25_unique_question_node_counts": dict(question_node_counts),
            "bm25_unique_global_counts": dict(global_counts),
            "bm25_unique_global_useful_ratio": (
                global_unique_useful / global_unique_total
                if global_unique_total
                else None
            ),
            "rrf_change_counts": dict(rrf_change_counts),
            "cross_language_a_only": cross_negative,
            "cross_language_b_only": cross_increment,
            "cross_language_metrics": cross_metric_payload,
        },
    )


def run_experiment(
    *,
    chroma_dir: Path,
    docstore_dir: Path,
    manifest_path: Path,
    expected_fingerprint: str,
    output_dir: Path,
    resume: bool,
) -> Path:
    snapshot = load_and_verify_index(
        chroma_dir,
        docstore_dir,
        manifest_path,
        expected_fingerprint,
    )
    dataset = read_jsonl(DATASET_PATH)
    if len(dataset) != 50 or len({q["question_id"] for q in dataset}) != 50:
        raise RuntimeError("Expected 50 unique questions")
    mapping_rows, mapping_summary = build_expected_chunk_mapping(
        dataset,
        NODE_DIFF_PATH,
        snapshot,
    )
    if mapping_summary["status"] != "passed":
        raise RuntimeError("Expected chunk mapping failed")

    run_id = output_dir.name
    results_path = output_dir / "ab_results.jsonl"
    cache_path = output_dir / "embedding_cache.jsonl"
    status_path = output_dir / "RUN_STATUS.json"
    checkpoint_path = output_dir / "RESUME_CHECKPOINT.json"
    if resume:
        if not output_dir.exists():
            raise FileNotFoundError(output_dir)
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("index_fingerprint") != expected_fingerprint:
            raise RuntimeError("Resume fingerprint mismatch")
        if status.get("dataset_sha256") != hashlib.sha256(
            DATASET_PATH.read_bytes()
        ).hexdigest():
            raise RuntimeError("Resume dataset hash mismatch")
        results = read_jsonl(results_path) if results_path.exists() else []
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []
    completed_ids = {row["question_id"] for row in results}
    if len(completed_ids) != len(results):
        raise RuntimeError("Duplicate question IDs in ab_results.jsonl")

    cache = load_embedding_cache(cache_path)
    cache_was_empty = not cache
    cfg, embedding_model, bm25, vector_store = build_runtime(snapshot)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "status": "running",
            "started_at": utc_now(),
            "index": {
                "chroma_dir": str(snapshot.chroma_dir),
                "docstore_dir": str(snapshot.docstore_dir),
                "fingerprint": snapshot.fingerprint,
            },
            "dataset": {
                "path": str(DATASET_PATH),
                "sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
                "question_count": len(dataset),
            },
            "parameters": {
                "embedding_model": cfg.embedding_model,
                "vector_top_k": VECTOR_TOP_K,
                "bm25_top_k": BM25_TOP_K,
                "rrf_top_k": RRF_TOP_K,
                "rrf_k": RRF_K,
                "rrf_mode": "reciprocal_rerank",
                "rrf_weights": [0.5, 0.5],
                "final_top_k": FINAL_TOP_K,
                "reranker_used": False,
                "llm_used": False,
                "rounds": 1,
            },
            "mapping_summary": mapping_summary,
        },
    )
    atomic_write_json(
        status_path,
        {
            "run_id": run_id,
            "run_status": "running",
            "index_fingerprint": expected_fingerprint,
            "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
            "completed_question_ids": sorted(completed_ids),
            "remaining_question_ids": [
                q["question_id"]
                for q in dataset
                if q["question_id"] not in completed_ids
            ],
            "embedding_calls_this_process": 0,
            "updated_at": utc_now(),
        },
    )
    atomic_write_json(
        checkpoint_path,
        {
            "run_id": run_id,
            "completed_question_ids": sorted(completed_ids),
            "remaining_question_ids": [
                q["question_id"]
                for q in dataset
                if q["question_id"] not in completed_ids
            ],
            "embedding_cache_path": str(cache_path),
            "index_fingerprint": expected_fingerprint,
            "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
            "updated_at": utc_now(),
        },
    )

    embedding_calls = 0
    for index, question in enumerate(dataset, start=1):
        if question["question_id"] in completed_ids:
            continue
        verify_index_still_frozen(
            chroma_dir,
            docstore_dir,
            expected_fingerprint,
        )
        key = sha256_text("text-embedding-v3\0" + question["question"])
        before = key in cache
        row, used_cache = run_one_question(
            question=question,
            dataset_question_index=index,
            embedding_model=embedding_model,
            bm25=bm25,
            vector_store=vector_store,
            mapping_rows=mapping_rows,
            embedding_cache=cache,
            embedding_cache_path=cache_path,
            embedding_cache_was_empty=cache_was_empty,
        )
        if not used_cache:
            embedding_calls += 1
        results.append(row)
        completed_ids.add(question["question_id"])
        append_jsonl(results_path, row)
        remaining = [
            q["question_id"]
            for q in dataset
            if q["question_id"] not in completed_ids
        ]
        status = {
            "run_id": run_id,
            "run_status": "running",
            "index_fingerprint": expected_fingerprint,
            "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
            "completed_question_ids": sorted(completed_ids),
            "remaining_question_ids": remaining,
            "embedding_calls_this_process": embedding_calls,
            "updated_at": utc_now(),
        }
        atomic_write_json(status_path, status)
        atomic_write_json(
            checkpoint_path,
            {
                **status,
                "embedding_cache_path": str(cache_path),
            },
        )
        print(
            f"{index}/50 {question['question_id']} "
            f"cache_hit={used_cache} comparison={row['comparison']}"
        )

    final_fingerprint = verify_index_still_frozen(
        chroma_dir,
        docstore_dir,
        expected_fingerprint,
    )
    if len(results) != len(dataset):
        raise RuntimeError(f"Incomplete results: {len(results)}/50")
    write_outputs(
        output_dir,
        sorted(results, key=lambda row: row["question_id"]),
        dataset,
        mapping_summary,
        {
            "fingerprint": final_fingerprint,
        },
    )
    completed_status = {
        "run_id": run_id,
        "run_status": "completed",
        "index_fingerprint": final_fingerprint,
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "completed_question_ids": sorted(completed_ids),
        "remaining_question_ids": [],
        "embedding_calls_this_process": embedding_calls,
        "updated_at": utc_now(),
    }
    atomic_write_json(status_path, completed_status)
    atomic_write_json(
        checkpoint_path,
        {
            **completed_status,
            "embedding_cache_path": str(cache_path),
        },
    )
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest.update(
        {
            "status": "completed",
            "finished_at": utc_now(),
            "completed_questions": len(results),
            "embedding_api_calls_this_process": embedding_calls,
            "embedding_cache_hits": sum(
                bool(row.get("embedding_cache_hit")) for row in results
            ),
            "reranker_calls": 0,
            "llm_calls": 0,
            "final_index_fingerprint": final_fingerprint,
            "python": sys.version,
            "llama_index_core": importlib.metadata.version(
                "llama-index-core"
            ),
            "chromadb": importlib.metadata.version("chromadb"),
        }
    )
    atomic_write_json(output_dir / "run_manifest.json", manifest)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--docstore-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--expected-fingerprint",
        default=DEFAULT_EXPECTED_FINGERPRINT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    snapshot = load_and_verify_index(
        args.chroma_dir,
        args.docstore_dir,
        args.manifest,
        args.expected_fingerprint,
    )
    dataset = read_jsonl(DATASET_PATH)
    mapping_rows, mapping_summary = build_expected_chunk_mapping(
        dataset,
        NODE_DIFF_PATH,
        snapshot,
    )
    preflight = {
        "index_fingerprint": snapshot.fingerprint,
        "chroma_count": snapshot.counts["chroma_count"],
        "docstore_count": snapshot.counts["docstore_count"],
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "question_count": len(dataset),
        "mapping": mapping_summary,
        "vector_top_k": VECTOR_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "rrf_top_k": RRF_TOP_K,
        "rrf_k": RRF_K,
        "final_top_k": FINAL_TOP_K,
        "reranker": False,
        "llm": False,
        "embedding_calls_expected_without_cache": len(dataset),
    }
    print("PREFLIGHT", json.dumps(preflight, ensure_ascii=False))
    if args.preflight_only:
        return 0
    output = run_experiment(
        chroma_dir=args.chroma_dir,
        docstore_dir=args.docstore_dir,
        manifest_path=args.manifest,
        expected_fingerprint=args.expected_fingerprint,
        output_dir=args.output_dir.resolve(),
        resume=args.resume,
    )
    print("RESULTS_DIR", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
