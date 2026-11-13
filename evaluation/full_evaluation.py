"""Complete RAG evaluation using the audited 50-question dataset.

Retrieval metrics are derived from stage-3 results. Answer generation uses the
same C retrieval configuration and the project's existing chat prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.llms import ChatResponse
from pypdf import PdfReader

from evaluation.no_answer_cases import NO_ANSWER_CASES, verify_no_answer_cases
from evaluation.response_quality import (
    CITATION_RE,
    contains_refusal,
    evaluate_answer_quality,
    evaluate_citations,
    evaluate_faithfulness,
)
from evaluation.retrieval_benchmark import (
    Runtime,
    build_runtime,
    run_round,
    sanitize_error,
)
from evaluation.retrieval_extended_metrics import (
    build_retrieval_metrics,
    write_category_csv as write_retrieval_category_csv,
    write_per_question_csv as write_retrieval_question_csv,
)
from src.chat import DEFAULT_STYLE, _build_system_prompt
from src.config import get_settings
from src.settings import OpenAICompatible


DATASET_PATH = ROOT / "evaluation" / "datasets" / "rag_eval_50.jsonl"
STAGE3_RUN = ROOT / "evaluation" / "results" / "20261112_231115"
STAGE3_RESULTS = STAGE3_RUN / "retrieval_results.jsonl"
RESULTS_ROOT = ROOT / "evaluation" / "results"
EXPECTED_CATEGORIES = {
    "参数查询",
    "安装操作",
    "故障排查",
    "跨产品型号归属辨析",
    "跨语言检索",
}


class StaticRetriever(BaseRetriever):
    _results: list[NodeWithScore] = PrivateAttr(default_factory=list)

    def __init__(self, results: list[NodeWithScore]) -> None:
        super().__init__()
        self._results = list(results)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._results


class UsageCapturingLLM(OpenAICompatible):
    _captured_usage: dict[str, int] | None = PrivateAttr(default=None)
    _usage_seen: bool = PrivateAttr(default=False)
    _call_count: int = PrivateAttr(default=0)

    def reset_usage(self) -> None:
        self._captured_usage = None
        self._usage_seen = False

    def usage(self) -> dict[str, int] | None:
        return dict(self._captured_usage) if self._captured_usage else None

    def call_count(self) -> int:
        return self._call_count

    def _capture_usage(self, response: Any) -> None:
        additional = getattr(response, "additional_kwargs", None)
        if isinstance(additional, dict):
            prompt = additional.get("prompt_tokens")
            completion = additional.get("completion_tokens")
            total = additional.get("total_tokens")
            if prompt is not None or completion is not None or total is not None:
                self._captured_usage = {
                    "prompt_tokens": int(prompt or 0),
                    "completion_tokens": int(completion or 0),
                    "total_tokens": int(total or 0),
                }
                self._usage_seen = True
        raw = getattr(response, "raw", None)
        usage = getattr(raw, "usage", None)
        if usage is not None:
            self._captured_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(usage, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            self._usage_seen = True

    def chat(self, *args, **kwargs) -> ChatResponse:
        self._call_count += 1
        response = super().chat(*args, **kwargs)
        self._capture_usage(response)
        return response

    def stream_chat(self, *args, **kwargs):
        self._call_count += 1
        source = super().stream_chat(*args, **kwargs)

        def _wrapped():
            for chunk in source:
                self._capture_usage(chunk)
                yield chunk

        return _wrapped()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    return data[lower] * (1 - (position - lower)) + data[upper] * (position - lower)


def mean(values: list[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return sum(data) / len(data) if data else None


def sum_usage(
    first: dict[str, int] | None, second: dict[str, int] | None
) -> dict[str, int] | None:
    if first is None and second is None:
        return None
    first = first or {}
    second = second or {}
    return {
        key: int(first.get(key, 0)) + int(second.get(key, 0))
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def make_run_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = RESULTS_ROOT / f"full_evaluation_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = RESULTS_ROOT / f"full_evaluation_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def load_pdf_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted((ROOT / "data" / "uploads").glob("*.pdf")):
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        name = path.name.split("_", 2)[-1]
        texts[name] = " ".join(
            (page.extract_text() or "") for page in reader.pages
        )
    return texts


def load_document_node_texts() -> dict[str, str]:
    collection = chromadb.PersistentClient(
        path=str(ROOT / "storage" / "chroma")
    ).get_collection("knowledge_base")
    raw = collection.get(include=["metadatas", "documents"])
    grouped: dict[str, list[str]] = {}
    for metadata, document in zip(raw["metadatas"], raw["documents"]):
        name = (metadata or {}).get("original_name")
        if name:
            grouped.setdefault(name, []).append(document or "")
    return {name: " ".join(parts) for name, parts in grouped.items()}


def preflight(
    records: list[dict[str, Any]],
    stage3_rows: list[dict[str, Any]],
    runtime: Runtime,
    no_answer_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_ids = [record["question_id"] for record in records]
    if len(records) != 50 or len(set(dataset_ids)) != 50:
        raise ValueError("formal dataset is incomplete or question_id is not unique")
    categories = Counter(record["category"] for record in records)
    if set(categories) != EXPECTED_CATEGORIES or any(
        count != 10 for count in categories.values()
    ):
        raise ValueError(f"unexpected category distribution: {dict(categories)}")

    expected_keys = {
        (record["question_id"], strategy, round_number)
        for record in stage3_rows
        for strategy in ("A", "B", "C")
        for round_number in (1, 2, 3)
    }
    actual_keys = {
        (row["question_id"], row["strategy"], row["round"])
        for row in stage3_rows
    }
    if actual_keys != expected_keys:
        raise ValueError("stage-3 retrieval results are incomplete or duplicated")

    cfg = get_settings()
    expected_models = {
        "llm_model": "nvidia/nemotron-3.5-lightning:free",
        "embedding_model": "text-embedding-v3",
        "rerank_model": "gte-rerank-v2",
    }
    actual_models = {
        "llm_model": cfg.llm_model,
        "embedding_model": cfg.embedding_model,
        "rerank_model": cfg.rerank_model,
    }
    if actual_models != expected_models:
        raise ValueError(f"model configuration mismatch: {actual_models}")
    if not cfg.openai_api_key.get_secret_value().strip():
        raise ValueError("OPENAI_API_KEY is not configured")
    if not cfg.dashscope_api_key.strip():
        raise ValueError("DASHSCOPE_API_KEY is not configured")
    if len(no_answer_cases) != 10:
        raise ValueError("no-answer evaluation requires exactly 10 cases")

    return {
        "dataset_question_count": len(records),
        "stage3_record_count": len(stage3_rows),
        "stage3_expected_record_count": len(expected_keys),
        "category_counts": dict(sorted(categories.items())),
        "chroma_count": runtime.preflight["chroma_count"],
        "docstore_count": runtime.preflight["docstore_count"],
        "expected_chunk_count": runtime.preflight["expected_chunk_count"],
        "models": actual_models,
        "no_answer_case_count": len(no_answer_cases),
    }


def create_llm(cfg) -> UsageCapturingLLM:
    return UsageCapturingLLM(
        model=cfg.llm_model,
        api_key=cfg.openai_api_key.get_secret_value().strip(),
        api_base=cfg.openai_api_base,
        max_tokens=2048,
    )


def get_c_retrieval(
    runtime: Runtime, question: dict[str, Any], round_number: int
) -> dict[str, Any]:
    Settings.llm = None
    rows = run_round(runtime, question, round_number)
    c_row = next(row for row in rows if row["strategy"] == "C")
    return c_row


def answer_question(
    runtime: Runtime,
    question: dict[str, Any],
    llm: UsageCapturingLLM,
    docstore: dict[str, Any],
    round_number: int,
    is_no_answer: bool,
) -> dict[str, Any]:
    request_started = time.perf_counter()
    retrieval_question = dict(question)
    if is_no_answer:
        retrieval_question["expected_chunk"] = "__NO_ANSWER__"
        retrieval_question["expected_chunks"] = ["__NO_ANSWER__"]

    c_row = get_c_retrieval(runtime, retrieval_question, round_number)
    retrieved = c_row.get("top3") or []
    base = {
        "question_id": question["question_id"],
        "category": question["category"],
        "question": question["question"],
        "question_language": question.get("question_language", "zh"),
        "is_no_answer_case": is_no_answer,
        "retrieved": retrieved,
        "retrieved_node_ids": [hit["node_id"] for hit in retrieved],
        "embedding_ms": c_row.get("embedding_ms"),
        "vector_ms": c_row.get("vector_ms"),
        "bm25_ms": c_row.get("bm25_ms"),
        "rrf_ms": c_row.get("rrf_ms"),
        "rerank_ms": c_row.get("rerank_ms"),
        "retrieval_ms": c_row.get("total_ms"),
        "rrf_candidate_ids": c_row.get("rrf_candidate_ids") or [],
        "rerank_input_document_count": len(c_row.get("rrf_candidate_ids") or []),
        "rerank_input_characters": sum(
            len(docstore[node_id].get_content())
            for node_id in (c_row.get("rrf_candidate_ids") or [])
            if node_id in docstore
        ),
        "query_characters": len(question["question"]),
        "llm_model": llm.model,
        "llm_first_token_ms": None,
        "llm_generation_ms": None,
        "total_ms": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "llm_calls": 0,
        "llm_retry_used": False,
        "api_error_stage": None,
        "api_error_type": None,
        "api_error_message": None,
        "answer": "",
    }
    if c_row.get("recall_at_3") is None:
        base["api_error_stage"] = "retrieval_c"
        base["api_error_type"] = c_row.get("error_type")
        base["api_error_message"] = c_row.get("error_message")
        base["total_ms"] = (time.perf_counter() - request_started) * 1000.0
        return base
    if not retrieved:
        base["api_error_stage"] = "retrieval_c"
        base["api_error_type"] = "EmptyRetrieval"
        base["api_error_message"] = "C retrieval returned no nodes"
        base["total_ms"] = (time.perf_counter() - request_started) * 1000.0
        return base

    nodes = [
        NodeWithScore(
            node=docstore[hit["node_id"]],
            score=hit.get("score"),
        )
        for hit in retrieved
    ]
    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=StaticRetriever(nodes),
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
        system_prompt=_build_system_prompt(DEFAULT_STYLE),
        skip_condense=True,
    )
    llm.reset_usage()
    Settings.llm = llm
    generation_started = time.perf_counter()
    llm_calls_before = llm.call_count()
    first_token_ms = None
    chunks: list[str] = []
    stream_usage = None
    try:
        stream = engine.stream_chat(question["question"])
        for token in stream.response_gen:
            if token:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - generation_started) * 1000
                chunks.append(token)
        stream_usage = llm.usage()
    except Exception as exc:
        base["api_error_stage"] = "llm_generation"
        base["api_error_type"] = type(exc).__name__
        base["api_error_message"] = sanitize_error(exc, runtime.secrets)
    if base["api_error_stage"] is None and not chunks:
        try:
            llm.reset_usage()
            retry_response = engine.chat(question["question"])
            retry_text = str(retry_response).strip()
            if retry_text:
                chunks.append(retry_text)
                base["llm_retry_used"] = True
                stream_usage = sum_usage(stream_usage, llm.usage())
        except Exception as exc:
            base["api_error_stage"] = "llm_generation_retry"
            base["api_error_type"] = type(exc).__name__
            base["api_error_message"] = sanitize_error(exc, runtime.secrets)
    generation_ms = (time.perf_counter() - generation_started) * 1000.0
    usage = stream_usage if base["llm_retry_used"] else llm.usage()
    base.update(
        {
            "answer": "".join(chunks).strip(),
            "llm_first_token_ms": first_token_ms,
            "llm_generation_ms": generation_ms,
            "total_ms": (time.perf_counter() - request_started) * 1000.0,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "total_tokens": usage.get("total_tokens") if usage else None,
            "llm_calls": llm.call_count() - llm_calls_before,
        }
    )
    if not base["answer"] and base["api_error_stage"] is None:
        base["api_error_stage"] = "llm_generation"
        base["api_error_type"] = "EmptyAnswer"
        base["api_error_message"] = "LLM returned no non-empty tokens"
    return base


def timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = {}
    for field in (
        "embedding_ms",
        "vector_ms",
        "bm25_ms",
        "rrf_ms",
        "rerank_ms",
        "retrieval_ms",
        "llm_first_token_ms",
        "llm_generation_ms",
        "total_ms",
    ):
        values = [row.get(field) for row in rows if row.get(field) is not None]
        metrics[field] = {
            "count": len(values),
            "mean": mean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": max(values) if values else None,
        }
    return metrics


def write_performance(rows: list[dict[str, Any]], path: Path) -> None:
    groups = {
        "answer_50": rows,
        "no_answer_10": [],
        "all_60": rows,
    }
    fields = ["scope", "metric", "count", "mean_ms", "p50_ms", "p95_ms", "max_ms"]
    output = []
    for scope, group in groups.items():
        summary = timing_summary(group)
        for metric, values in summary.items():
            output.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "count": values["count"],
                    "mean_ms": values["mean"],
                    "p50_ms": values["p50"],
                    "p95_ms": values["p95"],
                    "max_ms": values["max"],
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def update_performance_scopes(
    answer_rows: list[dict[str, Any]],
    no_answer_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    groups = {
        "answer_50": answer_rows,
        "no_answer_10": no_answer_rows,
        "all_60": answer_rows + no_answer_rows,
    }
    fields = ["scope", "metric", "count", "mean_ms", "p50_ms", "p95_ms", "max_ms"]
    output = []
    for scope, rows in groups.items():
        for metric, values in timing_summary(rows).items():
            output.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "count": values["count"],
                    "mean_ms": values["mean"],
                    "p50_ms": values["p50"],
                    "p95_ms": values["p95"],
                    "max_ms": values["max"],
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)


def evaluate_answer_rows(
    answer_rows: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows = []
    citation_rows = []
    faithfulness_rows = []
    for row in answer_rows:
        question = dataset_by_id[row["question_id"]]
        error = row.get("api_error_message")
        quality = evaluate_answer_quality(question, row["answer"], error)
        citations = evaluate_citations(
            question, row["answer"], row.get("retrieved") or []
        )
        faithfulness = evaluate_faithfulness(
            row["answer"], row.get("retrieved") or []
        )
        base = {
            "question_id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "expected_answer": question["expected_answer"],
            "evaluation_points": question["evaluation_points"],
            "retrieved": row.get("retrieved") or [],
            "api_error": row.get("api_error_message"),
        }
        quality_rows.append({**base, **quality})
        citation_rows.append({**base, **citations})
        faithfulness_rows.append({**base, **faithfulness})
    return quality_rows, citation_rows, faithfulness_rows


def write_answer_quality_summary(
    quality_rows: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    rows = []
    groups = [("ALL", quality_rows)] + [
        (category, [row for row in quality_rows if row["category"] == category])
        for category in sorted({row["category"] for row in quality_rows})
    ]
    for category, group in groups:
        completed = [row for row in group if row["status"] != "indeterminate"]
        counts = Counter(row["status"] for row in group)
        total_points = sum(row["total_points"] for row in group)
        covered_points = sum(row["covered_points"] for row in group)
        rows.append(
            {
                "category": category,
                "questions": len(group),
                "completed": len(completed),
                "completely_correct": counts["completely_correct"],
                "partially_correct": counts["partially_correct"],
                "wrong": counts["wrong"],
                "indeterminate": counts["indeterminate"],
                "answer_accuracy_auto_preliminary": (
                    counts["completely_correct"] / len(completed)
                    if completed
                    else None
                ),
                "point_total": total_points,
                "point_covered": covered_points,
                "point_coverage_auto_preliminary": (
                    covered_points / total_points if total_points else None
                ),
                "review_method": "automatic_lexical_preliminary",
                "human_review_required": True,
            }
        )
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_citation_summary(
    citation_rows: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    rows = []
    groups = [("ALL", citation_rows)] + [
        (category, [row for row in citation_rows if row["category"] == category])
        for category in sorted({row["category"] for row in citation_rows})
    ]
    for category, group in groups:
        citation_count = sum(row["citation_count"] for row in group)
        supported = sum(row["supported_citation_count"] for row in group)
        completeness = [
            row["citation_completeness"]
            for row in group
            if row["citation_completeness"] is not None
        ]
        rows.append(
            {
                "category": category,
                "questions": len(group),
                "citation_count": citation_count,
                "supported_citation_count": supported,
                "citation_accuracy": supported / citation_count if citation_count else None,
                "citation_completeness": mean(completeness),
                "review_method": "automatic_lexical_preliminary",
                "human_review_required": True,
            }
        )
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_faithfulness_summary(
    faithfulness_rows: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    rows = []
    groups = [("ALL", faithfulness_rows)] + [
        (
            category,
            [row for row in faithfulness_rows if row["category"] == category],
        )
        for category in sorted({row["category"] for row in faithfulness_rows})
    ]
    for category, group in groups:
        factual = sum(row["factual_statement_count"] for row in group)
        supported = sum(row["supported_statement_count"] for row in group)
        partial = sum(row["partially_supported_statement_count"] for row in group)
        unsupported = sum(row["unsupported_statement_count"] for row in group)
        conflict = sum(row["conflicting_statement_count"] for row in group)
        grounded = sum(row["grounded_answer"] for row in group)
        rows.append(
            {
                "category": category,
                "questions": len(group),
                "grounded_answers": grounded,
                "grounded_answer_ratio": grounded / len(group) if group else None,
                "factual_statements": factual,
                "supported_statements": supported,
                "partially_supported_statements": partial,
                "unsupported_statements": unsupported,
                "conflicting_statements": conflict,
                "supported_ratio": supported / factual if factual else None,
                "unsupported_ratio": unsupported / factual if factual else None,
                "review_method": "automatic_lexical_preliminary",
                "human_review_required": True,
            }
        )
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_manual_review(
    answer_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
    faithfulness_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
    no_answer_cases: list[dict[str, Any]],
    path: Path,
) -> None:
    quality = {row["question_id"]: row for row in quality_rows}
    citations = {row["question_id"]: row for row in citation_rows}
    faithfulness = {row["question_id"]: row for row in faithfulness_rows}
    fields = [
        "question_id",
        "kind",
        "category",
        "question",
        "ground_truth",
        "answer",
        "retrieved_node_ids",
        "auto_answer_status",
        "auto_point_coverage",
        "auto_citation_accuracy",
        "auto_citation_completeness",
        "auto_faithfulness_status",
        "human_answer_status",
        "human_covered_points",
        "human_citation_notes",
        "human_faithfulness_notes",
        "human_reviewer",
        "human_review_notes",
    ]
    rows = []
    for answer in answer_rows:
        qid = answer["question_id"]
        rows.append(
            {
                "question_id": qid,
                "kind": "answerable",
                "category": answer["category"],
                "question": answer["question"],
                "ground_truth": answer.get("expected_answer"),
                "answer": answer["answer"],
                "retrieved_node_ids": " | ".join(answer["retrieved_node_ids"]),
                "auto_answer_status": quality[qid]["status"],
                "auto_point_coverage": quality[qid].get("coverage"),
                "auto_citation_accuracy": citations[qid].get("citation_accuracy"),
                "auto_citation_completeness": citations[qid].get(
                    "citation_completeness"
                ),
                "auto_faithfulness_status": (
                    "grounded" if faithfulness[qid]["grounded_answer"] else "unresolved"
                ),
                "human_answer_status": "",
                "human_covered_points": "",
                "human_citation_notes": "",
                "human_faithfulness_notes": "",
                "human_reviewer": "",
                "human_review_notes": "",
            }
        )
    abstention_by_id = {row["question_id"]: row for row in abstention_rows}
    for case in no_answer_cases:
        row = abstention_by_id[case["question_id"]]
        rows.append(
            {
                "question_id": case["question_id"],
                "kind": "no_answer",
                "category": case["category"],
                "question": case["question"],
                "ground_truth": case["expected_behavior"],
                "answer": row["answer"],
                "retrieved_node_ids": " | ".join(row["retrieved_node_ids"]),
                "auto_answer_status": row["classification"],
                "auto_point_coverage": "",
                "auto_citation_accuracy": "",
                "auto_citation_completeness": "",
                "auto_faithfulness_status": "",
                "human_answer_status": "",
                "human_covered_points": "",
                "human_citation_notes": "",
                "human_faithfulness_notes": "",
                "human_reviewer": "",
                "human_review_notes": "",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cost_summary(
    path: Path,
    runtime: Runtime,
    answer_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
    warmup_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    all_rows = answer_rows + abstention_rows
    embedding_chars = sum(len(row["question"]) for row in all_rows + warmup_rows)
    rerank_documents = sum(
        row.get("rerank_input_document_count", 0)
        for row in all_rows + warmup_rows
    )
    prompt_tokens = sum(
        row["prompt_tokens"] for row in all_rows if row["prompt_tokens"] is not None
    )
    completion_tokens = sum(
        row["completion_tokens"]
        for row in all_rows
        if row["completion_tokens"] is not None
    )
    usage_available = all(
        row["total_tokens"] is not None for row in all_rows
    )
    payload = {
        "embedding": {
            "calls": runtime.call_counter.embedding_calls,
            "input_characters": embedding_chars,
            "input_tokens": None,
            "unit_price": None,
            "actual_cost": None,
            "status": "unknown; provider usage/pricing not exposed by project",
        },
        "reranker": {
            "calls": runtime.call_counter.rerank_calls,
            "input_documents": rerank_documents,
            "billing_unit": "unknown",
            "unit_price": None,
            "actual_cost": None,
            "status": "unknown; pricing not provided",
        },
        "llm": {
        "calls": sum(
            row.get("llm_calls", 0) for row in all_rows + warmup_rows
        ),
            "input_tokens": prompt_tokens if usage_available else None,
            "output_tokens": completion_tokens if usage_available else None,
            "usage_available": usage_available,
            "unit_price": 0,
            "unit_price_note": "OUR FREE MODEL currently shown as free; no long-term guarantee",
            "actual_cost": 0 if usage_available else None,
            "status": "free-tier; monetary cost observed as zero",
        },
        "per_question": {
            "embedding_cost": None,
            "reranker_cost": None,
            "llm_cost": 0 if usage_available else None,
            "total_cost": None,
        },
        "total_50_questions": {
            "embedding_cost": None,
            "reranker_cost": None,
            "llm_cost": 0 if usage_available else None,
            "total_cost": None,
        },
        "disclaimer": (
            "Character counts are not token counts. Unknown prices are not "
            "estimated and monetary totals are not fabricated."
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_report(
    run_dir: Path,
    preflight_data: dict[str, Any],
    retrieval_metrics: dict[str, Any],
    quality_summary: list[dict[str, Any]],
    citation_summary: list[dict[str, Any]],
    faithfulness_summary: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
    performance: dict[str, Any],
    cost: dict[str, Any],
) -> Path:
    answer_all = next(row for row in quality_summary if row["category"] == "ALL")
    citation_all = next(row for row in citation_summary if row["category"] == "ALL")
    faith_all = next(row for row in faithfulness_summary if row["category"] == "ALL")
    abstained = sum(row["correct_refusal"] for row in abstention_rows)
    abstention_attempted = sum(
        row["classification"] == "incorrect_answer" for row in abstention_rows
    )
    abstention_api_errors = sum(
        row["classification"] == "api_error" for row in abstention_rows
    )
    false_rejections = sum(contains_refusal(row["answer"]) for row in answer_rows)
    lines = [
        "# RAG 完整评估报告",
        "",
        "## 执行范围与状态",
        "",
        f"- 时间目录：`{run_dir.name}`",
        f"- 正式有答案问题：{len(answer_rows)}",
        f"- 无答案问题：{len(abstention_rows)}",
        "- 本报告中的答案质量、引用支持和忠实度使用可复现的自动文本覆盖检查，"
        "标记为 preliminary；`manual_review.csv` 保留人工最终判定列。",
        "- 未使用待测 LLM 自动判断标准答案，也未将自动结果冒充人工审核。",
        "",
        "## 预检与模型",
        "",
    ]
    for key, value in preflight_data.items():
        lines.append(f"- `{key}`：`{value}`")
    lines += [
        "",
        "## 检索评估",
        "",
        "旧实验的 A 保存 Vector 15 个候选，B 保存 RRF 前 10 个候选，"
        "C 仅保存 Rerank 后 Top 3。因此 C 的 Hit/Recall@5、@10 和 MRR@10 "
        "不可由历史数据恢复，未用 B 候选冒充。",
        "",
        "| 策略 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy in ("A", "B", "C"):
        row = retrieval_metrics["overall"][strategy]
        def fmt(key):
            value = row.get(key)
            return "N/A" if value is None else f"{value:.4f}"
        lines.append(
            f"| {strategy} | {fmt('hit_rate_at_1')} | {fmt('hit_rate_at_3')} | "
            f"{fmt('hit_rate_at_5')} | {fmt('hit_rate_at_10')} | {fmt('mrr')} | "
            f"{fmt('recall_at_1')} | {fmt('recall_at_3')} | "
            f"{fmt('recall_at_5')} | {fmt('recall_at_10')} |"
        )
    lines += [
        "",
        "## 答案质量（自动初审）",
        "",
        f"- 完全正确：{answer_all['completely_correct']}",
        f"- 部分正确：{answer_all['partially_correct']}",
        f"- 错误：{answer_all['wrong']}",
        f"- 无法判断：{answer_all['indeterminate']}",
        f"- 自动初审答案正确率：{answer_all['answer_accuracy_auto_preliminary']}",
        f"- 自动评分点覆盖率：{answer_all['point_coverage_auto_preliminary']}",
        f"- 有答案题中的拒答提示数：{false_rejections}",
        "- 最终答案正确率必须由 `manual_review.csv` 的人工结果确认。",
        "",
        "## 引用质量（自动初审）",
        "",
        f"- 总引用标记数：{citation_all['citation_count']}",
        f"- 通过内容覆盖检查的引用：{citation_all['supported_citation_count']}",
        f"- 引用准确率：{citation_all['citation_accuracy']}",
        f"- 引用完整率：{citation_all['citation_completeness']}",
        "- 检查基于被引用位置的 node 原文与对应陈述的实际词项/数字覆盖，"
        "不是只看文档名。",
        "",
        "## 忠实度（自动初审）",
        "",
        f"- 有依据回答比例：{faith_all['grounded_answer_ratio']}",
        f"- 有依据事实陈述比例：{faith_all['supported_ratio']}",
        f"- 无依据事实陈述比例：{faith_all['unsupported_ratio']}",
        f"- 部分支持陈述：{faith_all['partially_supported_statements']}",
        f"- 冲突陈述：{faith_all['conflicting_statements']}",
        "- 逐句证据映射见 `faithfulness_evaluation.jsonl`。",
        "",
        "## 拒答能力",
        "",
        f"- 正确拒答：{abstained}/{len(abstention_rows)}",
        f"- 无依据作答/编造：{abstention_attempted}",
        f"- API 错误：{abstention_api_errors}",
        f"- 正确拒答率（不含 API 错误）："
        f"{abstained / (len(abstention_rows) - abstention_api_errors) if len(abstention_rows) > abstention_api_errors else None}",
        "",
        "## 性能（完整问答）",
        "",
        "| 范围 | 指标 | Mean | P50 | P95 | Max |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope, metrics in performance.items():
        for metric in ("retrieval_ms", "llm_first_token_ms", "llm_generation_ms", "total_ms"):
            value = metrics[metric]
            lines.append(
                f"| {scope} | {metric} | {value['mean']} | {value['p50']} | "
                f"{value['p95']} | {value['max']} |"
            )
    lines += [
        "",
        "## 成本",
        "",
        f"- Embedding 调用：{cost['embedding']['calls']}，输入字符数 "
        f"{cost['embedding']['input_characters']}；实际 Token/金额：未知。",
        f"- Reranker 调用：{cost['reranker']['calls']}，输入文档数 "
        f"{cost['reranker']['input_documents']}；计费单位/金额：未知。",
        f"- LLM 调用：{cost['llm']['calls']}；输入 Token "
        f"{cost['llm']['input_tokens']}，输出 Token {cost['llm']['output_tokens']}；"
        f"当前标记免费，实际费用为 {cost['llm']['actual_cost']}。",
        "- 字符数不是 Token；未知单价和未知计费金额没有被估算。",
        "",
        "## 限制",
        "",
        "- 自动词项/数字覆盖检查可能低估同义改写，也可能高估部分措辞相似但语义错误的回答。",
        "- 最终答案正确率、引用准确率、引用完整率和忠实度需人工复核。",
        "- C 组历史结果只有 Rerank 后 Top 3，无法计算 C 的 @5/@10。",
        "- 模型配置为 OUR FREE MODEL，未来收费状况不确定。",
        "- 本评估不调参、不改 Prompt、不改检索参数、不重建索引。",
    ]
    path = run_dir / "full_evaluation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-answer-limit", type=int, default=None)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    records = load_jsonl(DATASET_PATH)
    stage3_rows = load_jsonl(STAGE3_RESULTS)
    runtime = build_runtime(records)
    pdf_texts = load_pdf_texts()
    node_texts = load_document_node_texts()
    no_answer_cases = verify_no_answer_cases(pdf_texts, node_texts)
    preflight_data = preflight(records, stage3_rows, runtime, no_answer_cases)
    print("PREFLIGHT", json.dumps(preflight_data, ensure_ascii=False))
    if args.preflight_only:
        retrieval, by_cat, per_q = build_retrieval_metrics(stage3_rows)
        print("RETRIEVAL_OVERALL", json.dumps(retrieval["overall"], ensure_ascii=False))
        print("preflight_only: ok")
        return 0

    if args.output_dir:
        run_dir = args.output_dir
        if run_dir.exists():
            raise FileExistsError(f"output directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    else:
        run_dir = make_run_dir()

    retrieval_metrics, by_category, per_question = build_retrieval_metrics(stage3_rows)
    (run_dir / "retrieval_metrics.json").write_text(
        json.dumps(retrieval_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_retrieval_category_csv(by_category, run_dir / "retrieval_metrics_by_category.csv")
    write_retrieval_question_csv(per_question, run_dir / "retrieval_metrics_per_question.csv")
    with (run_dir / "no_answer_eval_10.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for case in no_answer_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    selected_answers = records[: args.limit] if args.limit else records
    selected_no_answers = (
        no_answer_cases[: args.no_answer_limit]
        if args.no_answer_limit
        else no_answer_cases
    )
    cfg = get_settings()
    llm = create_llm(cfg)
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(cfg.docstore_dir)
    ).docs
    answer_path = run_dir / "answer_results.jsonl"
    answer_path.touch()

    if not args.skip_warmup and selected_answers:
        warmup_question = selected_answers[0]
        warmup_row = answer_question(
            runtime, warmup_question, llm, docstore, round_number=0, is_no_answer=False
        )
        append_jsonl(run_dir / "answer_warmup.jsonl", [warmup_row])
        print(
            "WARMUP",
            warmup_question["question_id"],
            "error=",
            warmup_row["api_error_message"],
        )

    answer_rows = []
    for index, question in enumerate(selected_answers, start=1):
        row = answer_question(
            runtime, question, llm, docstore, round_number=1, is_no_answer=False
        )
        row.update(
            {
                "expected_answer": question["expected_answer"],
                "evaluation_points": question["evaluation_points"],
                "source_document": question["source_document"],
                "source_page": question["source_page"],
            }
        )
        answer_rows.append(row)
        append_jsonl(answer_path, [row])
        print(
            f"ANSWER {index}/{len(selected_answers)} {question['question_id']} "
            f"error={row['api_error_message']} chars={len(row['answer'])}"
        )

    dataset_by_id = {record["question_id"]: record for record in records}
    quality_rows, citation_rows, faithfulness_rows = evaluate_answer_rows(
        answer_rows, dataset_by_id
    )
    quality_by_id = {row["question_id"]: row for row in quality_rows}
    citation_by_id = {row["question_id"]: row for row in citation_rows}
    faithfulness_by_id = {
        row["question_id"]: row for row in faithfulness_rows
    }
    for row in answer_rows:
        qid = row["question_id"]
        citation = citation_by_id[qid]
        quality = quality_by_id[qid]
        faithfulness = faithfulness_by_id[qid]
        row.update(
            {
                "citation_indices": [
                    item["citation_index"] for item in citation["citations"]
                ],
                "cited_node_ids": sorted(
                    {
                        item["node_id"]
                        for item in citation["citations"]
                        if item.get("node_id")
                    }
                ),
                "answer_quality_auto": quality["status"],
                "point_coverage_auto": quality["coverage"],
                "citation_accuracy_auto": citation["citation_accuracy"],
                "citation_completeness_auto": citation["citation_completeness"],
                "faithfulness_auto": {
                    "grounded_answer": faithfulness["grounded_answer"],
                    "supported_ratio": faithfulness["supported_ratio"],
                    "unsupported_ratio": faithfulness["unsupported_ratio"],
                },
            }
        )
    with answer_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in answer_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    append_jsonl(run_dir / "citation_evaluation.jsonl", citation_rows)
    append_jsonl(run_dir / "faithfulness_evaluation.jsonl", faithfulness_rows)

    abstention_rows = []
    abstention_path = run_dir / "abstention_results.jsonl"
    abstention_path.touch()
    for index, case in enumerate(selected_no_answers, start=1):
        row = answer_question(
            runtime, case, llm, docstore, round_number=1, is_no_answer=True
        )
        row["expected_behavior"] = case["expected_behavior"]
        row["why_unanswerable"] = case["why_unanswerable"]
        refusal = bool(row["answer"]) and contains_refusal(row["answer"])
        if row["api_error_message"]:
            classification = "api_error"
        elif refusal:
            classification = "correct_refusal"
        elif row["answer"]:
            classification = "incorrect_answer"
        else:
            classification = "empty_answer"
        row["refusal_detected"] = refusal
        row["classification"] = classification
        row["correct_refusal"] = classification == "correct_refusal"
        abstention_rows.append(row)
        append_jsonl(abstention_path, [row])
        print(
            f"NO_ANSWER {index}/{len(selected_no_answers)} {case['question_id']} "
            f"class={classification} error={row['api_error_message']}"
        )

    quality_summary = write_answer_quality_summary(
        quality_rows, run_dir / "answer_quality_summary.csv"
    )
    citation_summary = write_citation_summary(
        citation_rows, run_dir / "citation_summary.csv"
    )
    faithfulness_summary = write_faithfulness_summary(
        faithfulness_rows, run_dir / "faithfulness_summary.csv"
    )
    update_performance_scopes(
        answer_rows,
        abstention_rows,
        run_dir / "performance_summary.csv",
    )
    warmup_rows = (
        load_jsonl(run_dir / "answer_warmup.jsonl")
        if (run_dir / "answer_warmup.jsonl").exists()
        else []
    )
    cost = write_cost_summary(
        run_dir / "cost_summary.json",
        runtime,
        answer_rows,
        abstention_rows,
        warmup_rows,
    )
    write_manual_review(
        answer_rows,
        quality_rows,
        citation_rows,
        faithfulness_rows,
        abstention_rows,
        selected_no_answers,
        run_dir / "manual_review.csv",
    )
    performance = {}
    for scope, group in {
        "answer_50": answer_rows,
        "no_answer_10": abstention_rows,
        "all_60": answer_rows + abstention_rows,
    }.items():
        performance[scope] = timing_summary(group)
    report_path = write_report(
        run_dir,
        preflight_data,
        retrieval_metrics,
        quality_summary,
        citation_summary,
        faithfulness_summary,
        answer_rows,
        abstention_rows,
        performance,
        cost,
    )
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": str(DATASET_PATH),
        "historical_retrieval_results": str(STAGE3_RESULTS),
        "preflight": preflight_data,
        "parameters": runtime.parameters,
        "answer_count": len(answer_rows),
        "no_answer_count": len(abstention_rows),
        "call_counter": asdict(runtime.call_counter),
        "answer_api_errors": sum(bool(row["api_error_message"]) for row in answer_rows),
        "abstention_api_errors": sum(
            bool(row["api_error_message"]) for row in abstention_rows
        ),
        "review_status": "automatic preliminary metrics require manual confirmation",
        "report_path": str(report_path),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("RESULTS_DIR", run_dir)
    print("ANSWER_COUNT", len(answer_rows))
    print("NO_ANSWER_COUNT", len(abstention_rows))
    print("REPORT", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
