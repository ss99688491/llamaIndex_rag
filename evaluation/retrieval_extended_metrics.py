"""Extended retrieval metrics computed from the immutable stage-3 results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


CUTOFFS = (1, 3, 5, 10)
STRATEGY_NAMES = {
    "A": "vector",
    "B": "hybrid_rrf",
    "C": "hybrid_rrf_rerank",
}
EVIDENCE_MODES = {"MODEL-01": "required_all"}
DEFAULT_EVIDENCE_MODE = "alternative"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def candidate_ids(row: dict[str, Any], cutoff: int) -> list[str] | None:
    strategy = row["strategy"]
    if strategy == "A":
        return (row.get("vector_candidate_ids") or [])[:cutoff]
    if strategy == "B":
        return (row.get("rrf_candidate_ids") or [])[:cutoff]
    if strategy == "C":
        if cutoff > 3:
            return None
        return [hit["node_id"] for hit in (row.get("top3") or [])[:cutoff]]
    raise ValueError(f"unknown strategy: {strategy}")


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    expected = list(dict.fromkeys(row.get("expected_chunks") or []))
    if not expected:
        raise ValueError(f"{row['question_id']} has no expected chunks")
    mode = EVIDENCE_MODES.get(row["question_id"], DEFAULT_EVIDENCE_MODE)
    expected_set = set(expected)
    result = {
        "question_id": row["question_id"],
        "category": row["category"],
        "strategy": row["strategy"],
        "round": row["round"],
        "evidence_mode": mode,
        "expected_chunks": expected,
    }
    max_available = 3 if row["strategy"] == "C" else (10 if row["strategy"] == "B" else 15)
    first_hit_rank = None
    for rank, node_id in enumerate(candidate_ids(row, max_available) or [], start=1):
        if node_id in expected_set:
            first_hit_rank = rank
            break

    for cutoff in CUTOFFS:
        ranked = candidate_ids(row, cutoff)
        key = f"at_{cutoff}"
        if ranked is None:
            result[f"hit_rate_{key}"] = None
            result[f"recall_{key}"] = None
            continue
        hits = expected_set.intersection(ranked)
        result[f"hit_rate_{key}"] = 1.0 if hits else 0.0
        denominator = len(expected_set) if mode == "required_all" else 1
        result[f"recall_{key}"] = len(hits) / denominator

    if first_hit_rank is None:
        result["mrr"] = 0.0 if 1 in CUTOFFS else None
    else:
        result["mrr"] = 1.0 / first_hit_rank
    result["mrr_cutoff"] = max_available
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "record_count": len(rows),
        "evaluable_count_by_cutoff": {},
    }
    for cutoff in CUTOFFS:
        hit_values = [
            row[f"hit_rate_at_{cutoff}"]
            for row in rows
            if row[f"hit_rate_at_{cutoff}"] is not None
        ]
        recall_values = [
            row[f"recall_at_{cutoff}"]
            for row in rows
            if row[f"recall_at_{cutoff}"] is not None
        ]
        out[f"hit_rate_at_{cutoff}"] = mean(hit_values)
        out[f"recall_at_{cutoff}"] = mean(recall_values)
        out["evaluable_count_by_cutoff"][str(cutoff)] = len(hit_values)
    mrr_values = [row["mrr"] for row in rows if row["mrr"] is not None]
    out["mrr"] = mean(mrr_values)
    out["mrr_cutoff"] = rows[0]["mrr_cutoff"] if rows else None
    return out


def build_retrieval_metrics(
    raw_rows: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    evaluated = [evaluate_row(row) for row in raw_rows]
    evaluated.sort(
        key=lambda row: (row["question_id"], row["strategy"], row["round"])
    )

    by_category_rows = []
    for strategy in ("A", "B", "C"):
        strategy_rows = [row for row in evaluated if row["strategy"] == strategy]
        for category in ["ALL"] + sorted(
            {row["category"] for row in strategy_rows}
        ):
            group = (
                strategy_rows
                if category == "ALL"
                else [row for row in strategy_rows if row["category"] == category]
            )
            by_category_rows.append(
                {
                    "strategy": strategy,
                    "strategy_name": STRATEGY_NAMES[strategy],
                    "category": category,
                    **aggregate(group),
                }
            )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated:
        grouped[(row["question_id"], row["strategy"])].append(row)
    per_question = []
    for (question_id, strategy), rows in sorted(grouped.items()):
        aggregate_row = aggregate(rows)
        per_question.append(
            {
                "question_id": question_id,
                "category": rows[0]["category"],
                "strategy": strategy,
                "strategy_name": STRATEGY_NAMES[strategy],
                "evidence_mode": rows[0]["evidence_mode"],
                "expected_chunks": " | ".join(rows[0]["expected_chunks"]),
                **aggregate_row,
            }
        )

    overall = {
        strategy: aggregate(
            [row for row in evaluated if row["strategy"] == strategy]
        )
        for strategy in ("A", "B", "C")
    }
    semantics = {
        DEFAULT_EVIDENCE_MODE: (
            "expected_chunks are alternative valid evidence nodes. One hit gives "
            "Recall denominator 1."
        ),
        "required_all": (
            "All expected_chunks are jointly required for complete evidence recall. "
            "Hit Rate remains based on any relevant node."
        ),
        "required_all_questions": sorted(EVIDENCE_MODES),
        "alternative_multi_questions": sorted(
            {
                row["question_id"]
                for row in evaluated
                if len(row["expected_chunks"]) > 1
                and row["question_id"] not in EVIDENCE_MODES
            }
        ),
    }
    payload = {
        "source": "evaluation/results/20261112_231115/retrieval_results.jsonl",
        "cutoffs": list(CUTOFFS),
        "metrics": {
            "hit_rate_at_k": (
                "Questions with at least one expected node in the ranked top K / "
                "evaluable questions."
            ),
            "recall_at_k": (
                "For alternative evidence, any hit yields Recall 1. For required_all, "
                "the fraction of required expected nodes found in top K."
            ),
            "mrr": (
                "Mean reciprocal rank of the first relevant expected node. "
                "A uses top 15 candidates, B top 10, C top 3."
            ),
        },
        "evidence_semantics": semantics,
        "availability": {
            "A": "Hit/Recall@1,3,5,10; MRR cutoff 15",
            "B": "Hit/Recall@1,3,5,10; MRR cutoff 10",
            "C": "Hit/Recall@1,3 and MRR cutoff 3 only; stage-3 stored reranked top 3",
        },
        "overall": overall,
        "by_category": by_category_rows,
        "per_question": per_question,
        "per_round": evaluated,
    }
    return payload, by_category_rows, per_question


def write_category_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "strategy",
        "strategy_name",
        "category",
        "record_count",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "mrr",
        "mrr_cutoff",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_per_question_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "question_id",
        "category",
        "strategy",
        "strategy_name",
        "evidence_mode",
        "expected_chunks",
        "record_count",
        "hit_rate_at_1",
        "hit_rate_at_3",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "mrr",
        "mrr_cutoff",
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
