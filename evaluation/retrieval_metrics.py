"""Metrics and tabular outputs for the independent retrieval benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STRATEGIES = {
    "A": "vector",
    "B": "hybrid_rrf",
    "C": "hybrid_rrf_rerank",
}

TIMING_COLUMNS = [
    "embedding_ms",
    "vector_ms",
    "bm25_ms",
    "rrf_ms",
    "rerank_ms",
    "total_ms",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Iterable[float], probability: float) -> float | None:
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return sum(data) / len(data) if data else None


def _round(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _metrics_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        record for record in records if record.get("recall_at_3") is not None
    ]
    result: dict[str, Any] = {
        "records": len(records),
        "successful_records": len(successful),
        "failed_records": len(records) - len(successful),
        "recall_at_3": _mean(
            record["recall_at_3"] for record in successful
        ),
    }
    for column in TIMING_COLUMNS:
        values = [
            record.get(column)
            for record in successful
            if record.get(column) is not None
        ]
        result[f"avg_{column}"] = _mean(values)
        result[f"p50_{column}"] = percentile(values, 0.50)
        result[f"p95_{column}"] = percentile(values, 0.95)
        result[f"max_{column}"] = max(values) if values else None
    return result


def build_summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = sorted({record["category"] for record in records})
    rows: list[dict[str, Any]] = []
    for strategy, strategy_name in STRATEGIES.items():
        strategy_records = [
            record for record in records if record["strategy"] == strategy
        ]
        groups = [("ALL", strategy_records)] + [
            (
                category,
                [
                    record
                    for record in strategy_records
                    if record["category"] == category
                ],
            )
            for category in categories
        ]
        for category, group_records in groups:
            metrics = _metrics_for_records(group_records)
            row = {
                "strategy": strategy,
                "strategy_name": strategy_name,
                "category": category,
                **metrics,
            }
            for key, value in list(row.items()):
                if isinstance(value, float):
                    row[key] = _round(value)
            rows.append(row)
    return rows


def write_summary_csv(records: list[dict[str, Any]], path: Path) -> list[dict]:
    rows = build_summary_rows(records)
    fields = [
        "strategy",
        "strategy_name",
        "category",
        "records",
        "successful_records",
        "failed_records",
        "recall_at_3",
    ]
    for column in TIMING_COLUMNS:
        fields.extend(
            [
                f"avg_{column}",
                f"p50_{column}",
                f"p95_{column}",
                f"max_{column}",
            ]
        )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_per_question_csv(
    records: list[dict[str, Any]], path: Path
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (record["question_id"], record["strategy"])
        entry = grouped.setdefault(
            key,
            {
                "question_id": record["question_id"],
                "category": record["category"],
                "strategy": record["strategy"],
                "rounds": 0,
                "successful_rounds": 0,
                "recall_values": [],
            },
        )
        entry["rounds"] += 1
        if record.get("recall_at_3") is not None:
            entry["successful_rounds"] += 1
            entry["recall_values"].append(float(record["recall_at_3"]))

    by_question: dict[str, dict[str, float | None]] = defaultdict(dict)
    for (question_id, strategy), entry in grouped.items():
        recall = _mean(entry["recall_values"])
        by_question[question_id][strategy] = recall
        row = {
            "question_id": question_id,
            "category": entry["category"],
            "strategy": strategy,
            "rounds": entry["rounds"],
            "successful_rounds": entry["successful_rounds"],
            "failures": entry["rounds"] - entry["successful_rounds"],
            "mean_recall_at_3": _round(recall),
        }
        entry["output"] = row

    rows = [entry["output"] for entry in grouped.values()]
    rows.sort(key=lambda row: (row["question_id"], row["strategy"]))
    fields = [
        "question_id",
        "category",
        "strategy",
        "rounds",
        "successful_rounds",
        "failures",
        "mean_recall_at_3",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    differences = []
    for question_id in sorted(by_question):
        values = by_question[question_id]
        category = next(
            record["category"]
            for record in records
            if record["question_id"] == question_id
        )
        a = values.get("A")
        b = values.get("B")
        c = values.get("C")
        differences.append(
            {
                "question_id": question_id,
                "category": category,
                "A": _round(a),
                "B": _round(b),
                "C": _round(c),
                "B_minus_A": _round(None if a is None or b is None else b - a),
                "C_minus_B": _round(None if b is None or c is None else c - b),
                "C_minus_A": _round(None if a is None or c is None else c - a),
            }
        )
    diff_path = path.with_name("retrieval_per_question_comparison.csv")
    with diff_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "category",
                "A",
                "B",
                "C",
                "B_minus_A",
                "C_minus_B",
                "C_minus_A",
            ],
        )
        writer.writeheader()
        writer.writerows(differences)
    return rows


def question_level_differences(
    records: list[dict[str, Any]]
) -> dict[str, dict[str, int]]:
    by_question: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("recall_at_3") is None:
            continue
        by_question[record["question_id"]].setdefault(
            record["strategy"], []
        ).append(float(record["recall_at_3"]))
    comparisons = {
        "B_vs_A": {"win": 0, "loss": 0, "tie": 0, "incomplete": 0},
        "C_vs_B": {"win": 0, "loss": 0, "tie": 0, "incomplete": 0},
        "C_vs_A": {"win": 0, "loss": 0, "tie": 0, "incomplete": 0},
    }
    pairs = {
        "B_vs_A": ("B", "A"),
        "C_vs_B": ("C", "B"),
        "C_vs_A": ("C", "A"),
    }
    for values in by_question.values():
        for name, (left, right) in pairs.items():
            if left not in values or right not in values:
                comparisons[name]["incomplete"] += 1
                continue
            left_mean = sum(values[left]) / len(values[left])
            right_mean = sum(values[right]) / len(values[right])
            if left_mean > right_mean:
                comparisons[name]["win"] += 1
            elif left_mean < right_mean:
                comparisons[name]["loss"] += 1
            else:
                comparisons[name]["tie"] += 1
    return comparisons
