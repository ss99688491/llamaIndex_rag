"""Transparent lexical assistants for citation, faithfulness, and point coverage.

These are deterministic preliminary checks, not a substitute for independent
human review.
"""

from __future__ import annotations

import re
from typing import Any

import jieba


REFUSAL_PATTERNS = (
    "未提供",
    "未提到",
    "未提及",
    "没有提供",
    "无法确定",
    "无法回答",
    "无法判断",
    "资料中没有",
    "手册中没有",
    "not provided",
    "not mentioned",
    "does not specify",
    "doesn't specify",
    "cannot determine",
    "can't determine",
    "insufficient information",
)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "with", "as", "by", "this", "that", "it", "its",
    "can", "could", "should", "will", "would", "be", "from", "at", "if",
    "说明", "回答", "产品", "支持", "可以", "需要", "包括", "以及", "使用",
    "进行", "一个", "这个", "其中", "提供", "根据", "内容", "信息",
}

NUMBER_RE = re.compile(
    r"\d+(?:\.\d+)?(?:\s*%|\s*°\s*[CF]|\s*mm|\s*cm|\s*kg|\s*g|"
    r"\s*gbps|\s*mbps|\s*v|\s*w|小时|分钟|秒|米)?",
    re.IGNORECASE,
)
ASCII_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:+/-]*")
CITATION_RE = re.compile(r"(?:\[|【)(\d+)(?:\]|】)")


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def contains_refusal(answer: str) -> bool:
    lowered = normalize(answer).lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def key_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in ASCII_TERM_RE.findall(text):
        token = match.lower().strip(".,")
        if len(token) >= 2 and token not in STOPWORDS:
            tokens.add(token)
    for match in NUMBER_RE.findall(text):
        tokens.add(match.replace(" ", "").lower())
    for token in jieba.cut(text, cut_all=False):
        token = token.strip().lower()
        if (
            any("\u4e00" <= char <= "\u9fff" for char in token)
            and len(token) >= 2
            and token not in STOPWORDS
        ):
            tokens.add(token)
    return tokens


def number_tokens(text: str) -> set[str]:
    return {match.replace(" ", "").lower() for match in NUMBER_RE.findall(text)}


def split_sentences(answer: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", answer)
    sentences = []
    for piece in pieces:
        piece = normalize(piece)
        if not piece or len(key_tokens(piece)) < 2:
            continue
        sentences.append(piece)
    return sentences


def support_for_text(text: str, context_items: list[dict[str, Any]]) -> dict[str, Any]:
    answer_terms = key_tokens(text)
    answer_numbers = number_tokens(text)
    best = None
    for context in context_items:
        context_terms = key_tokens(context["text"])
        context_numbers = number_tokens(context["text"])
        overlap = answer_terms.intersection(context_terms)
        ratio = len(overlap) / max(1, len(answer_terms))
        number_match = answer_numbers.issubset(context_numbers)
        score = ratio + (0.15 if answer_numbers and number_match else 0.0)
        if best is None or score > best["score"]:
            best = {
                "node_id": context["node_id"],
                "document": context.get("document"),
                "page_label": context.get("page_label"),
                "score": round(score, 4),
                "overlap_terms": sorted(overlap),
                "number_match": number_match,
                "evidence_text": context["text"],
            }
    if best is None:
        return {"status": "no_context", "best_match": None}
    if answer_numbers and not best["number_match"]:
        status = "unsupported"
    elif best["score"] >= 0.55:
        status = "supported"
    elif best["score"] >= 0.20:
        status = "partially_supported"
    else:
        status = "unsupported"
    return {"status": status, "best_match": best}


def evaluate_point(point: str, answer: str) -> dict[str, Any]:
    answer_normalized = normalize(answer).lower()
    numbers = number_tokens(point)
    ascii_terms = {
        term.lower()
        for term in ASCII_TERM_RE.findall(point)
        if len(term) >= 2 and term.lower() not in STOPWORDS
    }
    point_terms = {
        token
        for token in key_tokens(point)
        if token not in numbers and token not in ascii_terms
    }
    answer_terms = key_tokens(answer_normalized)
    number_ok = all(number in answer_normalized.replace(" ", "") for number in numbers)
    ascii_ok = all(term in answer_normalized for term in ascii_terms)
    matched_terms = point_terms.intersection(answer_terms)
    term_ratio = len(matched_terms) / max(1, len(point_terms))
    if numbers or ascii_terms:
        covered = number_ok and ascii_ok and (not point_terms or term_ratio >= 0.4)
    else:
        covered = term_ratio >= 0.6
    return {
        "point": point,
        "auto_covered": covered,
        "auto_status": "covered" if covered else "pending_human",
        "matched_terms": sorted(matched_terms),
        "required_numbers": sorted(numbers),
        "number_match": number_ok,
        "ascii_terms": sorted(ascii_terms),
        "ascii_match": ascii_ok,
    }


def evaluate_answer_quality(
    question: dict[str, Any], answer: str, generation_error: str | None
) -> dict[str, Any]:
    if generation_error or not normalize(answer):
        status = "indeterminate"
    else:
        points = [
            evaluate_point(point, answer)
            for point in question["evaluation_points"]
        ]
        covered = sum(point["auto_covered"] for point in points)
        coverage = covered / len(points) if points else 0.0
        if contains_refusal(answer):
            status = "wrong"
        elif coverage >= 0.8:
            status = "completely_correct"
        elif coverage > 0:
            status = "partially_correct"
        else:
            status = "wrong"
        return {
            "status": status,
            "coverage": coverage,
            "covered_points": covered,
            "total_points": len(points),
            "point_results": points,
            "review_method": "automatic_lexical_preliminary",
            "human_review_required": True,
        }
    return {
        "status": status,
        "coverage": None,
        "covered_points": 0,
        "total_points": len(question["evaluation_points"]),
        "point_results": [],
        "review_method": "automatic_lexical_preliminary",
        "human_review_required": True,
    }


def evaluate_citations(
    question: dict[str, Any], answer: str, retrieved: list[dict[str, Any]]
) -> dict[str, Any]:
    sentences = split_sentences(answer)
    citation_items = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        containing = [
            sentence for sentence in sentences if match.group(0) in sentence
        ]
        statement = containing[0] if containing else answer[max(0, match.start() - 120):match.end() + 120]
        if not 1 <= index <= len(retrieved):
            citation_items.append(
                {
                    "citation_index": index,
                    "valid_index": False,
                    "node_id": None,
                    "statement": statement,
                    "support_status": "unsupported",
                    "support": None,
                }
            )
            continue
        context = retrieved[index - 1]
        support = support_for_text(statement, [context])
        citation_items.append(
            {
                "citation_index": index,
                "valid_index": True,
                "node_id": context["node_id"],
                "document": context.get("document"),
                "page_label": context.get("page_label"),
                "statement": statement,
                "support_status": support["status"],
                "support": support,
            }
        )

    total = len(citation_items)
    supported = sum(
        item["support_status"] == "supported" for item in citation_items
    )
    expected = list(dict.fromkeys(
        question.get("expected_chunks") or [question.get("expected_chunk")]
    ))
    cited_nodes = {
        item["node_id"]
        for item in citation_items
        if item["valid_index"] and item["support_status"] == "supported"
    }
    required = (
        len(set(expected)) if question.get("question_id") == "MODEL-01" else 1
    )
    covered_expected = len(set(expected).intersection(cited_nodes))
    completeness = min(1.0, covered_expected / required) if required else None
    return {
        "citations": citation_items,
        "citation_count": total,
        "supported_citation_count": supported,
        "citation_accuracy": supported / total if total else None,
        "expected_support_nodes": expected,
        "covered_expected_nodes": sorted(cited_nodes.intersection(expected)),
        "citation_completeness": completeness,
        "review_method": "automatic_lexical_preliminary",
        "human_review_required": True,
    }


def evaluate_faithfulness(
    answer: str, retrieved: list[dict[str, Any]]
) -> dict[str, Any]:
    statements = []
    for sentence in split_sentences(answer):
        if contains_refusal(sentence):
            statements.append(
                {
                    "statement": sentence,
                    "status": "abstention",
                    "support": None,
                }
            )
            continue
        support = support_for_text(sentence, retrieved)
        statements.append(
            {
                "statement": sentence,
                "status": support["status"],
                "support": support["best_match"],
            }
        )
    factual = [item for item in statements if item["status"] != "abstention"]
    supported = sum(item["status"] == "supported" for item in factual)
    partial = sum(item["status"] == "partially_supported" for item in factual)
    unsupported = sum(
        item["status"] in {"unsupported", "no_context"} for item in factual
    )
    conflict = sum(item["status"] == "conflict" for item in factual)
    total = len(factual)
    return {
        "statements": statements,
        "factual_statement_count": total,
        "supported_statement_count": supported,
        "partially_supported_statement_count": partial,
        "unsupported_statement_count": unsupported,
        "conflicting_statement_count": conflict,
        "grounded_answer": supported == total and total > 0,
        "supported_ratio": supported / total if total else None,
        "unsupported_ratio": unsupported / total if total else None,
        "review_method": "automatic_lexical_preliminary",
        "human_review_required": True,
    }
