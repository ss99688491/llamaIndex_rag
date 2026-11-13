"""Independent full RAG evaluation runner for the frozen clean index.

The production RAG implementation is imported for prompt/model compatibility,
but all index paths, retrieval observability, persistence and recovery are
owned by this evaluation module. The runner never reads the legacy index unless
the caller explicitly points it there, and the expected clean-index fingerprint
is checked before every question.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.llms import ChatResponse
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from llama_index.retrievers.bm25 import BM25Retriever
from pypdf import PdfReader

from evaluation.no_answer_cases import NO_ANSWER_CASES
from evaluation.response_quality import (
    contains_refusal,
    evaluate_answer_quality,
    evaluate_citations,
    evaluate_faithfulness,
)
from evaluation.retrieval_benchmark import sanitize_error
from src.chat import DEFAULT_STYLE, _build_system_prompt
from src.config import Settings as AppSettings
from src.settings import OpenAICompatible, _build_embedding


DATASET_PATH = ROOT / "evaluation" / "datasets" / "rag_eval_50.jsonl"
ARCHIVED_NO_ANSWER_PATH = (
    ROOT
    / "evaluation"
    / "results"
    / "full_evaluation_20261112_234856"
    / "no_answer_eval_10.jsonl"
)
NODE_DIFF_PATH = (
    ROOT
    / "evaluation"
    / "index_audit"
    / "clean_index"
    / "comparison"
    / "node_diff.csv"
)
OLD_NODES_CSV = (
    ROOT / "evaluation" / "index_audit" / "old_index_nodes.csv"
)
DEFAULT_MANIFEST_PATH = (
    ROOT / "evaluation" / "index_audit" / "clean_index" / "index_manifest.json"
)
RESULTS_ROOT = ROOT / "evaluation" / "results"

DEFAULT_EXPECTED_FINGERPRINT = (
    "4595dfd3be078f5a05749869fa2fae3e31c2a5ad6227807382f2b8e92d63221c"
)
EXPECTED_CATEGORIES = {
    "参数查询",
    "安装操作",
    "故障排查",
    "跨产品型号归属辨析",
    "跨语言检索",
}
REQUIRED_ALL_QUESTIONS = {"MODEL-01"}
RETRIEVAL_CUTOFFS = (1, 3, 4)
REQUIRED_METADATA = {
    "page_label",
    "file_name",
    "file_path",
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
    "file_hash",
    "file_sha256",
    "original_name",
    "uploaded_at",
    "source_page",
    "page_index",
}
NO_ANSWER_FIELDS = (
    "question_id",
    "category",
    "question",
    "question_language",
    "target_document",
    "absence_terms",
    "why_unanswerable",
    "expected_behavior",
)


class EvaluationBlocked(RuntimeError):
    """Raised when a preflight condition prevents a trustworthy run."""


class IndexFingerprintChanged(EvaluationBlocked):
    """Raised when the live index no longer matches the frozen fingerprint."""


class ConsecutiveFailureLimit(EvaluationBlocked):
    """Raised when the configured consecutive API-failure threshold is reached."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\u00a0", " ").split())


def compact_text(value: str | None) -> str:
    return re.sub(r"\s+", "", value or "")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(encoded)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evaluation_dataset(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = [record.get("question_id") for record in records]
    if len(records) != 50 or len(set(ids)) != 50:
        raise EvaluationBlocked(
            f"Expected 50 unique answerable questions, got "
            f"{len(records)} rows and {len(set(ids))} IDs."
        )
    categories = Counter(record.get("category") for record in records)
    if set(categories) != EXPECTED_CATEGORIES or any(
        count != 10 for count in categories.values()
    ):
        raise EvaluationBlocked(f"Unexpected category distribution: {categories}")
    required = {
        "question_id",
        "category",
        "question",
        "expected_answer",
        "evaluation_points",
        "expected_chunk",
        "expected_chunks",
        "source_document",
        "evidence",
    }
    missing = {
        record.get("question_id", "<unknown>"): sorted(
            required - set(record)
        )
        for record in records
        if required - set(record)
    }
    if missing:
        raise EvaluationBlocked(f"Dataset fields missing: {missing}")
    return {
        "record_count": len(records),
        "unique_ids": len(set(ids)),
        "categories": dict(sorted(categories.items())),
        "sha256": sha256_file(DATASET_PATH),
    }


def validate_no_answer_cases(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(cases) != 10:
        raise EvaluationBlocked(
            f"Expected exactly 10 no-answer cases, got {len(cases)}."
        )
    ids = [case.get("question_id") for case in cases]
    if len(set(ids)) != 10:
        raise EvaluationBlocked(f"Duplicate no-answer IDs: {ids}")
    missing = {
        case.get("question_id", "<unknown>"): [
            key for key in NO_ANSWER_FIELDS if key not in case
        ]
        for case in cases
        if any(key not in case for key in NO_ANSWER_FIELDS)
    }
    if missing:
        raise EvaluationBlocked(f"No-answer fields missing: {missing}")
    canonical = {case["question_id"]: case for case in NO_ANSWER_CASES}
    disk = {case["question_id"]: case for case in cases}
    if set(canonical) != set(disk):
        raise EvaluationBlocked("No-answer question IDs differ from canonical set.")
    differences = {
        question_id: {
            field: {
                "canonical": canonical[question_id].get(field),
                "selected": disk[question_id].get(field),
            }
            for field in NO_ANSWER_FIELDS
            if canonical[question_id].get(field) != disk[question_id].get(field)
        }
        for question_id in canonical
    }
    differences = {
        question_id: fields
        for question_id, fields in differences.items()
        if fields
    }
    if differences:
        raise EvaluationBlocked(
            f"No-answer definitions differ from canonical source: {differences}"
        )
    return {
        "record_count": len(cases),
        "unique_ids": len(set(ids)),
        "sha256": stable_hash(
            [
                {field: case[field] for field in NO_ANSWER_FIELDS}
                for case in cases
            ]
        ),
        "matches_canonical_source": True,
    }


def resolve_no_answer_cases(
    explicit_path: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if explicit_path is not None:
        path = explicit_path.resolve()
        if not path.is_file():
            raise EvaluationBlocked(f"No-answer dataset not found: {path}")
        return read_jsonl(path), str(path)
    if ARCHIVED_NO_ANSWER_PATH.is_file():
        return read_jsonl(ARCHIVED_NO_ANSWER_PATH), str(
            ARCHIVED_NO_ANSWER_PATH.resolve()
        )
    return [dict(case) for case in NO_ANSWER_CASES], "evaluation/no_answer_cases.py"


def verify_no_answer_absence_in_clean_index(
    cases: list[dict[str, Any]],
    snapshot: CleanIndexSnapshot,
) -> dict[str, Any]:
    node_text_by_document: dict[str, list[str]] = defaultdict(list)
    for record in snapshot.records.values():
        node_text_by_document[record["document"]].append(record["text"])
    pdf_text_by_document: dict[str, str] = {}
    upload_dir = ROOT / "data" / "uploads"
    for path in sorted(upload_dir.glob("*.pdf")):
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")
        document = path.name.split("_", 2)[-1]
        pdf_text_by_document[document] = " ".join(
            page.extract_text() or "" for page in reader.pages
        )

    results = []
    for case in cases:
        document = case["target_document"]
        pdf_text = pdf_text_by_document.get(document, "").lower()
        node_text = " ".join(node_text_by_document.get(document, [])).lower()
        compact_pdf = compact_text(pdf_text)
        compact_nodes = compact_text(node_text)
        checks = []
        for term in case["absence_terms"]:
            lowered = term.lower()
            compact_term = compact_text(lowered)
            cjk = any("\u4e00" <= char <= "\u9fff" for char in term)
            pdf_present = (
                compact_term in compact_pdf if cjk else lowered in pdf_text
            )
            nodes_present = (
                compact_term in compact_nodes if cjk else lowered in node_text
            )
            checks.append(
                {
                    "term": term,
                    "present_in_pdf": pdf_present,
                    "present_in_clean_nodes": nodes_present,
                }
            )
        if any(
            check["present_in_pdf"] or check["present_in_clean_nodes"]
            for check in checks
        ):
            raise EvaluationBlocked(
                f"{case['question_id']} has a potentially answerable absence term."
            )
        results.append(
            {
                "question_id": case["question_id"],
                "target_document": document,
                "absence_checks": checks,
                "verified_unanswerable": True,
            }
        )
    return {
        "case_count": len(results),
        "all_verified_unanswerable": True,
        "cases": results,
    }


def validate_explicit_index_paths(
    chroma_dir: Path,
    docstore_dir: Path,
) -> dict[str, str]:
    if chroma_dir is None or docstore_dir is None:
        raise EvaluationBlocked(
            "Explicit --chroma-dir and --docstore-dir are required."
        )
    chroma = chroma_dir.resolve()
    docstore = docstore_dir.resolve()
    if not chroma.is_dir() or not docstore.is_dir():
        raise EvaluationBlocked(
            f"Index directories do not exist: chroma={chroma}, docstore={docstore}"
        )
    legacy = (ROOT / "storage").resolve()
    if chroma == (legacy / "chroma").resolve():
        raise EvaluationBlocked("Clean runner refuses the legacy Chroma path.")
    if docstore == (legacy / "docstore").resolve():
        raise EvaluationBlocked("Clean runner refuses the legacy DocStore path.")
    clean_root = (
        ROOT / "evaluation" / "index_audit" / "clean_index" / "storage"
    ).resolve()
    if clean_root not in chroma.parents or clean_root not in docstore.parents:
        raise EvaluationBlocked(
            "Index paths must be under the frozen clean-index storage root."
        )
    return {"chroma_dir": str(chroma), "docstore_dir": str(docstore)}


@dataclass
class CleanIndexSnapshot:
    chroma_dir: Path
    docstore_dir: Path
    collection: Any
    docstore: SimpleDocumentStore
    records: dict[str, dict[str, Any]]
    fingerprint: str
    counts: dict[str, Any]


def _load_index_contents(
    chroma_dir: Path,
    docstore_dir: Path,
) -> tuple[Any, SimpleDocumentStore, dict[str, dict[str, Any]]]:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection(name="knowledge_base")
    raw = collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(docstore_dir)
    )
    records: dict[str, dict[str, Any]] = {}
    for node_id, metadata, text, embedding in zip(
        raw["ids"],
        raw["metadatas"],
        raw["documents"],
        raw["embeddings"],
    ):
        metadata = metadata or {}
        text = text or ""
        embedding_list = (
            [float(value) for value in embedding]
            if embedding is not None
            else []
        )
        records[node_id] = {
            "node_id": node_id,
            "metadata": metadata,
            "text": text,
            "embedding": embedding_list,
            "document_id": str(metadata.get("document_id") or ""),
            "document": str(metadata.get("original_name") or ""),
            "physical_page": (
                int(metadata["source_page"])
                if metadata.get("source_page") is not None
                else None
            ),
            "page_label": str(metadata.get("page_label") or ""),
        }
    return collection, docstore, records


def compute_index_fingerprint(
    records: dict[str, dict[str, Any]],
    docstore: SimpleDocumentStore,
) -> str:
    payload = []
    for node_id in sorted(records):
        item = records[node_id]
        node = docstore.docs.get(node_id)
        if node is None:
            raise EvaluationBlocked(
                f"Node {node_id} exists in Chroma but not DocStore."
            )
        payload.append(
            {
                "node_id": node_id,
                "document_id": str(
                    node.ref_doc_id
                    or item["metadata"].get("document_id")
                    or ""
                ),
                "text": normalize_text(item["text"]),
                "metadata": {
                    key: item["metadata"].get(key)
                    for key in sorted(REQUIRED_METADATA)
                },
                "embedding": item["embedding"],
            }
        )
    return stable_hash(payload)


def load_and_verify_index(
    chroma_dir: Path,
    docstore_dir: Path,
    manifest_path: Path,
    expected_fingerprint: str,
) -> CleanIndexSnapshot:
    path_check = validate_explicit_index_paths(chroma_dir, docstore_dir)
    manifest = load_manifest(manifest_path.resolve())
    if manifest.get("status") != "clean_index_verified":
        raise EvaluationBlocked(
            f"Index manifest is not clean_index_verified: "
            f"{manifest.get('status')!r}"
        )
    collection, docstore, records = _load_index_contents(
        Path(path_check["chroma_dir"]),
        Path(path_check["docstore_dir"]),
    )
    chroma_ids = set(records)
    docstore_ids = set(docstore.docs)
    if chroma_ids != docstore_ids:
        raise EvaluationBlocked(
            f"Chroma/DocStore ID mismatch: "
            f"{len(chroma_ids)} vs {len(docstore_ids)}"
        )
    if len(records) != 240:
        raise EvaluationBlocked(f"Expected 240 clean nodes, got {len(records)}")
    fingerprints = compute_index_fingerprint(records, docstore)
    if fingerprints != expected_fingerprint:
        raise IndexFingerprintChanged(
            f"Frozen fingerprint mismatch: {fingerprints} != "
            f"{expected_fingerprint}"
        )
    body_equal = 0
    metadata_equal = 0
    pages: set[tuple[str, int]] = set()
    empty_text = 0
    missing_embedding = 0
    dimensions: set[int] = set()
    for node_id, item in records.items():
        node = docstore.docs[node_id]
        if item["text"] == node.get_content():
            body_equal += 1
        node_metadata = dict(node.metadata or {})
        if all(
            key in item["metadata"]
            and key in node_metadata
            and item["metadata"][key] == node_metadata[key]
            for key in REQUIRED_METADATA
        ):
            metadata_equal += 1
        if not normalize_text(item["text"]):
            empty_text += 1
        if not item["embedding"]:
            missing_embedding += 1
        else:
            dimensions.add(len(item["embedding"]))
        if item["physical_page"] is not None:
            pages.add((item["document"], item["physical_page"]))
    counts = {
        "chroma_count": len(records),
        "docstore_count": len(docstore.docs),
        "body_equal": body_equal,
        "metadata_equal": metadata_equal,
        "physical_pages": len(pages),
        "documents": len({document for document, _ in pages}),
        "empty_text": empty_text,
        "missing_embedding": missing_embedding,
        "embedding_dimensions": sorted(dimensions),
    }
    if not (
        body_equal == 240
        and metadata_equal == 240
        and len(pages) == 181
        and counts["documents"] == 5
        and empty_text == 0
        and missing_embedding == 0
        and dimensions == {1024}
    ):
        raise EvaluationBlocked(f"Clean index integrity failed: {counts}")
    return CleanIndexSnapshot(
        chroma_dir=Path(path_check["chroma_dir"]),
        docstore_dir=Path(path_check["docstore_dir"]),
        collection=collection,
        docstore=docstore,
        records=records,
        fingerprint=fingerprints,
        counts=counts,
    )


def verify_index_still_frozen(
    chroma_dir: Path,
    docstore_dir: Path,
    expected_fingerprint: str,
) -> str:
    _, docstore, records = _load_index_contents(chroma_dir, docstore_dir)
    actual = compute_index_fingerprint(records, docstore)
    if actual != expected_fingerprint:
        raise IndexFingerprintChanged(
            f"Index fingerprint changed during evaluation: "
            f"{actual} != {expected_fingerprint}"
        )
    return actual


def evidence_fragments(evidence: str) -> list[str]:
    pieces = evidence.replace("...", "§").replace("…", "§").split("§")
    return [piece.strip() for piece in pieces if compact_text(piece)]


def expected_ids(question: dict[str, Any]) -> list[str]:
    values = question.get("expected_chunks") or [question["expected_chunk"]]
    return list(dict.fromkeys(value for value in values if value))


def evidence_mode(question_id: str) -> str:
    return "required_all" if question_id in REQUIRED_ALL_QUESTIONS else "alternative"


def build_expected_chunk_mapping(
    dataset: list[dict[str, Any]],
    node_diff_csv: Path,
    clean_snapshot: CleanIndexSnapshot,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with node_diff_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        diff_rows = list(csv.DictReader(handle))
    with OLD_NODES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        old_node_rows = {
            row["node_id"]: row for row in csv.DictReader(handle)
        }
    by_old: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in diff_rows:
        if row.get("old_node_id") and row.get("new_node_id"):
            by_old[row["old_node_id"]].append(row)

    question_checks: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    blocked: list[str] = []
    all_expected = {
        node_id
        for question in dataset
        for node_id in expected_ids(question)
    }
    for question in dataset:
        question_id = question["question_id"]
        direct_new_ids: list[str] = []
        direct_map: dict[str, list[dict[str, str]]] = {}
        for old_id in expected_ids(question):
            rows = by_old.get(old_id, [])
            valid_rows = [
                row
                for row in rows
                if row["status"]
                in {"id_changed_but_content_same", "content_changed"}
                and row["new_node_id"] in clean_snapshot.records
            ]
            if not valid_rows:
                blocked.append(
                    f"{question_id}:{old_id} has no verified clean-index mapping"
                )
                continue
            direct_map[old_id] = valid_rows
            direct_new_ids.extend(row["new_node_id"] for row in valid_rows)

        direct_new_ids = list(dict.fromkeys(direct_new_ids))
        union_text = " ".join(
            clean_snapshot.records[node_id]["text"]
            for node_id in direct_new_ids
        )
        fragments = evidence_fragments(question["evidence"])
        fragment_checks = [
            {
                "fragment": fragment,
                "present": compact_text(fragment) in compact_text(union_text),
            }
            for fragment in fragments
        ]

        for old_id, rows in direct_map.items():
            for row in rows:
                new_id = row["new_node_id"]
                new_record = clean_snapshot.records[new_id]
                if (
                    new_record["document"] != row["document_name"]
                    or int(new_record["physical_page"])
                    != int(row["physical_page_new"])
                ):
                    blocked.append(
                        f"{question_id}:{old_id}->{new_id} changes source "
                        "document or physical page"
                    )
                mapping_rows.append(
                    {
                        "question_id": question_id,
                        "old_node_id": old_id,
                        "new_node_id": new_id,
                        "mapping_role": "direct",
                        "mapping_basis": (
                            f"node_diff.csv:{row['status']}:"
                            f"{row['match_evidence']}:"
                            f"{row['old_chunk_start_char']}-"
                            f"{row['old_chunk_end_char']}->"
                            f"{row['new_chunk_start_char']}-"
                            f"{row['new_chunk_end_char']}"
                        ),
                        "old_document": row["document_name"],
                        "old_physical_page": int(row["physical_page_old"]),
                        "new_document": new_record["document"],
                        "new_physical_page": int(new_record["physical_page"]),
                        "body_equal": row["body_equal"],
                        "review_status": "passed",
                        "evidence_mode": evidence_mode(question_id),
                        "mapping_status": "passed",
                        "old_text_sha256_normalized": old_node_rows.get(
                            old_id, {}
                        ).get("text_sha256"),
                        "new_text_sha256_normalized": sha256_text(
                            normalize_text(new_record["text"])
                        ),
                    }
                )

        # Direct evidence is normally sufficient. If one old chunk was split,
        # supplementary same-document nodes are permitted only when they contain
        # a previously uncovered evidence fragment.
        for check in fragment_checks:
            if check["present"]:
                continue
            candidates = [
                record
                for record in clean_snapshot.records.values()
                if record["document"] in {
                    row["new_document"] for row in mapping_rows
                    if row["question_id"] == question_id
                }
                and compact_text(check["fragment"])
                in compact_text(record["text"])
            ]
            if not candidates:
                continue
            for record in sorted(candidates, key=lambda item: item["node_id"]):
                if record["node_id"] in direct_new_ids:
                    continue
                direct_new_ids.append(record["node_id"])
                mapping_rows.append(
                    {
                        "question_id": question_id,
                        "old_node_id": expected_ids(question)[0],
                        "new_node_id": record["node_id"],
                        "mapping_role": "supplementary_evidence_fragment",
                        "mapping_basis": "same_document_evidence_fragment_match",
                        "old_document": question["source_document"],
                        "old_physical_page": int(question["source_page"]),
                        "new_document": record["document"],
                        "new_physical_page": int(record["physical_page"]),
                        "body_equal": False,
                        "review_status": "manual_mapping_evidence_check",
                        "evidence_mode": evidence_mode(question_id),
                        "mapping_status": "passed",
                        "old_text_sha256_normalized": old_node_rows.get(
                            expected_ids(question)[0], {}
                        ).get("text_sha256"),
                        "new_text_sha256_normalized": sha256_text(
                            normalize_text(record["text"])
                        ),
                    }
                )

        final_union = " ".join(
            clean_snapshot.records[node_id]["text"]
            for node_id in direct_new_ids
        )
        final_checks = [
            {
                "fragment": check["fragment"],
                "present": (
                    compact_text(check["fragment"]) in compact_text(final_union)
                ),
            }
            for check in fragment_checks
        ]
        passed = bool(final_checks) and all(
            check["present"] for check in final_checks
        )
        if not passed:
            blocked.append(
                f"{question_id}: evidence fragments not found in mapped nodes"
            )
        for row in mapping_rows:
            if row["question_id"] != question_id:
                continue
            row["evidence_fragments_checked"] = " | ".join(
                check["fragment"] for check in final_checks
            )
            row["evidence_fragments_present"] = " | ".join(
                str(check["present"]) for check in final_checks
            )
            row["question_evidence_complete"] = passed
            row["focus_review"] = question_id in {"MODEL-09", "CROSS-09"}
            row["mapping_review_note"] = (
                "Old ID mapped through audited node_diff; required evidence "
                "fragments were checked against the mapped clean-index text."
            )
        question_checks.append(
            {
                "question_id": question_id,
                "expected_chunks_original": expected_ids(question),
                "expected_chunks_clean": direct_new_ids,
                "evidence_mode": evidence_mode(question_id),
                "evidence_fragments": final_checks,
                "evidence_complete": passed,
                "focus_review": question_id in {"MODEL-09", "CROSS-09"},
            }
        )

    if blocked:
        raise EvaluationBlocked(
            "Expected chunk mapping/evidence validation failed: " + " | ".join(blocked)
        )
    mapped_ids = {
        row["new_node_id"] for row in mapping_rows
    }
    if not all_expected.issubset(set(by_old)):
        raise EvaluationBlocked(
            "Some expected chunk IDs have no row in node_diff.csv."
        )
    if not mapped_ids.issubset(set(clean_snapshot.records)):
        raise EvaluationBlocked("Mapped IDs are missing from the clean index.")
    summary = {
        "expected_old_ids": len(all_expected),
        "mapped_clean_ids": len(mapped_ids),
        "mapping_row_count": len(mapping_rows),
        "supplementary_mapping_count": sum(
            row["mapping_role"] != "direct" for row in mapping_rows
        ),
        "question_count": len(dataset),
        "evidence_complete_questions": sum(
            row["evidence_complete"] for row in question_checks
        ),
        "focus_review": {
            key: next(
                row for row in question_checks if row["question_id"] == key
            )
            for key in ("MODEL-09", "CROSS-09")
        },
        "status": "passed",
    }
    return mapping_rows, summary


def question_mapping_ids(
    question_id: str,
    mapping_rows: list[dict[str, Any]],
) -> list[str]:
    return list(
        dict.fromkeys(
            row["new_node_id"]
            for row in mapping_rows
            if row["question_id"] == question_id
        )
    )


def evaluate_retrieval_result(
    question: dict[str, Any],
    ranked_node_ids: list[str],
    expected_clean_ids: list[str],
    retrieval_success: bool = True,
) -> dict[str, Any]:
    mode = evidence_mode(question["question_id"])
    expected = list(dict.fromkeys(expected_clean_ids))
    expected_set = set(expected)
    ranked = list(ranked_node_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, node_id in enumerate(ranked, start=1)
            if node_id in expected_set
        ),
        None,
    )
    result = {
        "question_id": question["question_id"],
        "category": question["category"],
        "evidence_mode": mode,
        "expected_chunks_original": expected_ids(question),
        "expected_chunks_clean": expected,
        "ranked_node_ids": ranked,
        "retrieval_success": retrieval_success,
        "first_relevant_rank": first_relevant_rank,
    }
    for cutoff in RETRIEVAL_CUTOFFS:
        top = ranked[:cutoff]
        hits = expected_set.intersection(top)
        denominator = len(expected_set) if mode == "required_all" else 1
        result[f"hit_at_{cutoff}"] = (
            1.0 if retrieval_success and hits else 0.0
        )
        result[f"recall_at_{cutoff}"] = (
            len(hits) / denominator
            if retrieval_success and denominator
            else 0.0
        )
        result[f"recall_num_at_{cutoff}"] = (
            len(hits) if retrieval_success else 0
        )
        result[f"recall_den_at_{cutoff}"] = denominator
    for cutoff in (3, 4):
        result[f"mrr_at_{cutoff}"] = (
            1.0 / first_relevant_rank
            if retrieval_success
            and first_relevant_rank is not None
            and first_relevant_rank <= cutoff
            else 0.0
        )
    result["mrr"] = result["mrr_at_4"]
    return result


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_retrieval_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful = [row for row in rows if row["retrieval_success"]]
    metrics: dict[str, Any] = {
        "attempted_questions": len(rows),
        "successful_questions": len(successful),
        "failed_questions": len(rows) - len(successful),
        "primary_denominator": len(rows),
        "successful_only_denominator": len(successful),
    }
    for cutoff in RETRIEVAL_CUTOFFS:
        hit_key = f"hit_at_{cutoff}"
        recall_key = f"recall_at_{cutoff}"
        metrics[hit_key] = mean([float(row[hit_key]) for row in rows])
        metrics[recall_key] = mean([float(row[recall_key]) for row in rows])
        metrics[f"{hit_key}_successful_only"] = mean(
            [float(row[hit_key]) for row in successful]
        )
        metrics[f"{recall_key}_successful_only"] = mean(
            [float(row[recall_key]) for row in successful]
        )
    for cutoff in (3, 4):
        key = f"mrr_at_{cutoff}"
        metrics[key] = mean([float(row[key]) for row in rows])
        metrics[f"{key}_successful_only"] = mean(
            [float(row[key]) for row in successful]
        )
    metrics["mrr"] = metrics["mrr_at_4"]
    metrics["formula"] = {
        "hit_rate_at_k": (
            "Number of attempted questions with at least one expected clean node "
            "in top K / all 50 attempted questions. API failures count as zero."
        ),
        "recall_at_k": (
            "Alternative evidence: 1 if any expected node is in top K, otherwise "
            "0; denominator 1. Required-all: number of expected nodes found in "
            "top K / number of jointly required nodes."
        ),
        "mrr_at_k": (
            "Mean of 1/first relevant rank for ranks <= K; no hit or failed "
            "retrieval contributes zero."
        ),
    }
    return metrics


def build_retrieval_metrics_bundle(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    categories = sorted({row["category"] for row in rows})
    by_category = []
    for category in ["ALL"] + categories:
        group = rows if category == "ALL" else [
            row for row in rows if row["category"] == category
        ]
        by_category.append({"category": category, **aggregate_retrieval_metrics(group)})
    overall = aggregate_retrieval_metrics(rows)
    payload = {
        "run_scope": "answerable_50",
        "overall": overall,
        "by_category": by_category,
        "per_question": rows,
        "evidence_semantics": {
            "alternative": (
                "expected_chunks are interchangeable valid evidence nodes; one "
                "hit satisfies the question-level recall denominator of 1."
            ),
            "required_all": (
                "expected_chunks are jointly required; recall equals matched "
                "required nodes divided by total required nodes."
            ),
            "required_all_questions": sorted(REQUIRED_ALL_QUESTIONS),
            "alternative_multi_questions": sorted(
                {
                    row["question_id"]
                    for row in rows
                    if len(row["expected_chunks_original"]) > 1
                    and row["evidence_mode"] == "alternative"
                }
            ),
        },
    }
    return payload, by_category, rows


def _extract_usage(response: Any) -> dict[str, int] | None:
    additional = getattr(response, "additional_kwargs", None)
    if isinstance(additional, dict):
        prompt = additional.get("prompt_tokens")
        completion = additional.get("completion_tokens")
        total = additional.get("total_tokens")
        if prompt is not None or completion is not None or total is not None:
            return {
                "prompt_tokens": int(prompt or 0),
                "completion_tokens": int(completion or 0),
                "total_tokens": int(total or 0),
            }
    raw = getattr(response, "raw", None)
    usage = getattr(raw, "usage", None)
    if usage is not None:
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }
    return None


class APICallRecorder:
    def __init__(self, run_id: str, sink_path: Path | None = None) -> None:
        self.run_id = run_id
        self.sink_path = sink_path
        self.records: list[dict[str, Any]] = []
        self._counter = 0

    def seed_existing(self, records: list[dict[str, Any]]) -> None:
        self.records = [dict(record) for record in records]
        suffix_numbers = []
        for record in records:
            match = re.search(r"-call-(\d+)$", str(record.get("request_id", "")))
            if match:
                suffix_numbers.append(int(match.group(1)))
        self._counter = max(suffix_numbers, default=len(records))

    def start(
        self,
        *,
        question_id: str,
        scope: str,
        provider: str,
        model: str,
        call_type: str,
        streaming: bool,
        retry: bool = False,
        fallback: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._counter += 1
        return {
            "request_id": f"{self.run_id}-call-{self._counter:06d}",
            "question_id": question_id,
            "scope": scope,
            "provider": provider,
            "model": model,
            "call_type": call_type,
            "streaming": streaming,
            "retry": retry,
            "fallback": fallback,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "request_started_at": utc_now(),
            "request_finished_at": None,
            "duration_ms": None,
            "status": "started",
            "error": None,
            "metadata": metadata or {},
        }

    def finish(
        self,
        record: dict[str, Any],
        *,
        status: str,
        error: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        started = datetime.fromisoformat(record["request_started_at"])
        finished = datetime.now(timezone.utc)
        record["request_finished_at"] = finished.isoformat()
        record["duration_ms"] = (finished - started).total_seconds() * 1000.0
        record["status"] = status
        record["error"] = error
        if usage:
            record["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
            record["completion_tokens"] = int(
                usage.get("completion_tokens", 0)
            )
            record["total_tokens"] = int(usage.get("total_tokens", 0))
        self.records.append(record)
        if self.sink_path is not None:
            append_jsonl(self.sink_path, [record])
        return record


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    llm_records = [
        record for record in records if record.get("call_type") == "llm"
    ]
    usage_records = [
        record for record in llm_records if record.get("total_tokens") is not None
    ]
    return {
        "prompt_tokens": sum(
            int(record["prompt_tokens"] or 0) for record in usage_records
        ),
        "completion_tokens": sum(
            int(record["completion_tokens"] or 0) for record in usage_records
        ),
        "total_tokens": sum(
            int(record["total_tokens"] or 0) for record in usage_records
        ),
        "llm_calls": len(llm_records),
        "usage_reported_calls": len(usage_records),
        "usage_complete": len(usage_records) == len(llm_records),
    }


class RecordedLLM(OpenAICompatible):
    _recorder: APICallRecorder = PrivateAttr()
    _provider_secrets: list[str] = PrivateAttr(default_factory=list)
    _question_id: str = PrivateAttr(default="")
    _scope: str = PrivateAttr(default="")
    _next_role: str = PrivateAttr(default="primary")
    _next_retry: bool = PrivateAttr(default=False)
    _next_fallback: bool = PrivateAttr(default=False)

    def __init__(
        self,
        *,
        recorder: APICallRecorder,
        provider_secrets: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._recorder = recorder
        self._provider_secrets = provider_secrets

    def begin_question(self, question_id: str, scope: str) -> None:
        self._question_id = question_id
        self._scope = scope
        self.configure_next_call("primary")

    def configure_next_call(
        self,
        call_role: str,
        *,
        retry: bool = False,
        fallback: bool = False,
    ) -> None:
        self._next_role = call_role
        self._next_retry = retry
        self._next_fallback = fallback

    def _take_call_options(self) -> tuple[str, bool, bool]:
        values = (
            self._next_role,
            self._next_retry,
            self._next_fallback,
        )
        self._next_role = "primary"
        self._next_retry = False
        self._next_fallback = False
        return values

    def _start_record(self, streaming: bool) -> dict[str, Any]:
        role, retry, fallback = self._take_call_options()
        return self._recorder.start(
            question_id=self._question_id,
            scope=self._scope,
            provider="openai_compatible",
            model=self.model,
            call_type="llm",
            streaming=streaming,
            retry=retry,
            fallback=fallback,
            metadata={"call_role": role},
        )

    def chat(self, *args, **kwargs) -> ChatResponse:
        record = self._start_record(streaming=False)
        try:
            response = super().chat(*args, **kwargs)
            self._recorder.finish(
                record,
                status="success",
                usage=_extract_usage(response),
            )
            return response
        except Exception as exc:
            self._recorder.finish(
                record,
                status="failed",
                error=sanitize_error(exc, self._provider_secrets),
            )
            raise

    def stream_chat(self, *args, **kwargs):
        record = self._start_record(streaming=True)
        try:
            source = super().stream_chat(*args, **kwargs)
        except Exception as exc:
            self._recorder.finish(
                record,
                status="failed",
                error=sanitize_error(exc, self._provider_secrets),
            )
            raise

        def _wrapped():
            usage = None
            status = "success"
            error = None
            try:
                for chunk in source:
                    chunk_usage = _extract_usage(chunk)
                    if chunk_usage is not None:
                        usage = chunk_usage
                    yield chunk
            except Exception as exc:
                status = "failed"
                error = sanitize_error(exc, self._provider_secrets)
                raise
            finally:
                self._recorder.finish(
                    record,
                    status=status,
                    error=error,
                    usage=usage,
                )

        return _wrapped()


class FailureTracker:
    def __init__(self, threshold: int) -> None:
        if threshold < 1:
            raise ValueError("failure threshold must be >= 1")
        self.threshold = threshold
        self.consecutive = 0

    def record(self, failed: bool) -> None:
        self.consecutive = self.consecutive + 1 if failed else 0

    def seed(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self.record(bool(row.get("had_api_failure")))

    def should_pause(self) -> bool:
        return self.consecutive >= self.threshold


class StaticSequenceRetriever(BaseRetriever):
    _results: list[NodeWithScore] = PrivateAttr(default_factory=list)

    def __init__(self, results: list[NodeWithScore]) -> None:
        super().__init__()
        self._results = list(results)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._results


def node_hit(node_with_score: NodeWithScore, rank: int) -> dict[str, Any]:
    metadata = node_with_score.node.metadata or {}
    score = node_with_score.score
    return {
        "rank": rank,
        "node_id": node_with_score.node.node_id,
        "document": metadata.get("original_name") or metadata.get("file_name"),
        "physical_page": metadata.get("source_page"),
        "page_label": metadata.get("page_label"),
        "score": float(score) if score is not None else None,
        "text": node_with_score.node.get_content(),
    }


@dataclass
class CleanRetrievalRuntime:
    cfg: AppSettings
    snapshot: CleanIndexSnapshot
    vector_store: Any
    nodes: list[Any]
    bm25_retriever: BM25Retriever
    embedding_model: Any
    reranker: DashScopeRerank
    recorder: APICallRecorder
    secrets: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        snapshot: CleanIndexSnapshot,
        recorder: APICallRecorder,
        cfg: AppSettings | None = None,
    ) -> "CleanRetrievalRuntime":
        cfg = cfg or AppSettings()  # type: ignore[call-arg]
        if cfg.embedding_provider.strip().lower() != "dashscope":
            raise EvaluationBlocked(
                f"Unexpected embedding provider: {cfg.embedding_provider}"
            )
        if cfg.embedding_model.strip() != "text-embedding-v3":
            raise EvaluationBlocked(
                f"Unexpected embedding model: {cfg.embedding_model}"
            )
        if cfg.rerank_model.strip() != "gte-rerank-v2":
            raise EvaluationBlocked(f"Unexpected reranker: {cfg.rerank_model}")
        if cfg.rerank_top_n != 4:
            raise EvaluationBlocked(
                f"Expected production rerank_top_n=4, got {cfg.rerank_top_n}"
            )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from src.retriever import _chinese_tokenizer

        vector_store = ChromaVectorStore(
            chroma_collection=snapshot.collection
        )
        nodes = list(snapshot.docstore.docs.values())
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=max(cfg.bm25_top_k * 3, 15),
            tokenizer=_chinese_tokenizer,
        )
        embedding_model = _build_embedding(cfg)
        reranker = DashScopeRerank(
            model=cfg.rerank_model,
            top_n=cfg.rerank_top_n,
            api_key=cfg.dashscope_api_key,
        )
        Settings.embed_model = embedding_model
        Settings.llm = None
        secrets = [cfg.dashscope_api_key]
        if cfg.openai_api_key:
            secrets.append(cfg.openai_api_key.get_secret_value())
        return cls(
            cfg=cfg,
            snapshot=snapshot,
            vector_store=vector_store,
            nodes=nodes,
            bm25_retriever=bm25_retriever,
            embedding_model=embedding_model,
            reranker=reranker,
            recorder=recorder,
            secrets=secrets,
            parameters={
                "embedding_provider": cfg.embedding_provider,
                "embedding_model": cfg.embedding_model,
                "rerank_model": cfg.rerank_model,
                "vector_candidate_k": max(cfg.vector_top_k * 3, 15),
                "bm25_candidate_k": max(cfg.bm25_top_k * 3, 15),
                "rrf_candidate_pool_k": cfg.vector_top_k + cfg.bm25_top_k,
                "rrf_mode": "reciprocal_rerank",
                "rrf_num_queries": 1,
                "rrf_use_async": False,
                "final_top_k": cfg.rerank_top_n,
                "production_rerank_top_n": cfg.rerank_top_n,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
            },
        )

    def retrieve(
        self,
        *,
        question_id: str,
        scope: str,
        query: str,
    ) -> dict[str, Any]:
        timings = {
            "embedding_ms": None,
            "vector_ms": None,
            "bm25_ms": None,
            "rrf_ms": None,
            "rerank_ms": None,
        }
        local_errors: list[dict[str, str]] = []
        embedding_record = self.recorder.start(
            question_id=question_id,
            scope=scope,
            provider="dashscope",
            model=self.cfg.embedding_model,
            call_type="embedding",
            streaming=False,
            metadata={"input_characters": len(query)},
        )
        started = time.perf_counter()
        try:
            embedding = self.embedding_model.get_query_embedding(query)
            timings["embedding_ms"] = (time.perf_counter() - started) * 1000
            self.recorder.finish(embedding_record, status="success")
        except Exception as exc:
            timings["embedding_ms"] = (time.perf_counter() - started) * 1000
            self.recorder.finish(
                embedding_record,
                status="failed",
                error=sanitize_error(exc, self.secrets),
            )
            return {
                "status": "failed",
                "error_stage": "query_embedding",
                "error": sanitize_error(exc, self.secrets),
                "local_errors": local_errors,
                "top4": [],
                "ranked_node_ids": [],
                "rrf_candidate_ids": [],
                **timings,
            }

        vector_results: list[NodeWithScore] = []
        vector_error = None
        started = time.perf_counter()
        try:
            vector_result = self.vector_store.query(
                VectorStoreQuery(
                    query_embedding=embedding,
                    similarity_top_k=self.parameters["vector_candidate_k"],
                )
            )
            similarities = vector_result.similarities or []
            vector_results = [
                NodeWithScore(
                    node=node,
                    score=(
                        float(similarities[index])
                        if index < len(similarities)
                        else None
                    ),
                )
                for index, node in enumerate(vector_result.nodes or [])
            ]
        except Exception as exc:
            vector_error = sanitize_error(exc, self.secrets)
            local_errors.append(
                {"stage": "vector", "error_type": type(exc).__name__, "error": vector_error}
            )
        finally:
            timings["vector_ms"] = (time.perf_counter() - started) * 1000

        bm25_results: list[NodeWithScore] = []
        bm25_error = None
        started = time.perf_counter()
        try:
            bm25_results = self.bm25_retriever.retrieve(
                QueryBundle(query_str=query)
            )
        except Exception as exc:
            bm25_error = sanitize_error(exc, self.secrets)
            local_errors.append(
                {"stage": "bm25", "error_type": type(exc).__name__, "error": bm25_error}
            )
        finally:
            timings["bm25_ms"] = (time.perf_counter() - started) * 1000

        fused_results: list[NodeWithScore] = []
        fusion_error = None
        started = time.perf_counter()
        try:
            fusion = QueryFusionRetriever(
                retrievers=[
                    StaticSequenceRetriever(vector_results),
                    StaticSequenceRetriever(bm25_results),
                ],
                similarity_top_k=self.parameters["rrf_candidate_pool_k"],
                num_queries=self.parameters["rrf_num_queries"],
                mode=self.parameters["rrf_mode"],
                use_async=self.parameters["rrf_use_async"],
                verbose=False,
            )
            fused_results = fusion.retrieve(QueryBundle(query_str=query))
        except Exception as exc:
            fusion_error = sanitize_error(exc, self.secrets)
            local_errors.append(
                {"stage": "rrf", "error_type": type(exc).__name__, "error": fusion_error}
            )
            fused_results = vector_results or bm25_results
        finally:
            timings["rrf_ms"] = (time.perf_counter() - started) * 1000

        if not fused_results:
            return {
                "status": "failed",
                "error_stage": "rrf",
                "error": fusion_error or "No vector/BM25 candidates available",
                "local_errors": local_errors,
                "top4": [],
                "ranked_node_ids": [],
                "rrf_candidate_ids": [],
                **timings,
            }

        rerank_record = self.recorder.start(
            question_id=question_id,
            scope=scope,
            provider="dashscope",
            model=self.cfg.rerank_model,
            call_type="reranker",
            streaming=False,
            metadata={
                "input_documents": len(fused_results),
                "input_characters": sum(
                    len(item.node.get_content()) for item in fused_results
                ),
            },
        )
        rerank_status = "success"
        started = time.perf_counter()
        try:
            final_results = self.reranker.postprocess_nodes(
                fused_results,
                query_bundle=QueryBundle(query_str=query),
            )[: self.cfg.rerank_top_n]
            self.recorder.finish(rerank_record, status="success")
        except Exception as exc:
            rerank_status = "fallback_to_rrf"
            local_errors.append(
                {
                    "stage": "rerank",
                    "error_type": type(exc).__name__,
                    "error": sanitize_error(exc, self.secrets),
                }
            )
            self.recorder.finish(
                rerank_record,
                status="failed",
                error=sanitize_error(exc, self.secrets),
            )
            final_results = fused_results[: self.cfg.rerank_top_n]
        finally:
            timings["rerank_ms"] = (time.perf_counter() - started) * 1000

        top4 = [
            node_hit(item, rank)
            for rank, item in enumerate(final_results, start=1)
        ]
        return {
            "status": "success" if top4 else "failed",
            "error_stage": None if top4 else "final_retrieval",
            "error": None if top4 else "Final retrieval returned no nodes",
            "local_errors": local_errors,
            "rerank_status": rerank_status,
            "top4": top4,
            "ranked_node_ids": [item["node_id"] for item in top4],
            "rrf_candidate_ids": [
                item.node.node_id for item in fused_results
            ],
            **timings,
        }


def _configured_llm(
    runtime: CleanRetrievalRuntime,
    recorder: APICallRecorder,
) -> RecordedLLM:
    return RecordedLLM(
        recorder=recorder,
        provider_secrets=runtime.secrets,
        model=runtime.cfg.llm_model,
        api_key=runtime.cfg.openai_api_key.get_secret_value().strip(),
        api_base=runtime.cfg.openai_api_base,
        max_tokens=2048,
    )


def execute_question(
    *,
    runtime: CleanRetrievalRuntime,
    llm: RecordedLLM,
    question: dict[str, Any],
    mapping_rows: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    question_id = question["question_id"]
    request_started = time.perf_counter()
    call_start = len(runtime.recorder.records)
    llm.begin_question(question_id, scope)
    retrieval = runtime.retrieve(
        question_id=question_id,
        scope=scope,
        query=question["question"],
    )
    base: dict[str, Any] = {
        "question_id": question_id,
        "category": question.get("category", "无答案"),
        "question": question["question"],
        "question_language": question.get("question_language", "zh"),
        "scope": scope,
        "retrieved": retrieval["top4"],
        "retrieved_node_ids": retrieval["ranked_node_ids"],
        "rrf_candidate_ids": retrieval["rrf_candidate_ids"],
        "retrieval_status": retrieval["status"],
        "retrieval_error": retrieval.get("error"),
        "rerank_status": retrieval.get("rerank_status"),
        "embedding_ms": retrieval["embedding_ms"],
        "vector_ms": retrieval["vector_ms"],
        "bm25_ms": retrieval["bm25_ms"],
        "rrf_ms": retrieval["rrf_ms"],
        "rerank_ms": retrieval["rerank_ms"],
        "retrieval_ms": sum(
            value or 0.0
            for value in (
                retrieval["embedding_ms"],
                retrieval["vector_ms"],
                retrieval["bm25_ms"],
                retrieval["rrf_ms"],
                retrieval["rerank_ms"],
            )
        ),
        "llm_first_token_ms": None,
        "llm_generation_ms": None,
        "total_ms": None,
        "answer": "",
        "execution_status": "failed",
        "api_error_message": retrieval.get("error"),
        "local_retrieval_errors": retrieval["local_errors"],
    }
    if scope == "answer":
        base["retrieval_evaluation"] = evaluate_retrieval_result(
            question,
            retrieval["ranked_node_ids"],
            question_mapping_ids(question_id, mapping_rows),
            retrieval_success=retrieval["status"] == "success",
        )
    if retrieval["status"] != "success":
        base["total_ms"] = (time.perf_counter() - request_started) * 1000
        calls = runtime.recorder.records[call_start:]
        base["usage"] = summarize_usage(calls)
        base["had_api_failure"] = any(
            record["status"] == "failed" for record in calls
        )
        return base

    nodes = [
        NodeWithScore(
            node=runtime.snapshot.docstore.docs[hit["node_id"]],
            score=hit.get("score"),
        )
        for hit in retrieval["top4"]
    ]
    engine = CondensePlusContextChatEngine.from_defaults(
        retriever=StaticSequenceRetriever(nodes),
        llm=llm,
        memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
        system_prompt=_build_system_prompt(DEFAULT_STYLE),
        skip_condense=True,
    )
    Settings.llm = llm
    generation_started = time.perf_counter()
    chunks: list[str] = []
    first_token_ms = None
    stream_error = None

    llm.configure_next_call("primary_stream")
    try:
        stream = engine.stream_chat(question["question"])
        for token in stream.response_gen:
            if token:
                if first_token_ms is None:
                    first_token_ms = (
                        time.perf_counter() - generation_started
                    ) * 1000
                chunks.append(token)
    except Exception as exc:
        stream_error = sanitize_error(exc, runtime.secrets)

    if not chunks:
        llm.configure_next_call(
            "nonstream_fallback",
            retry=True,
            fallback=True,
        )
        try:
            fallback_response = engine.chat(question["question"])
            fallback_text = str(fallback_response).strip()
            if fallback_text:
                chunks.append(fallback_text)
        except Exception as exc:
            if stream_error is None:
                stream_error = sanitize_error(exc, runtime.secrets)
            else:
                stream_error += " | fallback=" + sanitize_error(
                    exc, runtime.secrets
                )

    base["answer"] = "".join(chunks).strip()
    base["llm_first_token_ms"] = first_token_ms
    base["llm_generation_ms"] = (
        time.perf_counter() - generation_started
    ) * 1000
    base["total_ms"] = (time.perf_counter() - request_started) * 1000
    calls = runtime.recorder.records[call_start:]
    usage = summarize_usage(calls)
    base["usage"] = usage
    base["prompt_tokens"] = (
        usage["prompt_tokens"] if usage["usage_reported_calls"] else None
    )
    base["completion_tokens"] = (
        usage["completion_tokens"] if usage["usage_reported_calls"] else None
    )
    base["total_tokens"] = (
        usage["total_tokens"] if usage["usage_complete"] else None
    )
    base["had_api_failure"] = any(
        record["status"] == "failed" for record in calls
    )
    if base["answer"]:
        base["execution_status"] = "completed"
    else:
        base["api_error_message"] = stream_error or "LLM returned no non-empty answer"
    return base


def classify_abstention(row: dict[str, Any]) -> str:
    if row.get("api_error_message") or not row.get("answer"):
        return "api_error_or_empty"
    if contains_refusal(row["answer"]):
        return "correct_refusal"
    return "incorrect_answer"


def evaluate_quality_outputs(
    answer_rows: list[dict[str, Any]],
    dataset_by_id: dict[str, dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quality_rows = []
    citation_rows = []
    faithfulness_rows = []
    for row in answer_rows:
        question = dataset_by_id[row["question_id"]]
        adapted = {
            **question,
            "expected_chunks": question_mapping_ids(
                question["question_id"], mapping_rows
            ),
        }
        error = row.get("api_error_message")
        quality = evaluate_answer_quality(question, row["answer"], error)
        citations = evaluate_citations(
            adapted,
            row["answer"],
            row.get("retrieved") or [],
        )
        faithfulness = evaluate_faithfulness(
            row["answer"],
            row.get("retrieved") or [],
        )
        base = {
            "question_id": row["question_id"],
            "category": row["category"],
            "question": row["question"],
            "answer": row["answer"],
            "expected_answer": question["expected_answer"],
            "evaluation_points": question["evaluation_points"],
            "retrieved": row.get("retrieved") or [],
        }
        quality_rows.append({**base, **quality})
        citation_rows.append({**base, **citations})
        faithfulness_rows.append({**base, **faithfulness})
    return quality_rows, citation_rows, faithfulness_rows


def summarize_answer_quality(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = sorted({row["category"] for row in rows})
    by_category = []
    for category in ["ALL"] + categories:
        group = rows if category == "ALL" else [
            row for row in rows if row["category"] == category
        ]
        counts = Counter(row["status"] for row in group)
        point_total = sum(row["total_points"] for row in group)
        point_covered = sum(row["covered_points"] for row in group)
        by_category.append(
            {
                "category": category,
                "questions": len(group),
                "completely_correct_auto": counts["completely_correct"],
                "partially_correct_auto": counts["partially_correct"],
                "wrong_auto": counts["wrong"],
                "indeterminate_auto": counts["indeterminate"],
                "answer_accuracy_auto": (
                    counts["completely_correct"]
                    / max(1, len(group) - counts["indeterminate"])
                ),
                "point_coverage_auto": (
                    point_covered / point_total if point_total else None
                ),
                "review_status": "preliminary_requires_human_review",
            }
        )
    return by_category, {
        "overall": next(row for row in by_category if row["category"] == "ALL"),
        "by_category": by_category,
        "per_question": rows,
        "method": "evaluation_point_deterministic_coverage_not_text_similarity",
        "human_review_required": True,
    }


def summarize_citations(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = sorted({row["category"] for row in rows})
    by_category = []
    for category in ["ALL"] + categories:
        group = rows if category == "ALL" else [
            row for row in rows if row["category"] == category
        ]
        citation_count = sum(row["citation_count"] for row in group)
        supported = sum(row["supported_citation_count"] for row in group)
        completeness = [
            row["citation_completeness"]
            for row in group
            if row["citation_completeness"] is not None
        ]
        by_category.append(
            {
                "category": category,
                "questions": len(group),
                "citation_count": citation_count,
                "supported_citation_count": supported,
                "citation_accuracy": (
                    supported / citation_count if citation_count else None
                ),
                "citation_completeness": (
                    sum(completeness) / len(completeness)
                    if completeness
                    else None
                ),
                "review_status": "preliminary_requires_human_review",
            }
        )
    return by_category, {
        "overall": next(row for row in by_category if row["category"] == "ALL"),
        "by_category": by_category,
        "per_question": rows,
        "method": "citation_index_to_retrieved_node_span_and_lexical_support",
        "human_review_required": True,
    }


def summarize_faithfulness(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = sorted({row["category"] for row in rows})
    by_category = []
    for category in ["ALL"] + categories:
        group = rows if category == "ALL" else [
            row for row in rows if row["category"] == category
        ]
        factual = sum(row["factual_statement_count"] for row in group)
        supported = sum(row["supported_statement_count"] for row in group)
        partial = sum(
            row["partially_supported_statement_count"] for row in group
        )
        unsupported = sum(row["unsupported_statement_count"] for row in group)
        conflict = sum(row["conflicting_statement_count"] for row in group)
        by_category.append(
            {
                "category": category,
                "questions": len(group),
                "grounded_answers": sum(
                    row["grounded_answer"] for row in group
                ),
                "grounded_answer_ratio": (
                    sum(row["grounded_answer"] for row in group) / len(group)
                    if group
                    else None
                ),
                "factual_statements": factual,
                "supported_statements": supported,
                "partially_supported_statements": partial,
                "unsupported_statements": unsupported,
                "conflicting_statements": conflict,
                "supported_ratio": supported / factual if factual else None,
                "unsupported_ratio": unsupported / factual if factual else None,
                "review_status": "preliminary_requires_human_review",
            }
        )
    return by_category, {
        "overall": next(row for row in by_category if row["category"] == "ALL"),
        "by_category": by_category,
        "per_question": rows,
        "method": "deterministic_sentence_to_retrieved_text_support",
        "human_review_required": True,
    }


def summarize_performance(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    fields = (
        "embedding_ms",
        "vector_ms",
        "bm25_ms",
        "rrf_ms",
        "rerank_ms",
        "retrieval_ms",
        "llm_first_token_ms",
        "llm_generation_ms",
        "total_ms",
    )
    payload: dict[str, Any] = {
        "scope": scope,
        "question_count": len(rows),
        "metrics": {},
    }
    for field_name in fields:
        values = sorted(
            float(row[field_name])
            for row in rows
            if row.get(field_name) is not None
        )
        if values:
            position50 = (len(values) - 1) * 0.5
            position95 = (len(values) - 1) * 0.95
            lower50 = math.floor(position50)
            upper50 = math.ceil(position50)
            lower95 = math.floor(position95)
            upper95 = math.ceil(position95)
            p50 = (
                values[lower50]
                if lower50 == upper50
                else values[lower50] * (1 - (position50 - lower50))
                + values[upper50] * (position50 - lower50)
            )
            p95 = (
                values[lower95]
                if lower95 == upper95
                else values[lower95] * (1 - (position95 - lower95))
                + values[upper95] * (position95 - lower95)
            )
            payload["metrics"][field_name] = {
                "count": len(values),
                "mean": sum(values) / len(values),
                "p50": p50,
                "p95": p95,
                "max": max(values),
            }
        else:
            payload["metrics"][field_name] = {
                "count": 0,
                "mean": None,
                "p50": None,
                "p95": None,
                "max": None,
            }
    return payload


def build_cost_metrics(api_records: list[dict[str, Any]]) -> dict[str, Any]:
    def group(call_type: str) -> list[dict[str, Any]]:
        return [
            record for record in api_records
            if record.get("call_type") == call_type
        ]

    embedding = group("embedding")
    reranker = group("reranker")
    llm = group("llm")
    usage = summarize_usage(api_records)
    return {
        "embedding": {
            "calls": len(embedding),
            "successful_calls": sum(row["status"] == "success" for row in embedding),
            "failed_calls": sum(row["status"] == "failed" for row in embedding),
            "actual_cost": "unknown",
            "cost_status": "provider_pricing_not_available",
        },
        "reranker": {
            "calls": len(reranker),
            "successful_calls": sum(row["status"] == "success" for row in reranker),
            "failed_calls": sum(row["status"] == "failed" for row in reranker),
            "actual_cost": "unknown",
            "cost_status": "provider_pricing_not_available",
        },
        "llm": {
            "calls": len(llm),
            "successful_calls": sum(row["status"] == "success" for row in llm),
            "failed_calls": sum(row["status"] == "failed" for row in llm),
            "input_tokens": usage["prompt_tokens"],
            "output_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "usage_complete": usage["usage_complete"],
            "streaming_calls": sum(row["streaming"] for row in llm),
            "nonstream_calls": sum(not row["streaming"] for row in llm),
            "retry_calls": sum(row["retry"] for row in llm),
            "fallback_calls": sum(row["fallback"] for row in llm),
            "actual_cost": "unknown",
            "cost_status": "free_tier_unverified",
        },
        "total_actual_cost": "unknown",
    }


def summarize_abstention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"] for row in rows)
    return {
        "attempted": len(rows),
        "correct_refusals": counts["correct_refusal"],
        "incorrect_answers": counts["incorrect_answer"],
        "api_errors_or_empty": counts["api_error_or_empty"],
        "correct_refusal_rate_all_attempts": (
            counts["correct_refusal"] / len(rows) if rows else None
        ),
        "unsupported_or_fabricated_count": counts["incorrect_answer"],
        "review_status": "preliminary_requires_human_review",
        "by_question": rows,
    }


OUTPUT_CONTRACT_FILES = (
    "answer_results.jsonl",
    "abstention_results.jsonl",
    "retrieval_metrics.json",
    "answer_metrics.json",
    "citation_metrics.json",
    "faithfulness_metrics.json",
    "abstention_metrics.json",
    "performance_metrics.json",
    "cost_metrics.json",
    "retrieval_metrics_by_category.csv",
    "answer_metrics_by_category.csv",
    "per_question_results.csv",
    "api_usage.jsonl",
    "RUN_STATUS.json",
    "RESUME_CHECKPOINT.json",
    "evaluation_report.md",
)


def initialize_output_contract(run_dir: Path, run_id: str) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: run_dir / name for name in OUTPUT_CONTRACT_FILES}
    for name in ("answer_results.jsonl", "abstention_results.jsonl", "api_usage.jsonl"):
        paths[name].touch(exist_ok=True)
    atomic_write_json(
        paths["RUN_STATUS.json"],
        {
            "run_id": run_id,
            "run_status": "initialized",
            "completed_answer_ids": [],
            "completed_no_answer_ids": [],
            "updated_at": utc_now(),
        },
    )
    atomic_write_json(
        paths["RESUME_CHECKPOINT.json"],
        {
            "run_id": run_id,
            "completed_answer_ids": [],
            "completed_no_answer_ids": [],
            "created_at": utc_now(),
        },
    )
    return paths


def upsert_jsonl_by_question_id(
    path: Path,
    row: dict[str, Any],
) -> None:
    existing = read_jsonl(path) if path.exists() else []
    question_id = row["question_id"]
    if any(item.get("question_id") == question_id for item in existing):
        return
    append_jsonl(path, [row])


def write_evaluation_outputs(
    *,
    run_dir: Path,
    run_id: str,
    answer_rows: list[dict[str, Any]],
    abstention_rows: list[dict[str, Any]],
    api_records: list[dict[str, Any]],
    mapping_rows: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    mapping_summary: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    retrieval_metrics, retrieval_by_category, retrieval_per_question = (
        build_retrieval_metrics_bundle(
            [row["retrieval_evaluation"] for row in answer_rows]
        )
    )
    dataset_by_id = {row["question_id"]: row for row in dataset}
    quality_rows, citation_rows, faithfulness_rows = evaluate_quality_outputs(
        answer_rows,
        dataset_by_id,
        mapping_rows,
    )
    answer_by_category, answer_metrics = summarize_answer_quality(quality_rows)
    citation_by_category, citation_metrics = summarize_citations(citation_rows)
    faith_by_category, faithfulness_metrics = summarize_faithfulness(
        faithfulness_rows
    )
    abstention_metrics = summarize_abstention(abstention_rows)
    performance = {
        "answer_50": summarize_performance(answer_rows, "answer_50"),
        "no_answer_10": summarize_performance(abstention_rows, "no_answer_10"),
        "all_60": summarize_performance(
            answer_rows + abstention_rows,
            "all_60",
        ),
    }
    cost = build_cost_metrics(api_records)
    quality_by_id = {row["question_id"]: row for row in quality_rows}
    citation_by_id = {row["question_id"]: row for row in citation_rows}
    faithfulness_by_id = {
        row["question_id"]: row for row in faithfulness_rows
    }

    atomic_write_json(run_dir / "retrieval_metrics.json", retrieval_metrics)
    atomic_write_json(run_dir / "answer_metrics.json", answer_metrics)
    atomic_write_json(run_dir / "citation_metrics.json", citation_metrics)
    atomic_write_json(run_dir / "faithfulness_metrics.json", faithfulness_metrics)
    atomic_write_json(run_dir / "abstention_metrics.json", abstention_metrics)
    atomic_write_json(run_dir / "performance_metrics.json", performance)
    atomic_write_json(run_dir / "cost_metrics.json", cost)
    write_csv(run_dir / "retrieval_metrics_by_category.csv", retrieval_by_category)
    write_csv(run_dir / "answer_metrics_by_category.csv", answer_by_category)
    per_question = []
    for row in answer_rows:
        per_question.append(
            {
                "question_id": row["question_id"],
                "kind": "answerable",
                "category": row["category"],
                "question": row["question"],
                "execution_status": row["execution_status"],
                "answer": row["answer"],
                "retrieved_node_ids": " | ".join(row["retrieved_node_ids"]),
                "hit_at_3": row["retrieval_evaluation"]["hit_at_3"],
                "recall_at_3": row["retrieval_evaluation"]["recall_at_3"],
                "mrr_at_4": row["retrieval_evaluation"]["mrr_at_4"],
                "answer_status_auto": quality_by_id[row["question_id"]]["status"],
                "point_coverage_auto": quality_by_id[row["question_id"]]["coverage"],
                "citation_accuracy_auto": citation_by_id[row["question_id"]]["citation_accuracy"],
                "citation_completeness_auto": citation_by_id[row["question_id"]]["citation_completeness"],
                "grounded_answer_auto": faithfulness_by_id[row["question_id"]]["grounded_answer"],
                "supported_statement_ratio_auto": faithfulness_by_id[row["question_id"]]["supported_ratio"],
                "unsupported_statement_ratio_auto": faithfulness_by_id[row["question_id"]]["unsupported_ratio"],
                "total_ms": row["total_ms"],
                "api_error": row.get("api_error_message"),
            }
        )
    for row in abstention_rows:
        per_question.append(
            {
                "question_id": row["question_id"],
                "kind": "no_answer",
                "category": row["category"],
                "question": row["question"],
                "execution_status": row["execution_status"],
                "answer": row["answer"],
                "retrieved_node_ids": " | ".join(row["retrieved_node_ids"]),
                "hit_at_3": "",
                "recall_at_3": "",
                "mrr_at_4": "",
                "answer_status_auto": row["classification"],
                "point_coverage_auto": "",
                "citation_accuracy_auto": "",
                "citation_completeness_auto": "",
                "grounded_answer_auto": "",
                "supported_statement_ratio_auto": "",
                "unsupported_statement_ratio_auto": "",
                "total_ms": row["total_ms"],
                "api_error": row.get("api_error_message"),
            }
        )
    write_csv(run_dir / "per_question_results.csv", per_question)

    answer_summary = answer_metrics["overall"]
    retrieval_summary = retrieval_metrics["overall"]
    correct_examples = [
        row for row in quality_rows if row["status"] == "completely_correct"
    ][:3]
    error_examples = [
        row for row in quality_rows
        if row["status"] in {"partially_correct", "wrong", "indeterminate"}
    ][:5]
    retrieval_issues = [
        row for row in retrieval_per_question
        if not row["retrieval_success"] or row["hit_at_3"] == 0
    ][:5]
    citation_issues = [
        row
        for row in citation_rows
        if row["citation_accuracy"] is None
        or row["citation_accuracy"] < 1
        or (row["citation_completeness"] or 0) < 1
    ][:5]
    bottleneck_metrics = {
        key: value["mean"]
        for key, value in performance["answer_50"]["metrics"].items()
        if value["mean"] is not None
    }
    bottleneck_name = (
        max(bottleneck_metrics, key=bottleneck_metrics.get)
        if bottleneck_metrics
        else None
    )
    improvement_priorities = []
    if (retrieval_summary["hit_at_3"] or 0) < 1:
        improvement_priorities.append("Improve retrieval evidence coverage and ranking for failed Hit@3 questions.")
    if (answer_summary["point_coverage_auto"] or 0) < 1:
        improvement_priorities.append("Improve answer completeness against evaluation points.")
    if (
        citation_metrics["overall"]["citation_accuracy"] is None
        or citation_metrics["overall"]["citation_accuracy"] < 1
    ):
        improvement_priorities.append("Strengthen citation support and necessary-evidence coverage.")
    if abstention_metrics["incorrect_answers"]:
        improvement_priorities.append("Strengthen abstention for absent-information questions.")
    if bottleneck_name:
        improvement_priorities.append(
            f"Profile the largest average stage: {bottleneck_name}."
        )
    improvement_priorities = improvement_priorities[:3] or [
        "No metric-driven repair is indicated; retain the frozen configuration."
    ]
    report_lines = [
        "# Clean Index Full RAG Evaluation",
        "",
        f"- Run: `{run_id}`",
        f"- Index fingerprint: `{preflight['index']['fingerprint']}`",
        f"- Answerable questions: {len(answer_rows)}",
        f"- No-answer questions: {len(abstention_rows)}",
        f"- Expected chunk mapping: {mapping_summary['status']}",
        "",
        "## Retrieval",
        "",
        f"- Hit Rate@3: {retrieval_summary['hit_at_3']}",
        f"- Recall@3: {retrieval_summary['recall_at_3']}",
        f"- MRR@4: {retrieval_summary['mrr_at_4']}",
        f"- Attempted denominator: {retrieval_summary['primary_denominator']}",
        "",
        "## Answer quality (automatic preliminary)",
        "",
        f"- Answer accuracy: {answer_summary['answer_accuracy_auto']}",
        f"- Point coverage: {answer_summary['point_coverage_auto']}",
        "",
        "## Citation and faithfulness",
        "",
        f"- Citation accuracy: {citation_metrics['overall']['citation_accuracy']}",
        f"- Citation completeness: {citation_metrics['overall']['citation_completeness']}",
        f"- Grounded answer ratio: {faithfulness_metrics['overall']['grounded_answer_ratio']}",
        f"- Unsupported statement ratio: {faithfulness_metrics['overall']['unsupported_ratio']}",
        "",
        "## Abstention",
        "",
        f"- Correct refusals: {abstention_metrics['correct_refusals']}/{len(abstention_rows)}",
        f"- Incorrect answers: {abstention_metrics['incorrect_answers']}",
        "",
        "## Cost",
        "",
        "- Embedding cost: unknown",
        "- Reranker cost: unknown",
        f"- LLM actual cost: {cost['llm']['actual_cost']}",
        "",
        "## Correct examples",
        "",
        *(
            [
                f"- `{row['question_id']}`: {normalize_text(row['answer'])[:220]}"
                for row in correct_examples
            ]
            or ["- None detected by the preliminary automatic check."]
        ),
        "",
        "## Error and retrieval cases",
        "",
        *(
            [
                f"- `{row['question_id']}` answer status `{row['status']}`; "
                f"coverage={row['coverage']}"
                for row in error_examples
            ]
            or ["- No automatic error example was detected."]
        ),
        *(
            [
                f"- `{row['question_id']}` retrieval hit@3={row['hit_at_3']}, "
                f"first rank={row['first_relevant_rank']}"
                for row in retrieval_issues
            ]
            or ["- No retrieval failure among the displayed candidates."]
        ),
        "",
        "## Citation issues",
        "",
        *(
            [
                f"- `{row['question_id']}` accuracy={row['citation_accuracy']}, "
                f"completeness={row['citation_completeness']}"
                for row in citation_issues
            ]
            or ["- No automatic citation issue was detected."]
        ),
        "",
        "## Improvement priorities",
        "",
        *[f"{index}. {item}" for index, item in enumerate(improvement_priorities, start=1)],
        "",
        "Automatic answer/citation/faithfulness metrics require human confirmation.",
    ]
    (run_dir / "evaluation_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "retrieval_metrics": retrieval_metrics,
        "answer_metrics": answer_metrics,
        "citation_metrics": citation_metrics,
        "faithfulness_metrics": faithfulness_metrics,
        "abstention_metrics": abstention_metrics,
        "performance_metrics": performance,
        "cost_metrics": cost,
    }


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    dataset: list[dict[str, Any]]
    no_answer_cases: list[dict[str, Any]]
    mapping_rows: list[dict[str, Any]]
    mapping_summary: dict[str, Any]
    preflight: dict[str, Any]
    config_fingerprint: str
    answer_rows: list[dict[str, Any]] = field(default_factory=list)
    abstention_rows: list[dict[str, Any]] = field(default_factory=list)
    api_records: list[dict[str, Any]] = field(default_factory=list)

    def checkpoint_payload(self, status: str, phase: str) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_status": status,
            "phase": phase,
            "dataset_sha256": self.preflight["dataset"]["sha256"],
            "no_answer_sha256": self.preflight["no_answer"]["sha256"],
            "index_fingerprint": self.preflight["index"]["fingerprint"],
            "config_fingerprint": self.config_fingerprint,
            "completed_answer_ids": [
                row["question_id"] for row in self.answer_rows
            ],
            "remaining_answer_ids": [
                row["question_id"]
                for row in self.dataset
                if row["question_id"]
                not in {item["question_id"] for item in self.answer_rows}
            ],
            "completed_no_answer_ids": [
                row["question_id"] for row in self.abstention_rows
            ],
            "remaining_no_answer_ids": [
                row["question_id"]
                for row in self.no_answer_cases
                if row["question_id"]
                not in {item["question_id"] for item in self.abstention_rows}
            ],
            "api_call_count": len(self.api_records),
            "updated_at": utc_now(),
        }

    def persist(self, status: str, phase: str) -> None:
        checkpoint = self.checkpoint_payload(status, phase)
        atomic_write_json(self.run_dir / "RESUME_CHECKPOINT.json", checkpoint)
        atomic_write_json(
            self.run_dir / "RUN_STATUS.json",
            {**checkpoint, "status_reason": None},
        )


def preflight_all(
    *,
    chroma_dir: Path,
    docstore_dir: Path,
    manifest_path: Path,
    expected_fingerprint: str,
    no_answer_path: Path | None,
) -> tuple[
    CleanIndexSnapshot,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    dataset = read_jsonl(DATASET_PATH)
    dataset_summary = validate_evaluation_dataset(dataset)
    no_answer_cases, no_answer_source = resolve_no_answer_cases(no_answer_path)
    no_answer_summary = validate_no_answer_cases(no_answer_cases)
    no_answer_summary["source"] = no_answer_source
    snapshot = load_and_verify_index(
        chroma_dir,
        docstore_dir,
        manifest_path,
        expected_fingerprint,
    )
    no_answer_summary["absence_verification"] = (
        verify_no_answer_absence_in_clean_index(no_answer_cases, snapshot)
    )
    mapping_rows, mapping_summary = build_expected_chunk_mapping(
        dataset,
        NODE_DIFF_PATH,
        snapshot,
    )
    config = AppSettings()  # type: ignore[call-arg]
    if config.rerank_top_n != 4:
        raise EvaluationBlocked(
            f"Production rerank_top_n must be 4, got {config.rerank_top_n}"
        )
    preflight = {
        "index": {
            "chroma_dir": str(snapshot.chroma_dir),
            "docstore_dir": str(snapshot.docstore_dir),
            "fingerprint": snapshot.fingerprint,
            "counts": snapshot.counts,
        },
        "dataset": dataset_summary,
        "no_answer": no_answer_summary,
        "mapping": mapping_summary,
        "models": {
            "embedding_provider": config.embedding_provider,
            "embedding_model": config.embedding_model,
            "rerank_model": config.rerank_model,
            "llm_model": config.llm_model,
        },
        "retrieval": {
            "vector_top_k": config.vector_top_k,
            "bm25_top_k": config.bm25_top_k,
            "rerank_top_n": config.rerank_top_n,
            "final_top_k": config.rerank_top_n,
        },
    }
    return snapshot, dataset, no_answer_cases, mapping_rows, preflight


def config_fingerprint(preflight: dict[str, Any]) -> str:
    return stable_hash(
        {
            "dataset_sha256": preflight["dataset"]["sha256"],
            "no_answer_sha256": preflight["no_answer"]["sha256"],
            "index_fingerprint": preflight["index"]["fingerprint"],
            "models": preflight["models"],
            "retrieval": preflight["retrieval"],
            "runner_version": 1,
        }
    )


def load_resume_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    ids = [row.get("question_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise EvaluationBlocked(f"Duplicate question IDs in resume file: {path}")
    return rows


def validate_resume(
    run_dir: Path,
    *,
    expected_config_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_path = run_dir / "RESUME_CHECKPOINT.json"
    if not checkpoint_path.exists():
        raise EvaluationBlocked(f"Missing checkpoint: {checkpoint_path}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("config_fingerprint") != expected_config_fingerprint:
        raise EvaluationBlocked(
            "Resume blocked: dataset, index, model or retrieval configuration changed."
        )
    answer_rows = load_resume_rows(run_dir / "answer_results.jsonl")
    abstention_rows = load_resume_rows(run_dir / "abstention_results.jsonl")
    answer_ids = {row["question_id"] for row in answer_rows}
    no_answer_ids = {row["question_id"] for row in abstention_rows}
    checkpoint_answers = set(checkpoint.get("completed_answer_ids") or [])
    checkpoint_no_answers = set(
        checkpoint.get("completed_no_answer_ids") or []
    )
    if answer_ids != checkpoint_answers:
        raise EvaluationBlocked(
            "Resume blocked: completed answer IDs do not match checkpoint."
        )
    if no_answer_ids != checkpoint_no_answers:
        raise EvaluationBlocked(
            "Resume blocked: completed no-answer IDs do not match checkpoint."
        )
    return answer_rows, abstention_rows


def write_preflight_outputs(
    run_dir: Path,
    mapping_rows: list[dict[str, Any]],
    mapping_summary: dict[str, Any],
    no_answer_cases: list[dict[str, Any]],
    preflight: dict[str, Any],
    run_id: str,
) -> None:
    initialize_output_contract(run_dir, run_id)
    write_csv(run_dir / "expected_chunk_mapping.csv", mapping_rows)
    atomic_write_json(
        run_dir / "expected_chunk_mapping_summary.json",
        mapping_summary,
    )
    append_jsonl(run_dir / "no_answer_eval_10.jsonl", no_answer_cases)
    atomic_write_json(run_dir / "run_manifest.json", {
        "run_id": run_id,
        "created_at": utc_now(),
        "preflight": preflight,
        "mapping_summary": mapping_summary,
        "runner": __file__,
    })


def run_evaluation(
    *,
    chroma_dir: Path,
    docstore_dir: Path,
    manifest_path: Path,
    expected_fingerprint: str,
    output_dir: Path | None,
    no_answer_path: Path | None,
    resume: bool,
    failure_threshold: int,
    warmup: bool,
) -> Path:
    snapshot, dataset, no_answer_cases, mapping_rows, preflight = preflight_all(
        chroma_dir=chroma_dir,
        docstore_dir=docstore_dir,
        manifest_path=manifest_path,
        expected_fingerprint=expected_fingerprint,
        no_answer_path=no_answer_path,
    )
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        run_dir = RESULTS_ROOT / f"full_evaluation_clean_index_{stamp}"
        if run_dir.exists():
            run_dir = RESULTS_ROOT / f"full_evaluation_clean_index_{stamp}_{uuid.uuid4().hex[:6]}"
    else:
        run_dir = output_dir.resolve()
    run_id = run_dir.name
    current_config_fingerprint = config_fingerprint(preflight)
    if resume:
        if not run_dir.exists():
            raise EvaluationBlocked(f"Resume directory does not exist: {run_dir}")
        answer_rows, abstention_rows = validate_resume(
            run_dir,
            expected_config_fingerprint=current_config_fingerprint,
        )
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_preflight_outputs(
            run_dir,
            mapping_rows,
            preflight["mapping"],
            no_answer_cases,
            preflight,
            run_id,
        )
        answer_rows = []
        abstention_rows = []

    context = RunContext(
        run_dir=run_dir,
        run_id=run_id,
        dataset=dataset,
        no_answer_cases=no_answer_cases,
        mapping_rows=mapping_rows,
        mapping_summary=preflight["mapping"],
        preflight=preflight,
        config_fingerprint=current_config_fingerprint,
        answer_rows=answer_rows,
        abstention_rows=abstention_rows,
        api_records=(
            read_jsonl(run_dir / "api_usage.jsonl")
            if (run_dir / "api_usage.jsonl").exists()
            else []
        ),
    )
    recorder = APICallRecorder(
        run_id,
        sink_path=run_dir / "api_usage.jsonl",
    )
    recorder.seed_existing(context.api_records)
    runtime = CleanRetrievalRuntime.build(snapshot, recorder)
    llm = _configured_llm(runtime, recorder)
    tracker = FailureTracker(failure_threshold)
    tracker.seed(answer_rows + abstention_rows)
    completed_answer_ids = {row["question_id"] for row in answer_rows}
    completed_no_answer_ids = {
        row["question_id"] for row in abstention_rows
    }

    context.persist("running", "answer_50")
    if warmup and not (
        run_dir / "answer_warmup.jsonl"
    ).exists():
        warmup_question = dataset[0]
        runtime.snapshot = load_and_verify_index(
            chroma_dir,
            docstore_dir,
            manifest_path,
            expected_fingerprint,
        )
        warmup_row = execute_question(
            runtime=runtime,
            llm=llm,
            question={**warmup_question, "question_id": f"WARMUP-{warmup_question['question_id']}"},
            mapping_rows=mapping_rows,
            scope="warmup",
        )
        append_jsonl(run_dir / "answer_warmup.jsonl", [warmup_row])

    try:
        for question in dataset:
            if question["question_id"] in completed_answer_ids:
                continue
            load_and_verify_index(
                chroma_dir,
                docstore_dir,
                manifest_path,
                expected_fingerprint,
            )
            row = execute_question(
                runtime=runtime,
                llm=llm,
                question=question,
                mapping_rows=mapping_rows,
                scope="answer",
            )
            context.answer_rows.append(row)
            upsert_jsonl_by_question_id(
                run_dir / "answer_results.jsonl",
                row,
            )
            context.api_records = list(recorder.records)
            context.persist("running", "answer_50")
            tracker.record(bool(row.get("had_api_failure")))
            if tracker.should_pause():
                raise ConsecutiveFailureLimit(
                    f"{tracker.consecutive} consecutive questions had API failures"
                )

        context.persist("running", "no_answer_10")
        for case in no_answer_cases:
            if case["question_id"] in completed_no_answer_ids:
                continue
            load_and_verify_index(
                chroma_dir,
                docstore_dir,
                manifest_path,
                expected_fingerprint,
            )
            row = execute_question(
                runtime=runtime,
                llm=llm,
                question=case,
                mapping_rows=mapping_rows,
                scope="no_answer",
            )
            row["classification"] = classify_abstention(row)
            row["correct_refusal"] = row["classification"] == "correct_refusal"
            context.abstention_rows.append(row)
            upsert_jsonl_by_question_id(
                run_dir / "abstention_results.jsonl",
                row,
            )
            context.api_records = list(recorder.records)
            context.persist("running", "no_answer_10")
            tracker.record(bool(row.get("had_api_failure")))
            if tracker.should_pause():
                raise ConsecutiveFailureLimit(
                    f"{tracker.consecutive} consecutive questions had API failures"
                )

        final_fingerprint = verify_index_still_frozen(
            chroma_dir,
            docstore_dir,
            expected_fingerprint,
        )
        if len(context.answer_rows) != 50:
            raise EvaluationBlocked(
                f"Incomplete answer set: {len(context.answer_rows)}/50"
            )
        if len(context.abstention_rows) != 10:
            raise EvaluationBlocked(
                f"Incomplete no-answer set: {len(context.abstention_rows)}/10"
            )
        write_evaluation_outputs(
            run_dir=run_dir,
            run_id=run_id,
            answer_rows=context.answer_rows,
            abstention_rows=context.abstention_rows,
            api_records=recorder.records,
            mapping_rows=mapping_rows,
            dataset=dataset,
            mapping_summary=preflight["mapping"],
            preflight=preflight,
        )
        final_manifest = {
            "run_id": run_id,
            "created_at": utc_now(),
            "status": "completed",
            "index_fingerprint_before": expected_fingerprint,
            "index_fingerprint_after": final_fingerprint,
            "preflight": preflight,
            "answer_count": len(context.answer_rows),
            "no_answer_count": len(context.abstention_rows),
            "api_call_count": len(recorder.records),
            "automatic_quality_metrics_require_manual_review": True,
        }
        atomic_write_json(run_dir / "run_manifest.json", final_manifest)
        context.persist("completed", "completed")
        return run_dir
    except IndexFingerprintChanged as exc:
        context.persist("paused_index_fingerprint_changed", "paused")
        atomic_write_json(
            run_dir / "RUN_STATUS.json",
            {
                **context.checkpoint_payload(
                    "paused_index_fingerprint_changed", "paused"
                ),
                "status_reason": str(exc),
            },
        )
        raise
    except ConsecutiveFailureLimit as exc:
        context.persist("paused_consecutive_api_failures", "paused")
        atomic_write_json(
            run_dir / "RUN_STATUS.json",
            {
                **context.checkpoint_payload(
                    "paused_consecutive_api_failures", "paused"
                ),
                "status_reason": str(exc),
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-dir", type=Path, required=True)
    parser.add_argument("--docstore-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
    )
    parser.add_argument(
        "--expected-fingerprint",
        default=DEFAULT_EXPECTED_FINGERPRINT,
    )
    parser.add_argument("--no-answer-dataset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--max-consecutive-api-failures",
        type=int,
        default=3,
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    snapshot, dataset, no_answer_cases, mapping_rows, preflight = preflight_all(
        chroma_dir=args.chroma_dir,
        docstore_dir=args.docstore_dir,
        manifest_path=args.manifest,
        expected_fingerprint=args.expected_fingerprint,
        no_answer_path=args.no_answer_dataset,
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "index": preflight["index"],
                    "dataset": preflight["dataset"],
                    "no_answer": preflight["no_answer"],
                    "mapping": preflight["mapping"],
                    "retrieval": preflight["retrieval"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    run_dir = run_evaluation(
        chroma_dir=args.chroma_dir,
        docstore_dir=args.docstore_dir,
        manifest_path=args.manifest,
        expected_fingerprint=args.expected_fingerprint,
        output_dir=args.output_dir,
        no_answer_path=args.no_answer_dataset,
        resume=args.resume,
        failure_threshold=args.max_consecutive_api_failures,
        warmup=args.warmup,
    )
    print(f"RESULTS_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

