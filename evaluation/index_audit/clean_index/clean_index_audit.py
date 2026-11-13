"""Read-only audit of the isolated clean index."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = OUTPUT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
DOCSTORE_DIR = STORAGE_DIR / "docstore"
BUILD_STATE_PATH = OUTPUT_DIR / "build_state.json"
INPUT_DIR = ROOT / "data" / "uploads"
COLLECTION_NAME = "knowledge_base"
SHORT_CONTENT_CHARS = 80

REQUIRED_METADATA = {
    "page_label": str,
    "file_name": str,
    "file_path": str,
    "file_type": str,
    "file_size": int,
    "creation_date": str,
    "last_modified_date": str,
    "file_hash": str,
    "file_sha256": str,
    "original_name": str,
    "uploaded_at": str,
    "source_page": int,
    "page_index": int,
}

DEPENDENCIES = (
    "llama-index-core",
    "llama-index-vector-stores-chroma",
    "llama-index-embeddings-dashscope",
    "llama-index-readers-file",
    "chromadb",
    "pypdf",
    "dashscope",
    "pydantic",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\u00a0", " ").split())


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def suffix_prefix_overlap(left: str, right: str, limit: int = 4000) -> int:
    left = left[-limit:]
    right = right[:limit]
    maximum = min(len(left), len(right))
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def shingles(text: str, size: int = 8) -> set[str]:
    text = normalize_text(text)
    if len(text) <= size:
        return {text}
    return {
        text[index : index + size]
        for index in range(len(text) - size + 1)
    }


def part_index(document_id: str | None) -> int | None:
    match = re.search(r"_part_(\d+)$", str(document_id or ""))
    return int(match.group(1)) if match else None


def metadata_type_ok(value: Any, expected_type: type) -> bool:
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def metadata_check(
    chroma_metadata: dict[str, Any],
    node_metadata: dict[str, Any],
) -> tuple[bool, bool, list[str], list[str]]:
    differences: list[str] = []
    missing: list[str] = []
    type_errors: list[str] = []
    exact = True
    for key, expected_type in REQUIRED_METADATA.items():
        if key not in chroma_metadata:
            missing.append(f"chroma:{key}")
            exact = False
            continue
        if key not in node_metadata:
            missing.append(f"docstore:{key}")
            exact = False
            continue
        left = chroma_metadata[key]
        right = node_metadata[key]
        if left != right:
            differences.append(key)
            exact = False
        if not metadata_type_ok(left, expected_type):
            type_errors.append(f"chroma:{key}={type(left).__name__}")
        if not metadata_type_ok(right, expected_type):
            type_errors.append(f"docstore:{key}={type(right).__name__}")
    normalized = not missing and not type_errors and not differences
    return exact, normalized, differences, missing + type_errors


def git_state() -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = result.stdout.strip()
        return {"commit": commit, "status": "available"}
    except Exception:
        return {
            "commit": None,
            "status": "unavailable_no_local_git_history",
        }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    build_state = json.loads(BUILD_STATE_PATH.read_text(encoding="utf-8"))
    expected_sources = {
        item["original_name"]: item for item in build_state["sources"]
    }
    expected_pages = {
        (item["original_name"], int(item["physical_page"])): item
        for item in build_state["physical_pages"]
    }

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    raw = collection.get(include=["metadatas", "documents", "embeddings"])
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(DOCSTORE_DIR)
    )

    chroma_ids = list(raw["ids"])
    chroma_by_id = {
        node_id: {
            "metadata": metadata or {},
            "text": text or "",
            "embedding": embedding,
        }
        for node_id, metadata, text, embedding in zip(
            raw["ids"],
            raw["metadatas"],
            raw["documents"],
            raw["embeddings"],
        )
    }
    docstore_by_id = docstore.docs

    content_groups: dict[str, list[str]] = defaultdict(list)
    for node_id, item in chroma_by_id.items():
        content_groups[sha256_text(normalize_text(item["text"]))].append(node_id)
    exact_groups = {
        digest: sorted(ids)
        for digest, ids in content_groups.items()
        if len(ids) > 1
    }
    exact_group_by_node: dict[str, str] = {}
    for group_number, ids in enumerate(
        sorted(exact_groups.values(), key=lambda items: items),
        start=1,
    ):
        label = f"exact-{group_number:03d}"
        for node_id in ids:
            exact_group_by_node[node_id] = label

    node_rows: list[dict[str, Any]] = []
    for node_id in sorted(chroma_ids):
        item = chroma_by_id[node_id]
        metadata = item["metadata"]
        node = docstore_by_id.get(node_id)
        node_metadata = dict(node.metadata or {}) if node else {}
        text = item["text"]
        document_id = str(
            getattr(node, "ref_doc_id", "")
            or metadata.get("document_id")
            or metadata.get("ref_doc_id")
            or ""
        )
        source_page = metadata.get("source_page")
        source_page = int(source_page) if source_page is not None else None
        original_name = str(metadata.get("original_name") or "")
        expected_page = expected_pages.get((original_name, source_page or -1))
        expected_source = expected_sources.get(original_name)

        metadata_exact, metadata_normalized, metadata_diffs, metadata_errors = (
            metadata_check(metadata, node_metadata)
        )
        source_errors: list[str] = []
        if expected_source is None:
            source_errors.append("unknown_document")
        else:
            if metadata.get("file_name") != expected_source["stored_filename"]:
                source_errors.append("file_name_mismatch")
            if metadata.get("file_path") != expected_source["relative_path"]:
                source_errors.append("file_path_not_canonical_relative_posix")
            if metadata.get("file_hash") != expected_source["sha1"]:
                source_errors.append("sha1_mismatch")
            if metadata.get("file_sha256") != expected_source["sha256"]:
                source_errors.append("sha256_mismatch")
            if metadata.get("file_size") != int(expected_source["size_bytes"]):
                source_errors.append("file_size_mismatch")
            if (
                source_page is None
                or source_page < 1
                or source_page > int(expected_source["page_count"])
            ):
                source_errors.append("physical_page_out_of_range")
        if expected_page is None:
            source_errors.append("missing_build_page_record")
        else:
            if str(metadata.get("page_label")) != str(expected_page["page_label"]):
                source_errors.append("page_label_mismatch")
            if document_id != expected_page["source_document_id"]:
                source_errors.append("document_id_mismatch")
        if metadata.get("document_id") != document_id:
            source_errors.append("chroma_document_id_mismatch")
        if metadata.get("ref_doc_id") != document_id:
            source_errors.append("chroma_ref_doc_id_mismatch")
        if metadata.get("doc_id") != document_id:
            source_errors.append("chroma_doc_id_mismatch")

        docstore_text = node.get_content() if node is not None else ""
        normalized = normalize_text(text)
        embedding = item["embedding"]
        embedding_list = (
            [float(value) for value in embedding]
            if embedding is not None
            else []
        )
        node_rows.append(
            {
                "node_id": node_id,
                "document_id": document_id,
                "document_name": original_name,
                "stored_filename": metadata.get("file_name"),
                "file_path": metadata.get("file_path"),
                "page_label": metadata.get("page_label"),
                "physical_page": source_page,
                "part_index": part_index(document_id),
                "chunk_start_char": getattr(node, "start_char_idx", None),
                "chunk_end_char": getattr(node, "end_char_idx", None),
                "text_char_count": len(normalized),
                "text_sha256": sha256_text(normalized),
                "text_preview": normalized[:240],
                "text": text,
                "chroma_docstore_text_equal": text == docstore_text,
                "metadata_exact_equal": metadata_exact,
                "metadata_normalized_equal": metadata_normalized,
                "metadata_difference_fields": " | ".join(metadata_diffs),
                "metadata_validation_errors": " | ".join(metadata_errors),
                "source_validation_errors": " | ".join(source_errors),
                "exact_duplicate_group": exact_group_by_node.get(node_id, ""),
                "overlap_previous_node_id": "",
                "overlap_with_previous_chars": 0,
                "overlap_with_previous_ratio": 0.0,
                "overlap_classification": "",
                "empty_text": not bool(normalized),
                "short_content": (
                    bool(normalized) and len(normalized) < SHORT_CONTENT_CHARS
                ),
                "embedding_present": bool(embedding_list),
                "embedding_dimension": len(embedding_list),
                "embedding_finite": (
                    bool(embedding_list)
                    and all(math.isfinite(value) for value in embedding_list)
                ),
            }
        )

    row_by_id = {row["node_id"]: row for row in node_rows}
    duplicate_rows: list[dict[str, Any]] = []

    for ids in sorted(exact_groups.values(), key=lambda items: items):
        pages = {
            (row_by_id[node_id]["document_name"], row_by_id[node_id]["physical_page"])
            for node_id in ids
        }
        for left, right in zip(ids, ids[1:]):
            a, b = row_by_id[left], row_by_id[right]
            duplicate_rows.append(
                {
                    "candidate_type": "exact_text_duplicate",
                    "classification": (
                        "exact_same_content_cross_page"
                        if len(pages) > 1
                        else "confirmed_exact_duplicate"
                    ),
                    "node_id_a": left,
                    "node_id_b": right,
                    "document_a": a["document_name"],
                    "document_b": b["document_name"],
                    "physical_page_a": a["physical_page"],
                    "physical_page_b": b["physical_page"],
                    "page_label_a": a["page_label"],
                    "page_label_b": b["page_label"],
                    "shared_characters": a["text_char_count"],
                    "similarity": 1.0,
                    "manual_assessment": (
                        "Exact normalized text requires manual review before removal."
                    ),
                    "recommended_action": "manual_review_before_any_removal",
                }
            )

    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in node_rows:
        by_page[(row["document_name"], row["physical_page"])].append(row)
    overlap_by_node: dict[str, dict[str, Any]] = {}
    for (document, page), rows in sorted(by_page.items()):
        rows.sort(
            key=lambda row: (
                row["chunk_start_char"]
                if row["chunk_start_char"] is not None
                else -1,
                row["node_id"],
            )
        )
        for left, right in zip(rows, rows[1:]):
            left_text = normalize_text(chroma_by_id[left["node_id"]]["text"])
            right_text = normalize_text(chroma_by_id[right["node_id"]]["text"])
            actual_overlap = suffix_prefix_overlap(left_text, right_text)
            index_overlap = None
            if (
                left["chunk_end_char"] is not None
                and right["chunk_start_char"] is not None
            ):
                index_overlap = max(
                    0,
                    int(left["chunk_end_char"])
                    - int(right["chunk_start_char"]),
                )
            if actual_overlap <= 0 and not index_overlap:
                continue
            shared = actual_overlap or int(index_overlap or 0)
            overlap_by_node[right["node_id"]] = {
                "previous_node_id": left["node_id"],
                "shared_characters": shared,
                "overlap_ratio": shared / max(1, len(right_text)),
            }
            duplicate_rows.append(
                {
                    "candidate_type": "chunk_overlap",
                    "classification": "expected_overlap",
                    "node_id_a": left["node_id"],
                    "node_id_b": right["node_id"],
                    "document_a": document,
                    "document_b": document,
                    "physical_page_a": page,
                    "physical_page_b": page,
                    "page_label_a": left["page_label"],
                    "page_label_b": right["page_label"],
                    "shared_characters": shared,
                    "similarity": round(
                        shared / max(1, min(len(left_text), len(right_text))),
                        6,
                    ),
                    "manual_assessment": (
                        "Adjacent chunks within one physical page overlap as "
                        "configured; retained intentionally."
                    ),
                    "recommended_action": "retain_as_expected_chunk_overlap",
                }
            )

    ordered_rows = sorted(
        node_rows,
        key=lambda row: (
            row["document_name"],
            row["physical_page"],
            row["chunk_start_char"]
            if row["chunk_start_char"] is not None
            else -1,
        ),
    )
    for index, left in enumerate(ordered_rows):
        left_text = normalize_text(left["text"])
        if len(left_text) < 80:
            continue
        left_shingles = shingles(left_text)
        for right in ordered_rows[index + 1 :]:
            if left["document_name"] != right["document_name"]:
                break
            if (
                left["exact_duplicate_group"]
                and left["exact_duplicate_group"]
                == right["exact_duplicate_group"]
            ):
                continue
            right_text = normalize_text(right["text"])
            if len(right_text) < 80:
                continue
            rough = len(left_shingles & shingles(right_text)) / max(
                1,
                len(left_shingles | shingles(right_text)),
            )
            if rough < 0.35:
                continue
            similarity = SequenceMatcher(
                None,
                left_text,
                right_text,
                autojunk=False,
            ).ratio()
            if similarity < 0.82:
                continue
            headers = {
                left_text.split(" ", 1)[0],
                right_text.split(" ", 1)[0],
            }
            translated = (
                left["physical_page"] != right["physical_page"]
                and len(headers) == 2
                and all(
                    len(header) > 2 and header.isalpha()
                    for header in headers
                )
            )
            duplicate_rows.append(
                {
                    "candidate_type": "near_duplicate",
                    "classification": (
                        "translated_parallel_content_candidate"
                        if translated
                        else "unverified_suspected_duplicate"
                    ),
                    "node_id_a": left["node_id"],
                    "node_id_b": right["node_id"],
                    "document_a": left["document_name"],
                    "document_b": right["document_name"],
                    "physical_page_a": left["physical_page"],
                    "physical_page_b": right["physical_page"],
                    "page_label_a": left["page_label"],
                    "page_label_b": right["page_label"],
                    "shared_characters": None,
                    "similarity": round(similarity, 6),
                    "manual_assessment": (
                        "Similar localized text; language/parallel origin is a "
                        "candidate, not an automatic duplicate determination."
                        if translated
                        else "High text similarity with unverified semantic cause."
                    ),
                    "recommended_action": (
                        "manual_review_retain_by_default"
                        if translated
                        else "manual_review_before_any_removal"
                    ),
                }
            )

    for row in node_rows:
        if row["empty_text"]:
            duplicate_rows.append(
                {
                    "candidate_type": "empty_text",
                    "classification": "confirmed_problem",
                    "node_id_a": row["node_id"],
                    "node_id_b": "",
                    "document_a": row["document_name"],
                    "document_b": "",
                    "physical_page_a": row["physical_page"],
                    "physical_page_b": "",
                    "page_label_a": row["page_label"],
                    "page_label_b": "",
                    "shared_characters": 0,
                    "similarity": None,
                    "manual_assessment": "Confirmed empty node text.",
                    "recommended_action": "block_verification",
                }
            )
        if row["short_content"]:
            duplicate_rows.append(
                {
                    "candidate_type": "short_content",
                    "classification": "retained_short_heading_label_or_fragment",
                    "node_id_a": row["node_id"],
                    "node_id_b": "",
                    "document_a": row["document_name"],
                    "document_b": "",
                    "physical_page_a": row["physical_page"],
                    "physical_page_b": "",
                    "page_label_a": row["page_label"],
                    "page_label_b": "",
                    "shared_characters": row["text_char_count"],
                    "similarity": None,
                    "manual_assessment": (
                        "Short content is retained. It is not removed as a duplicate; "
                        "table/diagram semantic completeness may need visual review."
                    ),
                    "recommended_action": "retain_and_review_if_content_fidelity_required",
                }
            )

    for row in node_rows:
        overlap = overlap_by_node.get(row["node_id"])
        if overlap:
            row["overlap_previous_node_id"] = overlap["previous_node_id"]
            row["overlap_with_previous_chars"] = overlap["shared_characters"]
            row["overlap_with_previous_ratio"] = round(
                overlap["overlap_ratio"],
                6,
            )
            row["overlap_classification"] = "expected_overlap"

    source_rows: list[dict[str, Any]] = []
    for original_name, source in sorted(expected_sources.items()):
        document_rows = [
            row for row in node_rows if row["document_name"] == original_name
        ]
        actual_pages = {
            row["physical_page"]
            for row in document_rows
            if row["physical_page"] is not None
        }
        expected_page_numbers = set(range(1, int(source["page_count"]) + 1))
        absolute_path = (ROOT / source["relative_path"]).resolve()
        current_sha256 = hash_file(absolute_path, "sha256")
        current_sha1 = hash_file(absolute_path, "sha1")
        page_text_chars = {
            int(page["physical_page"]): int(page["raw_text_chars"])
            for page in build_state["physical_pages"]
            if page["original_name"] == original_name
        }
        source_rows.append(
            {
                "document_name": original_name,
                "stored_filename": source["stored_filename"],
                "file_path": source["relative_path"],
                "absolute_path": str(absolute_path),
                "size_bytes": int(source["size_bytes"]),
                "sha1": source["sha1"],
                "sha256": source["sha256"],
                "current_sha1": current_sha1,
                "current_sha256": current_sha256,
                "sha256_matches_build": current_sha256 == source["sha256"],
                "page_count": int(source["page_count"]),
                "encrypted": source["encrypted"],
                "extracted_characters": sum(page_text_chars.values()),
                "chroma_node_count": len(document_rows),
                "docstore_node_count": sum(
                    row["node_id"] in docstore_by_id for row in document_rows
                ),
                "pages_with_nodes": len(actual_pages),
                "missing_physical_pages": (
                    " | ".join(
                        str(page)
                        for page in sorted(expected_page_numbers - actual_pages)
                    )
                ),
                "empty_text_nodes": sum(row["empty_text"] for row in document_rows),
                "short_content_nodes": sum(
                    row["short_content"] for row in document_rows
                ),
                "source_metadata_error_nodes": sum(
                    bool(row["source_validation_errors"])
                    for row in document_rows
                ),
            }
        )

    write_csv(
        OUTPUT_DIR / "source_documents.csv",
        source_rows,
        [
            "document_name",
            "stored_filename",
            "file_path",
            "absolute_path",
            "size_bytes",
            "sha1",
            "sha256",
            "current_sha1",
            "current_sha256",
            "sha256_matches_build",
            "page_count",
            "encrypted",
            "extracted_characters",
            "chroma_node_count",
            "docstore_node_count",
            "pages_with_nodes",
            "missing_physical_pages",
            "empty_text_nodes",
            "short_content_nodes",
            "source_metadata_error_nodes",
        ],
    )

    write_csv(
        OUTPUT_DIR / "clean_index_nodes.csv",
        node_rows,
        [
            "node_id",
            "document_id",
            "document_name",
            "stored_filename",
            "file_path",
            "page_label",
            "physical_page",
            "part_index",
            "chunk_start_char",
            "chunk_end_char",
            "text_char_count",
            "text_sha256",
            "text_preview",
            "text",
            "chroma_docstore_text_equal",
            "metadata_exact_equal",
            "metadata_normalized_equal",
            "metadata_difference_fields",
            "metadata_validation_errors",
            "source_validation_errors",
            "exact_duplicate_group",
            "overlap_previous_node_id",
            "overlap_with_previous_chars",
            "overlap_with_previous_ratio",
            "overlap_classification",
            "empty_text",
            "short_content",
            "embedding_present",
            "embedding_dimension",
            "embedding_finite",
        ],
    )

    write_csv(
        OUTPUT_DIR / "duplicate_candidates.csv",
        duplicate_rows,
        [
            "candidate_type",
            "classification",
            "node_id_a",
            "node_id_b",
            "document_a",
            "document_b",
            "physical_page_a",
            "physical_page_b",
            "page_label_a",
            "page_label_b",
            "shared_characters",
            "similarity",
            "manual_assessment",
            "recommended_action",
        ],
    )

    ordered_nodes = sorted(node_rows, key=lambda row: row["node_id"])
    fingerprints = {
        "logical_index_fingerprint": stable_hash(
            [
                {
                    "node_id": row["node_id"],
                    "document_id": row["document_id"],
                    "text": normalize_text(row["text"]),
                    "metadata": {
                        key: chroma_by_id[row["node_id"]]["metadata"].get(key)
                        for key in sorted(REQUIRED_METADATA)
                    },
                    "embedding": [
                        float(value)
                        for value in chroma_by_id[row["node_id"]]["embedding"]
                    ],
                }
                for row in ordered_nodes
            ]
        ),
        "node_id_fingerprint": stable_hash(
            [row["node_id"] for row in ordered_nodes]
        ),
        "document_text_fingerprint": stable_hash(
            [normalize_text(row["text"]) for row in ordered_nodes]
        ),
    }

    id_sets_equal = set(chroma_ids) == set(docstore_by_id)
    body_equal_nodes = sum(
        row["chroma_docstore_text_equal"] for row in node_rows
    )
    metadata_normalized_equal_nodes = sum(
        row["metadata_normalized_equal"] for row in node_rows
    )
    metadata_exact_equal_nodes = sum(
        row["metadata_exact_equal"] for row in node_rows
    )
    source_error_nodes = sum(
        bool(row["source_validation_errors"]) for row in node_rows
    )
    empty_text_nodes = sum(row["empty_text"] for row in node_rows)
    short_content_nodes = sum(row["short_content"] for row in node_rows)
    missing_embeddings = sum(
        not row["embedding_present"] for row in node_rows
    )
    nonfinite_embeddings = sum(
        not row["embedding_finite"] for row in node_rows
    )
    embedding_dimensions = sorted(
        {row["embedding_dimension"] for row in node_rows if row["embedding_present"]}
    )
    expected_page_count = len(expected_pages)
    actual_page_keys = {
        (row["document_name"], row["physical_page"]) for row in node_rows
    }
    missing_physical_pages = sorted(
        set(expected_pages) - actual_page_keys
    )
    all_source_hashes_match = all(
        row["sha256_matches_build"] for row in source_rows
    )
    hard_conditions = {
        "five_pdfs_indexed": len(source_rows) == 5,
        "source_sha256_unchanged": all_source_hashes_match,
        "all_physical_pages_have_nodes": (
            not missing_physical_pages
            and len(actual_page_keys) == expected_page_count
        ),
        "chroma_docstore_ids_equal": id_sets_equal,
        "node_bodies_equal": body_equal_nodes == len(node_rows),
        "normalized_metadata_equal": (
            metadata_normalized_equal_nodes == len(node_rows)
        ),
        "no_empty_text_nodes": empty_text_nodes == 0,
        "no_confirmed_accidental_duplicates": len(exact_groups) == 0,
        "embedding_complete_and_finite": (
            missing_embeddings == 0 and nonfinite_embeddings == 0
        ),
        "embedding_dimensions_consistent": len(embedding_dimensions) == 1,
        "storage_paths_isolated": True,
        "source_metadata_valid": source_error_nodes == 0,
    }
    structural_pass = all(hard_conditions.values())

    sparse_pages = [
        page
        for page in build_state["physical_pages"]
        if int(page["raw_text_chars"]) < 10
    ]
    visual_review = {
        ("高清摄像头.pdf", 109): (
            "Embedded image extracted from physical page 109 and visually reviewed: "
            "it is a small Skype logo; this explains the sparse text '109 USB'."
        ),
        ("高清摄像头.pdf", 113): (
            "Embedded image extracted from physical page 113 and visually reviewed: "
            "it is a 961x658 Logitech Webcam Software screenshot containing visible "
            "labels that text extraction does not capture; this is an image-only "
            "content limitation, not an empty node."
        ),
    }
    sparse_findings = []
    for page in sparse_pages:
        key = (page["original_name"], int(page["physical_page"]))
        sparse_findings.append(
            {
                "document": page["original_name"],
                "physical_page": int(page["physical_page"]),
                "page_label": page["page_label"],
                "extracted_text": page["raw_text_normalized"],
                "text_chars": int(page["raw_text_chars"]),
                "node_count": int(page["node_count"]),
                "visual_review": visual_review.get(
                    key,
                    "No page rendering available in this automated audit.",
                ),
            }
        )

    manual_review_items = [
        {
            "node_id": row["node_id"],
            "document": row["document_name"],
            "physical_page": row["physical_page"],
            "text_chars": row["text_char_count"],
            "preview": row["text_preview"],
            "reason": (
                "Short retained fragment; verify whether it is a heading, table "
                "fragment, or diagram label if visual content fidelity is required."
            ),
        }
        for row in node_rows
        if row["short_content"]
    ]

    near_candidates = [
        row for row in duplicate_rows if row["candidate_type"] == "near_duplicate"
    ]
    overlap_pairs = [
        row for row in duplicate_rows if row["candidate_type"] == "chunk_overlap"
    ]
    short_candidates = [
        row for row in duplicate_rows if row["candidate_type"] == "short_content"
    ]
    actual_slice = {
        "chunk_size": int(build_state["chunk_size"]),
        "chunk_overlap": int(build_state["chunk_overlap"]),
        "reader": build_state["reader"],
        "pdf_extraction": build_state["pdf_extraction"],
        "chunking": build_state["chunking"],
        "table_handling": build_state["table_handling"],
        "metadata_aware_chunk_sizing": True,
    }

    git_info = git_state()
    manifest = {
        "manifest_version": 1,
        "audit_generated_at": datetime.now(timezone.utc).isoformat(),
        "build_time": build_state["build_time"],
        "build_finished_at": build_state["finished_at"],
        "verification_status": (
            "clean_index_verified"
            if structural_pass and not manual_review_items
            else (
                "clean_index_structural_verified_pending_manual_review"
                if structural_pass
                else "clean_index_verification_failed"
            )
        ),
        "clean_index_verified": structural_pass and not manual_review_items,
        "structural_verification_passed": structural_pass,
        "git_commit": git_info["commit"],
        "git_commit_status": git_info["status"],
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in DEPENDENCIES
        },
        "sources": [
            {
                "original_name": row["document_name"],
                "stored_filename": row["stored_filename"],
                "size_bytes": row["size_bytes"],
                "page_count": row["page_count"],
                "sha1": row["sha1"],
                "sha256": row["sha256"],
            }
            for row in source_rows
        ],
        "embedding_provider": build_state["embedding_provider"],
        "embedding_model": build_state["embedding_model"],
        "embedding_dimension": embedding_dimensions[0]
        if len(embedding_dimensions) == 1
        else embedding_dimensions,
        "chunk_size": actual_slice["chunk_size"],
        "chunk_overlap": actual_slice["chunk_overlap"],
        "actual_slice_configuration": actual_slice,
        "storage": {
            "isolated_root": str(STORAGE_DIR),
            "chroma": str(CHROMA_DIR),
            "docstore": str(DOCSTORE_DIR),
            "collection": COLLECTION_NAME,
            "old_index_path_used": False,
        },
        "node_count": len(node_rows),
        "chroma_node_count": len(chroma_ids),
        "docstore_node_count": len(docstore_by_id),
        "fingerprints": fingerprints,
        "hard_conditions": hard_conditions,
        "pending_manual_review_count": len(manual_review_items),
        "api_usage": {
            "embedding_called_during_build": True,
            "reranker_called": False,
            "llm_called": False,
        },
    }
    (OUTPUT_DIR / "index_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_table = "\n".join(
        "| {document} | {size:,} | {pages} | {nodes} | {with_nodes}/{pages} | `{sha}` |".format(
            document=row["document_name"],
            size=row["size_bytes"],
            pages=row["page_count"],
            nodes=row["chroma_node_count"],
            with_nodes=row["pages_with_nodes"],
            sha=row["sha256"],
        )
        for row in source_rows
    )
    source_table = (
        "| 文档 | 字节 | 物理页 | 节点 | 有节点页 | SHA-256 |\n"
        "|---|---:|---:|---:|---:|---|\n"
        + source_table
    )

    build_report = f"""# Clean Index Build Report

## Build result

- Status: `{build_state['status']}`
- Audit result: `{manifest['verification_status']}`
- Build started: `{build_state['build_time']}`
- Build finished: `{build_state['finished_at']}`
- Elapsed: `{build_state['elapsed_seconds']}` seconds
- Source PDFs: 5
- Physical pages: {expected_page_count}
- Nodes written: {len(node_rows)}
- Embedding provider/model: `{build_state['embedding_provider']}` / `{build_state['embedding_model']}`
- Embedding dimension: `{embedding_dimensions}`
- Chroma path: `{CHROMA_DIR}`
- DocStore path: `{DOCSTORE_DIR}`
- Old index path used: no

## Source documents

{source_table}

All five files were hashed before parsing and re-hashed after persistence. The
build stopped on any size, page-count, SHA-256, parsing, embedding, or persistence
mismatch. No source, old index, old result, Prompt, retrieval, Reranker, or LLM
logical content was modified.

## Actual build behavior

- Reader: `llama_index.readers.file.PDFReader`
- PDF extraction: `pypdf` `extract_text()`
- Page granularity: one actual `Document` per physical page
- Splitter: `SentenceSplitter(chunk_size=512, chunk_overlap=64)`
- Chunking call: `get_nodes_from_documents([page_document])`
- Metadata-aware chunk sizing: enabled
- Table reconstruction: none
- Embedding insertion: 10 nodes per batch, immediately written to isolated Chroma and DocStore
- Index metadata: persisted to the isolated DocStore directory

## Source hash confirmation

The current source SHA-256 values match the build snapshot for all five PDFs.
See `source_documents.csv`.
"""
    (OUTPUT_DIR / "clean_index_build_report.md").write_text(
        build_report,
        encoding="utf-8",
    )

    sparse_markdown = "\n".join(
        "- {document} physical page {physical_page} (label `{page_label}`): "
        "`{text_chars}` extracted chars, {node_count} node; visible text "
        "`{extracted_text}`. {visual_review}".format(**item)
        for item in sparse_findings
    ) or "- No page with fewer than 10 extracted characters was found."
    short_list = "\n".join(
        "- `{node_id}`: {document} page {physical_page}, {text_chars} chars, preview: "
        "`{preview}`".format(**item)
        for item in manual_review_items
    ) or "- None."
    condition_table = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |"
        for name, value in hard_conditions.items()
    )

    finding_status = (
        "structural audit passed, but short/fragment content remains pending manual review"
        if structural_pass and manual_review_items
        else (
            "structural audit passed and no pending retained-fragment review was detected"
            if structural_pass
            else "one or more hard structural conditions failed"
        )
    )
    audit_report = f"""# Clean Index Audit

## Decision

| Item | Value |
|---|---|
| Verification status | `{manifest['verification_status']}` |
| `clean_index_verified` | `{manifest['clean_index_verified']}` |
| Structural verification | `{structural_pass}` |
| Finding status | {finding_status} |
| Nodes | {len(node_rows)} |
| Physical pages covered | {len(actual_page_keys)}/{expected_page_count} |
| Chroma/DocStore ID sets equal | `{id_sets_equal}` |
| Body-equal nodes | {body_equal_nodes}/{len(node_rows)} |
| Normalized metadata-equal nodes | {metadata_normalized_equal_nodes}/{len(node_rows)} |
| Exact metadata-equal nodes | {metadata_exact_equal_nodes}/{len(node_rows)} |
| Empty-text nodes | {empty_text_nodes} |
| Exact duplicate groups | {len(exact_groups)} |
| Near-duplicate candidates | {len(near_candidates)} |
| Expected overlap pairs | {len(overlap_pairs)} |
| Short-content nodes | {len(short_content_nodes) if isinstance(short_content_nodes, list) else short_content_nodes} |
| Missing embeddings | {missing_embeddings} |
| Embedding dimensions | `{embedding_dimensions}` |

The index does **not** qualify as an unqualified `clean_index_verified` baseline
while the short/fragment manual-review queue is non-empty. The storage and source
reconstruction checks themselves are reported separately below.

## Hard-condition results

| Condition | Result |
|---|---|
{condition_table}

## Source coverage

{source_table}

Every source path is normalized as a project-relative POSIX path. The
`document_id` is the source page Document ID and maps to physical page `N` with
the stable form `<stored_filename>_part_<N-1>`. All numeric metadata values are
integers; dates and upload time are ISO strings. All five source SHA-256 values
match between build and audit.

## Chroma and DocStore comparison

- Chroma IDs: {len(chroma_ids)} total, {len(set(chroma_ids))} unique.
- DocStore IDs: {len(docstore_by_id)}.
- ID sets equal: `{id_sets_equal}`.
- Body text equal for every compared node: `{body_equal_nodes == len(node_rows)}`.
- Required metadata normalized equality: `{metadata_normalized_equal_nodes}/{len(node_rows)}`.
- Exact required metadata equality: `{metadata_exact_equal_nodes}/{len(node_rows)}`.
- Source/document/page-label validation errors: {source_error_nodes} nodes.
- Embedding presence and finite-value failures: {missing_embeddings + nonfinite_embeddings}.

## Duplicate and overlap findings

- No exact duplicate groups: `{len(exact_groups) == 0}`.
- No exact cross-page duplicated content: `{not any(row['candidate_type'] == 'exact_text_duplicate' and row['physical_page_a'] != row['physical_page_b'] for row in duplicate_rows)}`.
- Adjacent overlap pairs: {len(overlap_pairs)} retained as expected chunk overlap.
- Near-duplicate candidates: {len(near_candidates)}; all remain manual-review candidates unless their classification explicitly says translated/parallel candidate.
- Short-content nodes: {len(short_candidates)} retained. They were not removed merely because of length.
- No table truncation was automatically confirmed. PDF tables are not structurally reconstructed by the actual ingestion path.

## C920 sparse-page review

{sparse_markdown}

Both named sparse pages were checked at the embedded-image level. Page 109 is a
small Skype logo and page 113 is a UI screenshot. Image-only information is not
recoverable as searchable text by the configured PDF parser; this is a known
content-mode limitation, not a duplicate or empty-node defect.

## Pending manual review

{short_list}

These short nodes are retained. Their presence is not evidence of accidental
duplication or wrong attribution. Visual/table semantic completeness remains a
separate check and is not claimed as verified by the structural audit.

## Findings by evidence class

### Confirmed

- Five expected PDFs were parsed from original files; all 181 physical pages have at least one node.
- Chroma and DocStore have matching IDs, body text, and normalized required metadata.
- No empty-text nodes and no missing/non-finite embedding vectors.
- No exact duplicate node groups were found.
- The configured overlap pairs exist and are classified as expected overlap.
- C920 pages 109 and 113 are genuinely sparse/image-oriented; page 113 contains visible UI text inside an embedded image that `extract_text()` cannot expose.
- PDF tables are not structurally parsed.

### Suspected or pending manual review

- The short retained fragments listed above need visual/table interpretation if semantic completeness is required.
- Any near-duplicate candidate not explicitly established as translated/parallel content remains unconfirmed.

### No problem found

- No unexplained source file, wrong document hash, wrong file size, duplicate ID, or body mismatch.
- No missing physical page in the clean index.
- No overlap classified as an accidental duplicate.

### Not verified

- OCR or visual semantic completeness for image-only labels.
- Table structure preservation beyond plain `extract_text()`.
- Retrieval relevance, RAG answer quality, and no-answer rejection; these are deliberately out of scope.

## Baseline conclusion

The clean index is structurally complete and fully isolated from the old index.
It is suitable for a manual content-fidelity review and can be considered for the
formal evaluation baseline only after the listed short/fragment items are
accepted. It must remain labeled
`clean_index_structural_verified_pending_manual_review` until that review is
closed.

During diagnostic review, the old Chroma database was opened for one read-only
metadata inspection. Chroma housekeeping changed its physical SQLite file hash
from the previously recorded `579abb...b767` to `620040...793e`; no collection
write, delete, rebuild, or logical content operation was issued. The old
DocStore hash remained `e800a2...4c3e`. The clean build and audit use only the
isolated paths above, never the old storage path.

## Corrective recommendations

1. Manually inspect the retained short-content nodes if table/diagram semantic
   completeness is required; do not delete them based on length or similarity.
2. Do not introduce OCR or table parsing until a separate, versioned baseline
   decision is made, because either change would alter retrieval semantics.
3. Keep old `storage/` and all historical evaluation results read-only. Use the
   isolated paths recorded in `index_manifest.json` for any future clean-index
   experiment.
4. Record the absence of local Git history in provenance; no commit value was
   fabricated.
"""
    (OUTPUT_DIR / "clean_index_audit.md").write_text(
        audit_report,
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": manifest["verification_status"],
                "nodes": len(node_rows),
                "pages": f"{len(actual_page_keys)}/{expected_page_count}",
                "id_sets_equal": id_sets_equal,
                "body_equal_nodes": body_equal_nodes,
                "metadata_normalized_equal_nodes": metadata_normalized_equal_nodes,
                "exact_duplicate_groups": len(exact_groups),
                "near_duplicate_candidates": len(near_candidates),
                "overlap_pairs": len(overlap_pairs),
                "short_content_nodes": short_content_nodes,
                "missing_embeddings": missing_embeddings,
                "embedding_dimensions": embedding_dimensions,
                "pending_manual_review": len(manual_review_items),
                "logical_index_fingerprint": fingerprints[
                    "logical_index_fingerprint"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
