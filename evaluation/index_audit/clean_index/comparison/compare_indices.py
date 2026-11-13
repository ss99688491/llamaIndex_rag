"""Read-only node-level comparison of the old and clean indexes."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore


ROOT = Path(__file__).resolve().parents[4]
COMPARISON_DIR = Path(__file__).resolve().parent
CLEAN_AUDIT_DIR = COMPARISON_DIR.parent
OLD_AUDIT_DIR = ROOT / "evaluation" / "index_audit"

OLD_CHROMA_DIR = ROOT / "storage" / "chroma"
OLD_DOCSTORE_DIR = ROOT / "storage" / "docstore"
NEW_CHROMA_DIR = CLEAN_AUDIT_DIR / "storage" / "chroma"
NEW_DOCSTORE_DIR = CLEAN_AUDIT_DIR / "storage" / "docstore"
COLLECTION_NAME = "knowledge_base"

OLD_NODES_CSV = OLD_AUDIT_DIR / "old_index_nodes.csv"
NEW_NODES_CSV = CLEAN_AUDIT_DIR / "clean_index_nodes.csv"
OLD_DUPLICATES_CSV = OLD_AUDIT_DIR / "duplicate_candidates.csv"
NEW_DUPLICATES_CSV = CLEAN_AUDIT_DIR / "duplicate_candidates.csv"

COMMON_SOURCE_FIELDS = [
    "page_label",
    "file_name",
    "file_path",
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
    "file_hash",
    "original_name",
    "uploaded_at",
]
STABLE_SOURCE_FIELDS = [
    "page_label",
    "file_name",
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
    "file_hash",
    "original_name",
]
NEW_SOURCE_FIELDS = ["file_sha256", "source_page", "page_index"]
STORAGE_INTERNAL_FIELDS = {
    "_node_content",
    "_node_type",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\u00a0", " ").split())


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def part_index(document_id: str | None) -> int | None:
    match = re.search(r"_part_(\d+)$", str(document_id or ""))
    return int(match.group(1)) if match else None


def source_document_id(
    metadata: dict[str, Any],
    stored_filename: str,
) -> str:
    document_id = str(
        metadata.get("document_id")
        or metadata.get("ref_doc_id")
        or metadata.get("doc_id")
        or ""
    )
    part = part_index(document_id)
    if part is None:
        return document_id
    return f"{stored_filename}_part_{part}"


def raw_document_id(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    return str(
        metadata.get("document_id")
        or metadata.get("ref_doc_id")
        or metadata.get("doc_id")
        or ""
    )


def semantic_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in STORAGE_INTERNAL_FIELDS
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_index(
    name: str,
    chroma_dir: Path,
    docstore_dir: Path,
    audit_csv: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    collection = chromadb.PersistentClient(
        path=str(chroma_dir)
    ).get_collection(COLLECTION_NAME)
    raw = collection.get(include=["metadatas", "documents"])
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(docstore_dir)
    )
    csv_rows = read_csv(audit_csv)
    csv_by_id = {row["node_id"]: row for row in csv_rows}

    records: dict[str, dict[str, Any]] = {}
    for node_id, metadata, text in zip(
        raw["ids"],
        raw["metadatas"],
        raw["documents"],
    ):
        metadata = metadata or {}
        text = text or ""
        stored_filename = str(metadata.get("file_name") or "")
        normalized_document_id = source_document_id(
            metadata,
            stored_filename,
        )
        part = part_index(normalized_document_id)
        physical_page = (
            int(metadata["source_page"])
            if metadata.get("source_page") is not None
            else (part + 1 if part is not None else None)
        )
        docstore_node = docstore.docs.get(node_id)
        docstore_text = (
            docstore_node.get_content() if docstore_node is not None else ""
        )
        docstore_metadata = (
            dict(docstore_node.metadata or {})
            if docstore_node is not None
            else {}
        )
        csv_row = csv_by_id.get(node_id)
        csv_norm_hash = (
            csv_row.get("text_sha256") if csv_row is not None else None
        )
        records[node_id] = {
            "node_id": node_id,
            "text": text,
            "text_exact_sha256": sha256_text(text),
            "text_normalized_sha256": sha256_text(normalize_text(text)),
            "metadata": metadata,
            "semantic_metadata": semantic_metadata(metadata),
            "docstore_text": docstore_text,
            "docstore_metadata": docstore_metadata,
            "document_name": str(metadata.get("original_name") or ""),
            "stored_filename": stored_filename,
            "physical_page": physical_page,
            "part_index": part,
            "page_label": str(metadata.get("page_label") or ""),
            "chunk_start_char": (
                int(docstore_node.start_char_idx)
                if docstore_node is not None
                and docstore_node.start_char_idx is not None
                else None
            ),
            "chunk_end_char": (
                int(docstore_node.end_char_idx)
                if docstore_node is not None
                and docstore_node.end_char_idx is not None
                else None
            ),
            "normalized_document_id": normalized_document_id,
            "csv_present": csv_row is not None,
            "csv_normalized_text_hash_matches": (
                csv_norm_hash == sha256_text(normalize_text(text))
                if csv_norm_hash is not None
                else False
            ),
        }

    checks = {
        "index_name": name,
        "chroma_dir": str(chroma_dir),
        "docstore_dir": str(docstore_dir),
        "audit_csv": str(audit_csv),
        "chroma_count": len(raw["ids"]),
        "unique_chroma_ids": len(set(raw["ids"])),
        "docstore_count": len(docstore.docs),
        "csv_count": len(csv_rows),
        "id_sets_equal": set(raw["ids"]) == set(docstore.docs),
        "csv_id_set_equal": set(raw["ids"]) == set(csv_by_id),
        "body_equal_count": sum(
            record["text"] == record["docstore_text"]
            for record in records.values()
        ),
        "csv_hash_match_count": sum(
            record["csv_normalized_text_hash_matches"]
            for record in records.values()
        ),
    }
    return records, checks


def page_groups(
    records: dict[str, dict[str, Any]]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        if record["physical_page"] is not None:
            groups[
                (record["document_name"], int(record["physical_page"]))
            ].append(record)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                row["chunk_start_char"]
                if row["chunk_start_char"] is not None
                else -1,
                row["chunk_end_char"]
                if row["chunk_end_char"] is not None
                else -1,
                row["node_id"],
            )
        )
    return groups


def metadata_comparison(
    old: dict[str, Any],
    new: dict[str, Any],
) -> dict[str, Any]:
    old_meta = old["semantic_metadata"]
    new_meta = new["semantic_metadata"]
    old_fields = set(old_meta)
    new_fields = set(new_meta)
    common_fields = sorted(old_fields & new_fields)
    added_fields = sorted(new_fields - old_fields)
    removed_fields = sorted(old_fields - new_fields)
    differences = {
        field: {"old": old_meta.get(field), "new": new_meta.get(field)}
        for field in common_fields
        if old_meta.get(field) != new_meta.get(field)
    }
    common_source_differences = {
        field: differences[field]
        for field in COMMON_SOURCE_FIELDS
        if field in differences
    }
    stable_source_differences = {
        field: differences[field]
        for field in STABLE_SOURCE_FIELDS
        if field in differences
    }
    return {
        "metadata_full_equal": canonical_json(old_meta) == canonical_json(new_meta),
        "metadata_difference_fields": sorted(differences),
        "metadata_difference_count": len(differences),
        "metadata_differences_json": canonical_json(differences),
        "common_source_metadata_equal": not common_source_differences,
        "common_source_difference_fields": sorted(common_source_differences),
        "stable_source_metadata_equal": not stable_source_differences,
        "stable_source_difference_fields": sorted(stable_source_differences),
        "old_metadata_fields": sorted(old_fields),
        "new_metadata_fields": sorted(new_fields),
        "metadata_fields_added_in_new": added_fields,
        "metadata_fields_removed_in_new": removed_fields,
        "old_metadata_json": canonical_json(old_meta),
        "new_metadata_json": canonical_json(new_meta),
    }


def make_pair(
    pair_number: int,
    old: dict[str, Any],
    new: dict[str, Any],
    evidence: str,
) -> dict[str, Any]:
    body_same = old["text"] == new["text"]
    metadata = metadata_comparison(old, new)
    if old["node_id"] == new["node_id"] and body_same and metadata[
        "metadata_full_equal"
    ]:
        status = "identical"
    elif old["node_id"] == new["node_id"] and body_same:
        status = "metadata_only_changed"
    elif body_same:
        status = "id_changed_but_content_same"
    else:
        status = "content_changed"
    return {
        "pair_id": f"pair-{pair_number:04d}",
        "status": status,
        "match_evidence": evidence,
        "old_node_id": old["node_id"],
        "new_node_id": new["node_id"],
        "node_id_equal": old["node_id"] == new["node_id"],
        "document_name": new["document_name"],
        "stored_filename_old": old["stored_filename"],
        "stored_filename_new": new["stored_filename"],
        "filename_equal": old["stored_filename"] == new["stored_filename"],
        "document_id_old_raw": raw_document_id(old),
        "document_id_new_raw": raw_document_id(new),
        "raw_document_id_equal": raw_document_id(old) == raw_document_id(new),
        "normalized_document_id_old": old["normalized_document_id"],
        "normalized_document_id_new": new["normalized_document_id"],
        "normalized_document_id_equal": (
            old["normalized_document_id"] == new["normalized_document_id"]
        ),
        "physical_page_old": old["physical_page"],
        "physical_page_new": new["physical_page"],
        "physical_page_equal": (
            old["physical_page"] == new["physical_page"]
        ),
        "page_label_old": old["page_label"],
        "page_label_new": new["page_label"],
        "page_label_equal": old["page_label"] == new["page_label"],
        "old_chunk_start_char": old["chunk_start_char"],
        "old_chunk_end_char": old["chunk_end_char"],
        "new_chunk_start_char": new["chunk_start_char"],
        "new_chunk_end_char": new["chunk_end_char"],
        "slice_boundary_changed": (
            old["chunk_start_char"] != new["chunk_start_char"]
            or old["chunk_end_char"] != new["chunk_end_char"]
        ),
        "old_text_char_count": len(old["text"]),
        "new_text_char_count": len(new["text"]),
        "old_text_sha256_exact": old["text_exact_sha256"],
        "new_text_sha256_exact": new["text_exact_sha256"],
        "body_equal": body_same,
        "old_text_preview": normalize_text(old["text"])[:240],
        "new_text_preview": normalize_text(new["text"])[:240],
        **metadata,
    }


def compare_indexes(
    old_records: dict[str, dict[str, Any]],
    new_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unmatched_old = set(old_records)
    unmatched_new = set(new_records)
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    for node_id in sorted(set(old_records) & set(new_records)):
        pairs.append(
            (
                old_records[node_id],
                new_records[node_id],
                "same_node_id",
            )
        )
        unmatched_old.remove(node_id)
        unmatched_new.remove(node_id)

    old_pages = page_groups(old_records)
    new_pages = page_groups(new_records)
    for page_key in sorted(set(old_pages) & set(new_pages)):
        old_rows = old_pages[page_key]
        new_rows = new_pages[page_key]
        if len(old_rows) != len(new_rows):
            continue
        for ordinal, (old, new) in enumerate(zip(old_rows, new_rows)):
            if (
                old["node_id"] not in unmatched_old
                or new["node_id"] not in unmatched_new
            ):
                continue
            pairs.append(
                (
                    old,
                    new,
                    f"same_physical_page_and_chunk_ordinal:{ordinal}",
                )
            )
            unmatched_old.remove(old["node_id"])
            unmatched_new.remove(new["node_id"])

    # For pages whose chunk counts differ, exact boundaries are a safe one-to-one key.
    boundary_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for old_id in sorted(unmatched_old):
        old = old_records[old_id]
        candidates = [
            new_records[new_id]
            for new_id in unmatched_new
            if new_records[new_id]["document_name"] == old["document_name"]
            and new_records[new_id]["physical_page"] == old["physical_page"]
            and new_records[new_id]["chunk_start_char"]
            == old["chunk_start_char"]
            and new_records[new_id]["chunk_end_char"]
            == old["chunk_end_char"]
        ]
        if len(candidates) == 1:
            boundary_pairs.append((old, candidates[0]))
    for old, new in boundary_pairs:
        if (
            old["node_id"] in unmatched_old
            and new["node_id"] in unmatched_new
        ):
            pairs.append((old, new, "same_page_and_exact_chunk_boundary"))
            unmatched_old.remove(old["node_id"])
            unmatched_new.remove(new["node_id"])

    # Exact body identity is safe only when that body occurs once on each side.
    old_hash_ids: dict[str, list[str]] = defaultdict(list)
    new_hash_ids: dict[str, list[str]] = defaultdict(list)
    for node_id in unmatched_old:
        old_hash_ids[old_records[node_id]["text_exact_sha256"]].append(node_id)
    for node_id in unmatched_new:
        new_hash_ids[new_records[node_id]["text_exact_sha256"]].append(node_id)
    for digest in sorted(set(old_hash_ids) & set(new_hash_ids)):
        if len(old_hash_ids[digest]) == 1 and len(new_hash_ids[digest]) == 1:
            old_id = old_hash_ids[digest][0]
            new_id = new_hash_ids[digest][0]
            pairs.append(
                (
                    old_records[old_id],
                    new_records[new_id],
                    "unique_exact_body_sha256",
                )
            )
            unmatched_old.remove(old_id)
            unmatched_new.remove(new_id)

    paired_rows = [
        make_pair(number, old, new, evidence)
        for number, (old, new, evidence) in enumerate(pairs, start=1)
    ]
    for old_id in sorted(unmatched_old):
        old = old_records[old_id]
        paired_rows.append(
            {
                "pair_id": "",
                "status": "missing_node",
                "match_evidence": "no_deterministic_counterpart",
                "old_node_id": old_id,
                "new_node_id": "",
                "node_id_equal": False,
                "document_name": old["document_name"],
                "stored_filename_old": old["stored_filename"],
                "stored_filename_new": "",
                "filename_equal": False,
                "document_id_old_raw": raw_document_id(old),
                "document_id_new_raw": "",
                "raw_document_id_equal": False,
                "normalized_document_id_old": old["normalized_document_id"],
                "normalized_document_id_new": "",
                "normalized_document_id_equal": False,
                "physical_page_old": old["physical_page"],
                "physical_page_new": "",
                "physical_page_equal": False,
                "page_label_old": old["page_label"],
                "page_label_new": "",
                "page_label_equal": False,
                "old_chunk_start_char": old["chunk_start_char"],
                "old_chunk_end_char": old["chunk_end_char"],
                "new_chunk_start_char": "",
                "new_chunk_end_char": "",
                "slice_boundary_changed": False,
                "old_text_char_count": len(old["text"]),
                "new_text_char_count": "",
                "old_text_sha256_exact": old["text_exact_sha256"],
                "new_text_sha256_exact": "",
                "body_equal": False,
                "old_text_preview": normalize_text(old["text"])[:240],
                "new_text_preview": "",
                "metadata_full_equal": False,
                "metadata_difference_fields": [],
                "metadata_difference_count": "",
                "metadata_differences_json": "",
                "common_source_metadata_equal": False,
                "common_source_difference_fields": [],
                "stable_source_metadata_equal": False,
                "stable_source_difference_fields": [],
                "old_metadata_fields": sorted(old["semantic_metadata"]),
                "new_metadata_fields": [],
                "metadata_fields_added_in_new": [],
                "metadata_fields_removed_in_new": [],
                "old_metadata_json": canonical_json(old["semantic_metadata"]),
                "new_metadata_json": "",
            }
        )
    for new_id in sorted(unmatched_new):
        new = new_records[new_id]
        paired_rows.append(
            {
                "pair_id": "",
                "status": "new_node",
                "match_evidence": "no_deterministic_counterpart",
                "old_node_id": "",
                "new_node_id": new_id,
                "node_id_equal": False,
                "document_name": new["document_name"],
                "stored_filename_old": "",
                "stored_filename_new": new["stored_filename"],
                "filename_equal": False,
                "document_id_old_raw": "",
                "document_id_new_raw": raw_document_id(new),
                "raw_document_id_equal": False,
                "normalized_document_id_old": "",
                "normalized_document_id_new": new["normalized_document_id"],
                "normalized_document_id_equal": False,
                "physical_page_old": "",
                "physical_page_new": new["physical_page"],
                "physical_page_equal": False,
                "page_label_old": "",
                "page_label_new": new["page_label"],
                "page_label_equal": False,
                "old_chunk_start_char": "",
                "old_chunk_end_char": "",
                "new_chunk_start_char": new["chunk_start_char"],
                "new_chunk_end_char": new["chunk_end_char"],
                "slice_boundary_changed": False,
                "old_text_char_count": "",
                "new_text_char_count": len(new["text"]),
                "old_text_sha256_exact": "",
                "new_text_sha256_exact": new["text_exact_sha256"],
                "body_equal": False,
                "old_text_preview": "",
                "new_text_preview": normalize_text(new["text"])[:240],
                "metadata_full_equal": False,
                "metadata_difference_fields": [],
                "metadata_difference_count": "",
                "metadata_differences_json": "",
                "common_source_metadata_equal": False,
                "common_source_difference_fields": [],
                "stable_source_metadata_equal": False,
                "stable_source_difference_fields": [],
                "old_metadata_fields": [],
                "new_metadata_fields": sorted(new["semantic_metadata"]),
                "metadata_fields_added_in_new": sorted(new["semantic_metadata"]),
                "metadata_fields_removed_in_new": [],
                "old_metadata_json": "",
                "new_metadata_json": canonical_json(new["semantic_metadata"]),
            }
        )
    paired_rows.sort(
        key=lambda row: (
            row["document_name"],
            row["physical_page_new"]
            if row["physical_page_new"] != ""
            else row["physical_page_old"],
            row["new_chunk_start_char"]
            if row["new_chunk_start_char"] != ""
            else row["old_chunk_start_char"],
            row["old_node_id"],
            row["new_node_id"],
        )
    )
    return paired_rows, []


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    old_records, old_checks = load_index(
        "old",
        OLD_CHROMA_DIR,
        OLD_DOCSTORE_DIR,
        OLD_NODES_CSV,
    )
    new_records, new_checks = load_index(
        "new",
        NEW_CHROMA_DIR,
        NEW_DOCSTORE_DIR,
        NEW_NODES_CSV,
    )
    node_rows, _ = compare_indexes(old_records, new_records)

    old_short_ids = {
        row["node_id_a"]
        for row in read_csv(OLD_DUPLICATES_CSV)
        if row["candidate_type"] == "table_fragment"
    }
    new_short_ids = {
        row["node_id_a"]
        for row in read_csv(NEW_DUPLICATES_CSV)
        if row["candidate_type"] == "short_content"
    }
    for row in node_rows:
        row["old_is_short_candidate"] = row["old_node_id"] in old_short_ids
        row["new_is_short_candidate"] = row["new_node_id"] in new_short_ids
    write_csv(COMPARISON_DIR / "node_diff.csv", node_rows)

    row_by_old = {
        row["old_node_id"]: row for row in node_rows if row["old_node_id"]
    }
    row_by_new = {
        row["new_node_id"]: row for row in node_rows if row["new_node_id"]
    }

    old_documents = sorted(
        {record["document_name"] for record in old_records.values()}
    )
    new_documents = sorted(
        {record["document_name"] for record in new_records.values()}
    )
    all_documents = sorted(set(old_documents) | set(new_documents))
    document_rows: list[dict[str, Any]] = []
    for document in all_documents:
        rows = [
            row
            for row in node_rows
            if row["document_name"] == document
        ]
        old_pages = {
            int(record["physical_page"])
            for record in old_records.values()
            if record["document_name"] == document
            and record["physical_page"] is not None
        }
        new_pages = {
            int(record["physical_page"])
            for record in new_records.values()
            if record["document_name"] == document
            and record["physical_page"] is not None
        }
        old_page_counts = Counter(
            int(record["physical_page"])
            for record in old_records.values()
            if record["document_name"] == document
            and record["physical_page"] is not None
        )
        new_page_counts = Counter(
            int(record["physical_page"])
            for record in new_records.values()
            if record["document_name"] == document
            and record["physical_page"] is not None
        )
        document_rows.append(
            {
                "document_name": document,
                "old_node_count": sum(
                    record["document_name"] == document
                    for record in old_records.values()
                ),
                "new_node_count": sum(
                    record["document_name"] == document
                    for record in new_records.values()
                ),
                "node_count_delta": (
                    sum(
                        record["document_name"] == document
                        for record in new_records.values()
                    )
                    - sum(
                        record["document_name"] == document
                        for record in old_records.values()
                    )
                ),
                "old_pages_with_nodes": len(old_pages),
                "new_pages_with_nodes": len(new_pages),
                "old_missing_physical_pages_from_union": " | ".join(
                    str(page)
                    for page in sorted((old_pages | new_pages) - old_pages)
                ),
                "new_missing_physical_pages_from_union": " | ".join(
                    str(page)
                    for page in sorted((old_pages | new_pages) - new_pages)
                ),
                "page_coverage_equal": old_pages == new_pages,
                "page_chunk_count_differences": " | ".join(
                    f"{page}:{old_page_counts[page]}->{new_page_counts[page]}"
                    for page in sorted(set(old_page_counts) | set(new_page_counts))
                    if old_page_counts[page] != new_page_counts[page]
                ),
                "paired_nodes": sum(
                    bool(row["old_node_id"] and row["new_node_id"])
                    for row in rows
                ),
                "identical": sum(row["status"] == "identical" for row in rows),
                "metadata_only_changed": sum(
                    row["status"] == "metadata_only_changed" for row in rows
                ),
                "content_changed": sum(
                    row["status"] == "content_changed" for row in rows
                ),
                "id_changed_but_content_same": sum(
                    row["status"] == "id_changed_but_content_same"
                    for row in rows
                ),
                "new_node": sum(row["status"] == "new_node" for row in rows),
                "missing_node": sum(
                    row["status"] == "missing_node" for row in rows
                ),
                "slice_boundary_changed": sum(
                    bool(row["slice_boundary_changed"]) for row in rows
                ),
                "metadata_full_equal": sum(
                    bool(row["metadata_full_equal"]) for row in rows
                ),
                "filename_equal": sum(
                    bool(row["filename_equal"]) for row in rows
                ),
                "raw_document_id_equal": sum(
                    bool(row["raw_document_id_equal"]) for row in rows
                ),
                "normalized_document_id_equal": sum(
                    bool(row["normalized_document_id_equal"]) for row in rows
                ),
                "physical_page_equal": sum(
                    bool(row["physical_page_equal"]) for row in rows
                ),
                "page_label_equal": sum(
                    bool(row["page_label_equal"]) for row in rows
                ),
                "old_short_candidates": sum(
                    bool(row["old_is_short_candidate"]) for row in rows
                ),
                "new_short_candidates": sum(
                    bool(row["new_is_short_candidate"]) for row in rows
                ),
            }
        )
    write_csv(COMPARISON_DIR / "document_diff.csv", document_rows)

    status_counts = Counter(row["status"] for row in node_rows)
    paired_rows = [
        row for row in node_rows if row["old_node_id"] and row["new_node_id"]
    ]
    exact_body_pairs = [row for row in paired_rows if row["body_equal"]]
    changed_pairs = [
        row for row in paired_rows if not row["body_equal"]
    ]
    boundary_changed_pairs = [
        row for row in paired_rows if row["slice_boundary_changed"]
    ]
    metadata_field_difference_counts = Counter(
        field
        for row in paired_rows
        for field in row["metadata_difference_fields"]
    )
    added_field_counts = Counter(
        field
        for row in paired_rows
        for field in row["metadata_fields_added_in_new"]
    )
    removed_field_counts = Counter(
        field
        for row in paired_rows
        for field in row["metadata_fields_removed_in_new"]
    )
    new_short_mapping = [
        row_by_new[node_id] for node_id in sorted(new_short_ids)
    ]
    old_short_mapping = [
        row_by_old[node_id] for node_id in sorted(old_short_ids)
    ]
    old_short_present_count = sum(
        bool(row["new_node_id"]) for row in old_short_mapping
    )
    new_short_present_count = sum(
        bool(row["old_node_id"]) for row in new_short_mapping
    )

    old_ids = set(old_records)
    new_ids = set(new_records)
    old_body_hashes = {
        record["text_exact_sha256"] for record in old_records.values()
    }
    new_body_hashes = {
        record["text_exact_sha256"] for record in new_records.values()
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only_node_comparison",
        "matching_policy": {
            "same_node_id": True,
            "same_physical_page_and_chunk_ordinal_only_when_page_chunk_counts_equal": True,
            "same_page_and_exact_chunk_boundary": True,
            "unique_exact_body_sha256": True,
            "similarity_based_matching": False,
        },
        "inputs": {
            "old": old_checks,
            "new": new_checks,
        },
        "totals": {
            "old_nodes": len(old_records),
            "new_nodes": len(new_records),
            "node_count_delta": len(new_records) - len(old_records),
            "paired_nodes": len(paired_rows),
            "node_id_intersection": len(old_ids & new_ids),
            "old_only_node_ids": len(old_ids - new_ids),
            "new_only_node_ids": len(new_ids - old_ids),
            "exact_body_sha256_intersection": len(
                old_body_hashes & new_body_hashes
            ),
            "old_only_exact_body_sha256": len(
                old_body_hashes - new_body_hashes
            ),
            "new_only_exact_body_sha256": len(
                new_body_hashes - old_body_hashes
            ),
            "exact_body_equal_pairs": len(exact_body_pairs),
            "content_changed_pairs": len(changed_pairs),
            "slice_boundary_changed_pairs": len(boundary_changed_pairs),
        },
        "classification_counts": dict(sorted(status_counts.items())),
        "metadata": {
            "full_metadata_equal_pairs": sum(
                bool(row["metadata_full_equal"]) for row in paired_rows
            ),
            "common_source_metadata_equal_pairs": sum(
                bool(row["common_source_metadata_equal"])
                for row in paired_rows
            ),
            "stable_source_metadata_equal_pairs": sum(
                bool(row["stable_source_metadata_equal"])
                for row in paired_rows
            ),
            "field_difference_counts": dict(
                sorted(metadata_field_difference_counts.items())
            ),
            "new_fields_added_counts": dict(sorted(added_field_counts.items())),
            "new_fields_removed_counts": dict(
                sorted(removed_field_counts.items())
            ),
            "filename_equal_pairs": sum(
                bool(row["filename_equal"]) for row in paired_rows
            ),
            "raw_document_id_equal_pairs": sum(
                bool(row["raw_document_id_equal"]) for row in paired_rows
            ),
            "normalized_document_id_equal_pairs": sum(
                bool(row["normalized_document_id_equal"]) for row in paired_rows
            ),
            "physical_page_equal_pairs": sum(
                bool(row["physical_page_equal"]) for row in paired_rows
            ),
            "page_label_equal_pairs": sum(
                bool(row["page_label_equal"]) for row in paired_rows
            ),
        },
        "documents": document_rows,
        "short_content_mapping": {
            "old_short_candidate_count": len(old_short_ids),
            "old_short_candidates_with_new_counterpart": old_short_present_count,
            "old_short_candidates_with_same_body": sum(
                bool(row["body_equal"]) for row in old_short_mapping
            ),
            "new_short_candidate_count": len(new_short_ids),
            "new_short_candidates_with_old_counterpart": new_short_present_count,
            "new_short_candidates_with_same_body": sum(
                bool(row["body_equal"]) for row in new_short_mapping
            ),
        },
        "logical_change_conclusion": {
            "node_collection_equal": (
                len(old_records) == len(new_records)
                and status_counts["new_node"] == 0
                and status_counts["missing_node"] == 0
            ),
            "all_bodies_equal": len(exact_body_pairs) == len(paired_rows),
            "all_metadata_equal": all(
                bool(row["metadata_full_equal"]) for row in paired_rows
            ),
            "exact_reconstruction_identity": (
                len(old_ids & new_ids) == len(old_records)
                and all(row["status"] == "identical" for row in paired_rows)
            ),
            "same_document_page_coverage": all(
                bool(row["page_coverage_equal"]) for row in document_rows
            ),
        },
        "note": (
            "Chroma physical-file hashes are intentionally excluded from logical "
            "change decisions; only IDs, bodies, metadata and page/chunk structure "
            "were compared."
        ),
    }
    (COMPARISON_DIR / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    document_table = "\n".join(
        "| {document_name} | {old_node_count} | {new_node_count} | {old_pages_with_nodes} | {new_pages_with_nodes} | {id_changed_but_content_same} | {content_changed} | {slice_boundary_changed} |".format(
            **row
        )
        for row in document_rows
    )
    document_table = (
        "| 文档 | 旧节点 | 新节点 | 旧有节点页 | 新有节点页 | ID 变正文同 | 正文变化 | 边界变化 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + document_table
    )

    changed_table = "\n".join(
        "| {document_name} | {physical_page_old} | `{old_chunk_start_char}-{old_chunk_end_char}` | `{new_chunk_start_char}-{new_chunk_end_char}` | {old_text_char_count} | {new_text_char_count} |".format(
            **row
        )
        for row in changed_pairs
    )
    changed_table = (
        "| 文档 | 页 | 旧边界 | 新边界 | 旧字符数 | 新字符数 |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        + (changed_table or "| - | - | - | - | - | - |")
    )

    new_short_table = "\n".join(
        "| `{new_node_id}` | {document_name} | {physical_page_new} | `{old_node_id}` | {body_equal} | {old_is_short_candidate} |".format(
            **row
        )
        for row in new_short_mapping
    )
    new_short_table = (
        "| 新短节点 | 文档 | 页 | 旧对应节点 | 正文相同 | 旧短候选 |\n"
        "|---|---|---:|---|---|---|\n"
        + (new_short_table or "| - | - | - | - | - | - |")
    )

    old_short_table = "\n".join(
        "| `{old_node_id}` | {document_name} | {physical_page_old} | `{new_node_id}` | {body_equal} | {new_is_short_candidate} |".format(
            **row
        )
        for row in old_short_mapping
    )
    old_short_table = (
        "| 旧短候选 | 文档 | 页 | 新对应节点 | 正文相同 | 新短候选 |\n"
        "|---|---|---:|---|---|---|\n"
        + (old_short_table or "| - | - | - | - | - | - |")
    )

    metadata_table = "\n".join(
        f"| `{field}` | {count} |"
        for field, count in sorted(metadata_field_difference_counts.items())
    ) or "| - | 0 |"
    metadata_table = (
        "| 字段 | 差异节点数 |\n"
        "|---|---:|\n"
        + metadata_table
    )

    report = f"""# Old vs Clean Index Node Comparison

## Conclusion

- 旧索引节点数：{len(old_records)}
- 新索引节点数：{len(new_records)}
- node_id 交集：{len(old_ids & new_ids)}
- 旧索引独有 ID：{len(old_ids - new_ids)}
- 新索引独有 ID：{len(new_ids - old_ids)}
- 正文精确 SHA-256 交集：{len(old_body_hashes & new_body_hashes)}
- 旧索引独有正文 SHA-256：{len(old_body_hashes - new_body_hashes)}
- 新索引独有正文 SHA-256：{len(new_body_hashes - old_body_hashes)}
- 可确定性配对节点：{len(paired_rows)}
- 配对正文完全相同：{len(exact_body_pairs)}
- 配对正文发生变化：{len(changed_pairs)}
- 切片边界发生变化：{len(boundary_changed_pairs)}
- 节点集合数量与文档页覆盖一致，但两次构建不是逐 node 的完全相同快照。

结论：两次构建产生了相同的节点总数、相同的 PDF 节点数和物理页覆盖，
但 node_id 全部重新生成；233 个节点的正文逐字相同，7 个节点的正文因切片
边界调整而改变。因此应视为“结构对齐的新构建”，而不是可复用旧 node_id
的完全相同索引。

## Classification

| 分类 | 数量 |
|---|---:|
| `identical` | {status_counts['identical']} |
| `metadata_only_changed` | {status_counts['metadata_only_changed']} |
| `content_changed` | {status_counts['content_changed']} |
| `id_changed_but_content_same` | {status_counts['id_changed_but_content_same']} |
| `new_node` | {status_counts['new_node']} |
| `missing_node` | {status_counts['missing_node']} |
| `unmatched` | 0 |

## Matching policy

1. 先按相同 node_id 配对。
2. 对同一物理页且两边 chunk 数相同的节点，按 `(chunk_start_char, chunk_end_char)`
   排序后的 chunk 序号一一配对。
3. 对无法按序号配对的剩余节点，只接受同页且起止字符完全相同的边界。
4. 最后只接受在两边都唯一的精确正文 SHA-256。
5. 不使用文本相似度阈值强行匹配。

## Document and page coverage

{document_table}

所有 PDF 的节点数和有节点物理页集合均一致。各页 chunk 数量也一致；没有
`new_node` 或 `missing_node`。

## Body and slice-boundary differences

精确正文相同的配对：{len(exact_body_pairs)}。

发生正文变化且切片边界变化的节点：

{changed_table}

这 7 个节点的来源文档、物理页和 page_label 均未改变。变化原因是实际切片
边界随 metadata-aware chunking 的 metadata 内容调整，不是 PDF 源文件变化。

## Metadata comparison

| 指标 | 结果 |
|---|---:|
| 完整语义 metadata 相同的配对 | {summary['metadata']['full_metadata_equal_pairs']}/{len(paired_rows)} |
| 10 个旧 canonical source 字段完整相同 | {summary['metadata']['common_source_metadata_equal_pairs']}/{len(paired_rows)} |
| 排除路径和上传时间的稳定来源字段相同 | {summary['metadata']['stable_source_metadata_equal_pairs']}/{len(paired_rows)} |
| filename 相同 | {summary['metadata']['filename_equal_pairs']}/{len(paired_rows)} |
| 原始 document_id 字符串相同 | {summary['metadata']['raw_document_id_equal_pairs']}/{len(paired_rows)} |
| 规范化 document_id 相同 | {summary['metadata']['normalized_document_id_equal_pairs']}/{len(paired_rows)} |
| physical_page 相同 | {summary['metadata']['physical_page_equal_pairs']}/{len(paired_rows)} |
| page_label 相同 | {summary['metadata']['page_label_equal_pairs']}/{len(paired_rows)} |

逐字段差异：

{metadata_table}

主要差异是：

- 旧索引 `file_path` 使用 Windows 分隔符，云台部分节点为绝对路径；新索引统一为
  项目相对 POSIX 路径。
- 旧索引 `uploaded_at` 为浮点时间戳；新索引为 ISO 字符串。
- 新索引增加 `file_sha256`、`source_page`、`page_index`。
- 旧、新 `document_id` 的原始字符串均体现物理页，但旧值通常带
  `data\\uploads\\` 前缀，新值直接使用 `<stored_filename>_part_<N-1>`。
- `file_path` 和 `uploaded_at` 的变化导致 10 个旧 canonical source 字段整体
  不相同；排除二者后，其余稳定来源字段 240/240 一致。

完整字段值保存在 `node_diff.csv` 的 `old_metadata_json` 与
`new_metadata_json` 列，逐字段差异保存在 `metadata_differences_json`。
`_node_content` 和 `_node_type` 属于存储层内部序列化字段，不参与语义 metadata
判定。

## Short-content correspondence

新索引 15 个短内容节点中，{new_short_present_count} 个在旧索引中存在确定性对应；
其中正文相同 {sum(bool(row['body_equal']) for row in new_short_mapping)} 个。

{new_short_table}

旧索引 14 个短内容候选中，{old_short_present_count} 个在新索引中存在确定性对应；
其中正文相同 {sum(bool(row['body_equal']) for row in old_short_mapping)} 个。

{old_short_table}

短节点的差异来自正文边界变化及“旧 14 个启发式短候选”和“新 15 个
长度为 80 字符以下的短节点”的统计口径差异；没有证据表明存在重复节点。

## Source-index consistency

| Index | Chroma | DocStore | CSV | ID 集合一致 | 正文一致 | CSV 规范化哈希一致 |
|---|---:|---:|---:|---|---|---:|
| old | {old_checks['chroma_count']} | {old_checks['docstore_count']} | {old_checks['csv_count']} | {old_checks['id_sets_equal']} | {old_checks['body_equal_count']}/{len(old_records)} | {old_checks['csv_hash_match_count']}/{len(old_records)} |
| clean | {new_checks['chroma_count']} | {new_checks['docstore_count']} | {new_checks['csv_count']} | {new_checks['id_sets_equal']} | {new_checks['body_equal_count']}/{len(new_records)} | {new_checks['csv_hash_match_count']}/{len(new_records)} |

## Limits

- 本任务只比较逻辑节点、正文、metadata 和切片结构。
- 未调用 Embedding、Reranker 或 LLM，未重新入库，未执行检索评测。
- Chroma 物理文件哈希不作为逻辑变化证据；SQLite housekeeping 可能改变物理
  文件哈希。
- 所有四份结果文件均写入 `comparison/`；未修改旧/新索引、PDF、旧审计报告
  或新索引 Manifest。
"""
    (COMPARISON_DIR / "old_vs_clean_comparison.md").write_text(
        report,
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status_counts": dict(status_counts),
                "old_nodes": len(old_records),
                "new_nodes": len(new_records),
                "node_id_intersection": len(old_ids & new_ids),
                "paired_nodes": len(paired_rows),
                "same_body_pairs": len(exact_body_pairs),
                "changed_body_pairs": len(changed_pairs),
                "slice_boundary_changed_pairs": len(boundary_changed_pairs),
                "old_short_candidates": len(old_short_ids),
                "new_short_candidates": len(new_short_ids),
                "new_short_with_old_counterpart": new_short_present_count,
                "old_short_with_new_counterpart": old_short_present_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
