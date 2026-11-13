"""Read-only structural audit of the current Chroma and DocStore index."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PDFReader
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CANONICAL_METADATA = {
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
}
CHROMA_INTERNAL_METADATA = {
    "_node_content",
    "_node_type",
    "doc_id",
    "document_id",
    "ref_doc_id",
}


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split())


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-6)


def metadata_differences(
    chroma_metadata: dict, docstore_metadata: dict
) -> tuple[bool, bool, list[str]]:
    keys = sorted(CANONICAL_METADATA)
    exact = all(
        key in chroma_metadata
        and key in docstore_metadata
        and chroma_metadata[key] == docstore_metadata[key]
        for key in keys
    )
    differing = []
    for key in keys:
        if key not in chroma_metadata or key not in docstore_metadata:
            differing.append(key)
            continue
        left, right = chroma_metadata[key], docstore_metadata[key]
        if left == right:
            continue
        if isinstance(left, float) and isinstance(right, float) and float_equal(left, right):
            differing.append(f"{key}:float_serialization")
        else:
            differing.append(key)
    normalized = not any(
        not value.endswith(":float_serialization") for value in differing
    )
    return exact, normalized, differing


def suffix_prefix_overlap(left: str, right: str, limit: int = 4000) -> int:
    left = left[-limit:]
    right = right[:limit]
    maximum = min(len(left), len(right))
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def shingles(text: str, size: int = 8) -> set[str]:
    if len(text) <= size:
        return {text}
    return {text[index:index + size] for index in range(len(text) - size + 1)}


def table_fragment_suspected(text: str) -> bool:
    compact = normalize(text)
    if not compact or len(compact) >= 160:
        return False
    digits = sum(char.isdigit() for char in compact)
    symbols = sum(char in "•|©®°×" for char in compact)
    return digits / max(1, len(compact)) >= 0.08 or symbols >= 3


def part_index(metadata: dict) -> int | None:
    value = metadata.get("doc_id") or metadata.get("document_id") or ""
    match = re.search(r"_part_(\d+)$", str(value))
    return int(match.group(1)) if match else None


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    pdf_info = {}
    pdf_pages = {}
    for path in pdfs:
        original_name = path.name.split("_", 2)[-1]
        reader = PdfReader(str(path))
        encrypted = reader.is_encrypted
        if encrypted:
            reader.decrypt("")
        pages = []
        for physical_page, page in enumerate(reader.pages, start=1):
            text = normalize(page.extract_text() or "")
            pages.append(
                {
                    "physical_page": physical_page,
                    "page_label": reader.page_labels[physical_page - 1],
                    "text": text,
                    "text_length": len(text),
                }
            )
        pdf_info[original_name] = {
            "document_name": original_name,
            "stored_filename": path.name,
            "file_path": str(path.relative_to(ROOT)),
            "absolute_path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha1": sha1(path),
            "sha256": sha256(path),
            "page_count": len(pages),
            "encrypted": encrypted,
            "extracted_characters": sum(page["text_length"] for page in pages),
        }
        pdf_pages[original_name] = pages

    collection = chromadb.PersistentClient(
        path=str(ROOT / "storage" / "chroma")
    ).get_collection("knowledge_base")
    raw = collection.get(include=["metadatas", "documents", "embeddings"])
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(ROOT / "storage" / "docstore")
    )

    chroma_ids = raw["ids"]
    chroma_by_id = {
        node_id: {
            "metadata": metadata or {},
            "text": text or "",
            "embedding": embedding,
        }
        for node_id, metadata, text, embedding in zip(
            raw["ids"], raw["metadatas"], raw["documents"], raw["embeddings"]
        )
    }
    docstore_by_id = docstore.docs

    content_groups: dict[str, list[str]] = defaultdict(list)
    for node_id, item in chroma_by_id.items():
        content_groups[text_hash(normalize(item["text"]))].append(node_id)
    exact_groups = {
        digest: ids for digest, ids in content_groups.items() if len(ids) > 1
    }
    exact_group_by_node = {}
    for group_index, ids in enumerate(
        sorted(exact_groups.values(), key=lambda values: sorted(values)), start=1
    ):
        label = f"exact-{group_index:03d}"
        for node_id in ids:
            exact_group_by_node[node_id] = label

    node_rows = []
    duplicate_rows = []
    for node_id in sorted(chroma_ids):
        chroma_item = chroma_by_id[node_id]
        chroma_metadata = chroma_item["metadata"]
        node = docstore_by_id.get(node_id)
        text = chroma_item["text"]
        document = chroma_metadata.get("original_name")
        part = part_index(chroma_metadata)
        physical_page = part + 1 if part is not None else None
        page_info = (
            pdf_pages.get(document, [])[part]
            if document in pdf_pages and part is not None and 0 <= part < len(pdf_pages[document])
            else None
        )
        docstore_text = node.get_content() if node is not None else ""
        docstore_metadata = dict(node.metadata or {}) if node is not None else {}
        metadata_exact, metadata_normalized, metadata_diffs = metadata_differences(
            chroma_metadata, docstore_metadata
        )
        missing = sorted(CANONICAL_METADATA - set(docstore_metadata))
        actual_pdf = pdf_info.get(document)
        abnormal_source = []
        if actual_pdf:
            if chroma_metadata.get("file_name") != actual_pdf["stored_filename"]:
                abnormal_source.append("file_name_mismatch")
            if str(chroma_metadata.get("file_path", "")).replace("/", "\\") != actual_pdf["file_path"].replace("/", "\\"):
                abnormal_source.append("file_path_mismatch")
            if chroma_metadata.get("file_hash") != actual_pdf["sha1"]:
                abnormal_source.append("file_hash_mismatch")
            if int(chroma_metadata.get("file_size") or -1) != actual_pdf["size_bytes"]:
                abnormal_source.append("file_size_mismatch")
            if str(chroma_metadata.get("page_label")) != str(page_info["page_label"] if page_info else ""):
                abnormal_source.append("page_label_mismatch")
            if part is None or part < 0 or part >= actual_pdf["page_count"]:
                abnormal_source.append("physical_page_out_of_range")
        else:
            abnormal_source.append("unknown_document")
        node_rows.append(
            {
                "node_id": node_id,
                "document_name": document,
                "stored_filename": chroma_metadata.get("file_name"),
                "file_path": chroma_metadata.get("file_path"),
                "page_label": chroma_metadata.get("page_label"),
                "physical_page": physical_page,
                "part_index": part,
                "chunk_start_char": getattr(node, "start_char_idx", None),
                "chunk_end_char": getattr(node, "end_char_idx", None),
                "text_char_count": len(text),
                "text_sha256": text_hash(normalize(text)),
                "text_preview": normalize(text)[:240],
                "text": text,
                "chroma_docstore_text_equal": text == docstore_text,
                "metadata_exact_equal": metadata_exact,
                "metadata_normalized_equal": metadata_normalized,
                "metadata_difference_fields": " | ".join(metadata_diffs),
                "missing_metadata_fields": " | ".join(missing),
                "abnormal_source_reasons": " | ".join(abnormal_source),
                "exact_duplicate_group": exact_group_by_node.get(node_id, ""),
                "empty_text": not bool(normalize(text)),
                "table_fragment_suspected": table_fragment_suspected(text),
            }
        )

    node_row_by_id = {row["node_id"]: row for row in node_rows}

    # Exact duplicates, including cross-page and same-page groups.
    for digest, ids in sorted(exact_groups.items(), key=lambda item: sorted(item[1])):
        ordered = sorted(ids)
        for left, right in zip(ordered, ordered[1:]):
            a, b = node_row_by_id[left], node_row_by_id[right]
            cross_page = (a["document_name"], a["physical_page"]) != (
                b["document_name"],
                b["physical_page"],
            )
            duplicate_rows.append(
                {
                    "candidate_type": "exact_text_duplicate",
                    "classification": "confirmed_exact_duplicate" if not cross_page else "exact_same_content_cross_page",
                    "node_id_a": left,
                    "node_id_b": right,
                    "document_a": a["document_name"],
                    "document_b": b["document_name"],
                    "physical_page_a": a["physical_page"],
                    "physical_page_b": b["physical_page"],
                    "page_label_a": a["page_label"],
                    "page_label_b": b["page_label"],
                    "shared_characters": len(normalize(chroma_by_id[left]["text"])),
                    "similarity": 1.0,
                    "manual_assessment": "Exact normalized text group requires review.",
                    "recommended_action": "manual_review_before_any_removal",
                }
            )

    # Expected overlap between adjacent chunks from the same page-level document.
    overlap_by_node = {}
    by_part = defaultdict(list)
    for row in node_rows:
        if row["document_name"] and row["part_index"] is not None:
            by_part[(row["document_name"], row["part_index"])].append(row)
    for (document, part), rows in sorted(by_part.items()):
        rows.sort(
            key=lambda row: (
                row["chunk_start_char"] if row["chunk_start_char"] is not None else -1,
                row["node_id"],
            )
        )
        for left, right in zip(rows, rows[1:]):
            left_text = normalize(chroma_by_id[left["node_id"]]["text"])
            right_text = normalize(chroma_by_id[right["node_id"]]["text"])
            actual_overlap = suffix_prefix_overlap(left_text, right_text)
            index_overlap = None
            if (
                left["chunk_end_char"] is not None
                and right["chunk_start_char"] is not None
            ):
                index_overlap = max(
                    0,
                    int(left["chunk_end_char"]) - int(right["chunk_start_char"]),
                )
            if actual_overlap > 0 or (index_overlap and index_overlap > 0):
                size = actual_overlap or int(index_overlap or 0)
                overlap_by_node[right["node_id"]] = {
                    "previous_node_id": left["node_id"],
                    "shared_characters": size,
                    "overlap_ratio": size / max(1, len(right_text)),
                    "index_overlap": index_overlap,
                }
                duplicate_rows.append(
                    {
                        "candidate_type": "chunk_overlap",
                        "classification": "expected_overlap",
                        "node_id_a": left["node_id"],
                        "node_id_b": right["node_id"],
                        "document_a": document,
                        "document_b": document,
                        "physical_page_a": left["physical_page"],
                        "physical_page_b": right["physical_page"],
                        "page_label_a": left["page_label"],
                        "page_label_b": right["page_label"],
                        "shared_characters": size,
                        "similarity": round(size / max(1, min(len(left_text), len(right_text))), 6),
                        "manual_assessment": (
                            "Adjacent chunks overlap within the same physical page; "
                            "consistent with configured chunk overlap and preserved for context."
                        ),
                        "recommended_action": "retain_as_expected_chunk_overlap",
                    }
                )

    # Near-duplicate candidates are deliberately reported as unverified only.
    ordered_rows = sorted(
        node_rows,
        key=lambda row: (
            row["document_name"] or "",
            row["physical_page"] or 0,
            row["chunk_start_char"] if row["chunk_start_char"] is not None else -1,
        ),
    )
    for index, left in enumerate(ordered_rows):
        left_text = normalize(chroma_by_id[left["node_id"]]["text"])
        if len(left_text) < 80:
            continue
        left_shingles = shingles(left_text)
        for right in ordered_rows[index + 1:]:
            if left["document_name"] != right["document_name"]:
                break
            if right["exact_duplicate_group"] and left["exact_duplicate_group"] == right["exact_duplicate_group"]:
                continue
            right_text = normalize(chroma_by_id[right["node_id"]]["text"])
            if len(right_text) < 80:
                continue
            ratio_rough = len(left_shingles.intersection(shingles(right_text))) / max(
                1, len(left_shingles.union(shingles(right_text)))
            )
            if ratio_rough < 0.35:
                continue
            ratio = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
            if ratio >= 0.82:
                left_heading = left_text.split(" ", 1)[0]
                right_heading = right_text.split(" ", 1)[0]
                language_headers = {
                    "English", "Deutsch", "Français", "Italiano", "Español",
                    "Nederlands", "Svenska", "Dansk", "Norsk", "Suomi",
                    "Português", "Ελληνικά", "Türkçe", "العربية",
                }
                translated_variant = (
                    left_heading in language_headers
                    and right_heading in language_headers
                    and left_heading != right_heading
                )
                duplicate_rows.append(
                    {
                        "candidate_type": "near_duplicate",
                        "classification": (
                            "translated_parallel_content"
                            if translated_variant
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
                        "similarity": round(ratio, 6),
                        "manual_assessment": (
                            "Parallel localized text, not an accidental duplicate."
                            if translated_variant
                            else "High text similarity but semantic duplication is unverified."
                        ),
                        "recommended_action": (
                            "retain_as_language_variant"
                            if translated_variant
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
                    "recommended_action": "manual_review",
                }
            )
        if row["table_fragment_suspected"]:
            compact = normalize(row["text"])
            if len(compact) < 10:
                classification = "sparse_extracted_text_confirmed"
                assessment = (
                    "PDF extraction produced fewer than 10 characters on this page; "
                    "not evidence of a duplicate."
                )
            else:
                classification = "short_content_or_layout_fragment"
                assessment = (
                    "Short node may represent a cover, heading, diagram label, or split "
                    "table/specification row. No duplicate or truncation is confirmed."
                )
            duplicate_rows.append(
                {
                    "candidate_type": "table_fragment",
                    "classification": classification,
                    "node_id_a": row["node_id"],
                    "node_id_b": "",
                    "document_a": row["document_name"],
                    "document_b": "",
                    "physical_page_a": row["physical_page"],
                    "physical_page_b": "",
                    "page_label_a": row["page_label"],
                    "page_label_b": "",
                    "shared_characters": None,
                    "similarity": None,
                    "manual_assessment": assessment,
                    "recommended_action": "manual_review_table_context",
                }
            )

    for row in node_rows:
        overlap = overlap_by_node.get(row["node_id"])
        row["overlap_previous_node_id"] = overlap["previous_node_id"] if overlap else ""
        row["overlap_with_previous_chars"] = overlap["shared_characters"] if overlap else 0
        row["overlap_with_previous_ratio"] = round(overlap["overlap_ratio"], 6) if overlap else 0
        row["overlap_classification"] = "expected_overlap" if overlap else ""

    source_rows = []
    for document, info in sorted(pdf_info.items()):
        doc_nodes = [row for row in node_rows if row["document_name"] == document]
        parts = {
            row["part_index"]
            for row in doc_nodes
            if row["part_index"] is not None
        }
        expected_parts = set(range(info["page_count"]))
        labels = sorted(
            {str(row["page_label"]) for row in doc_nodes},
            key=lambda value: int(value) if value.isdigit() else value,
        )
        source_rows.append(
            {
                **info,
                "chroma_node_count": len(doc_nodes),
                "docstore_node_count": sum(
                    row["node_id"] in docstore_by_id for row in doc_nodes
                ),
                "unique_part_indices": len(parts),
                "pages_with_nodes": " | ".join(str(value + 1) for value in sorted(parts)),
                "missing_physical_pages": " | ".join(
                    str(value + 1) for value in sorted(expected_parts - parts)
                ),
                "unique_page_labels": " | ".join(labels),
                "empty_text_nodes": sum(row["empty_text"] for row in doc_nodes),
                "abnormal_source_nodes": sum(
                    bool(row["abnormal_source_reasons"]) for row in doc_nodes
                ),
                "source_hash_matches_metadata": all(
                    row["abnormal_source_reasons"].find("file_hash_mismatch") < 0
                    for row in doc_nodes
                ),
            }
        )

    with (OUTPUT_DIR / "source_documents.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "document_name",
            "stored_filename",
            "file_path",
            "absolute_path",
            "size_bytes",
            "sha1",
            "sha256",
            "page_count",
            "encrypted",
            "extracted_characters",
            "chroma_node_count",
            "docstore_node_count",
            "unique_part_indices",
            "pages_with_nodes",
            "missing_physical_pages",
            "unique_page_labels",
            "empty_text_nodes",
            "abnormal_source_nodes",
            "source_hash_matches_metadata",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_rows)

    with (OUTPUT_DIR / "old_index_nodes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "node_id",
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
            "missing_metadata_fields",
            "abnormal_source_reasons",
            "exact_duplicate_group",
            "overlap_previous_node_id",
            "overlap_with_previous_chars",
            "overlap_with_previous_ratio",
            "overlap_classification",
            "empty_text",
            "table_fragment_suspected",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(node_rows)

    with (OUTPUT_DIR / "duplicate_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
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
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(duplicate_rows)

    report = {
        "documents": len(pdf_info),
        "pdf_pages": sum(info["page_count"] for info in pdf_info.values()),
        "chroma_nodes": len(chroma_ids),
        "unique_chroma_ids": len(set(chroma_ids)),
        "docstore_nodes": len(docstore_by_id),
        "id_sets_equal": set(chroma_ids) == set(docstore_by_id),
        "text_equal_nodes": sum(
            row["chroma_docstore_text_equal"] for row in node_rows
        ),
        "metadata_exact_equal_nodes": sum(
            row["metadata_exact_equal"] for row in node_rows
        ),
        "metadata_normalized_equal_nodes": sum(
            row["metadata_normalized_equal"] for row in node_rows
        ),
        "empty_text_nodes": sum(row["empty_text"] for row in node_rows),
        "source_anomaly_nodes": sum(
            bool(row["abnormal_source_reasons"]) for row in node_rows
        ),
        "exact_duplicate_groups": len(exact_groups),
        "exact_cross_page_groups": sum(
            len(
                {
                    (node_row_by_id[node_id]["document_name"], node_row_by_id[node_id]["physical_page"])
                    for node_id in ids
                }
            )
            > 1
            for ids in exact_groups.values()
        ),
        "chunk_overlap_pairs": sum(
            row["candidate_type"] == "chunk_overlap" for row in duplicate_rows
        ),
        "near_duplicate_candidates": sum(
            row["candidate_type"] == "near_duplicate" for row in duplicate_rows
        ),
        "table_fragment_candidates": sum(
            row["candidate_type"] == "table_fragment" for row in duplicate_rows
        ),
        "embedding_missing": sum(
            embedding is None or len(embedding) == 0
            for embedding in raw["embeddings"]
        ),
        "embedding_dimensions": sorted(
            {len(embedding) for embedding in raw["embeddings"] if embedding is not None}
        ),
        "slice_config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "pdf_reader": "llama_index.readers.file.PDFReader",
            "pdf_extraction": "pypdf extract_text(), one Document per physical page",
            "table_handling": "none detected; no tabula/camelot/table reconstruction",
            "chunking": "SentenceSplitter.get_nodes_from_documents per page Document",
        },
    }
    (OUTPUT_DIR / "audit_stats.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
