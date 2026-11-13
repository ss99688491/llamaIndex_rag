"""Finalize the clean-index audit after the user's manual review.

This script performs no index writes and no model calls. It reloads Chroma and
DocStore, verifies their logical state, recomputes the logical fingerprint, and
only writes acceptance documentation when every required condition passes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core.storage.docstore import SimpleDocumentStore


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = OUTPUT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
DOCSTORE_DIR = STORAGE_DIR / "docstore"
COLLECTION_NAME = "knowledge_base"

MANIFEST_PATH = OUTPUT_DIR / "index_manifest.json"
AUDIT_PATH = OUTPUT_DIR / "clean_index_audit.md"
NODE_DIFF_PATH = OUTPUT_DIR / "comparison" / "node_diff.csv"
DUPLICATE_PATH = OUTPUT_DIR / "duplicate_candidates.csv"
COMPARISON_REPORT_PATH = (
    OUTPUT_DIR / "comparison" / "old_vs_clean_comparison.md"
)
COMBINED_REVIEW_PATH = OUTPUT_DIR / "combined_manual_review.md"
FINAL_REPORT_PATH = OUTPUT_DIR / "final_acceptance_report.md"

REVIEW_COMMENT = "用户已完成人工复核，确认无实质性问题。"
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(encoded)


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").replace("\u00a0", " ").split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_and_verify_index(manifest: dict[str, Any]) -> dict[str, Any]:
    collection = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    ).get_collection(COLLECTION_NAME)
    raw = collection.get(
        include=["metadatas", "documents", "embeddings"]
    )
    docstore = SimpleDocumentStore.from_persist_dir(
        persist_dir=str(DOCSTORE_DIR)
    )

    ids = list(raw["ids"])
    docstore_ids = set(docstore.docs)
    by_id = {
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

    body_equal = 0
    metadata_equal = 0
    metadata_missing = 0
    embedding_missing = 0
    dimensions: set[int] = set()
    empty_text = 0
    pages: set[tuple[str, int]] = set()
    fingerprint_nodes: list[dict[str, Any]] = []

    for node_id in sorted(ids):
        item = by_id[node_id]
        node = docstore.docs.get(node_id)
        if node is None:
            continue
        metadata = item["metadata"]
        node_metadata = dict(node.metadata or {})
        text = item["text"]
        if text == node.get_content():
            body_equal += 1
        required_ok = all(
            key in metadata
            and key in node_metadata
            and metadata[key] == node_metadata[key]
            for key in REQUIRED_METADATA
        )
        if required_ok:
            metadata_equal += 1
        else:
            metadata_missing += sum(
                key not in metadata or key not in node_metadata
                for key in REQUIRED_METADATA
            )
        embedding = item["embedding"]
        embedding_list = (
            [float(value) for value in embedding]
            if embedding is not None
            else []
        )
        if not embedding_list or not all(
            math.isfinite(value) for value in embedding_list
        ):
            embedding_missing += 1
        else:
            dimensions.add(len(embedding_list))
        if not normalize_text(text):
            empty_text += 1
        original_name = str(metadata.get("original_name") or "")
        source_page = int(metadata["source_page"])
        pages.add((original_name, source_page))
        fingerprint_nodes.append(
            {
                "node_id": node_id,
                "document_id": str(
                    node.ref_doc_id
                    or metadata.get("document_id")
                    or ""
                ),
                "text": normalize_text(text),
                "metadata": {
                    key: metadata.get(key)
                    for key in sorted(REQUIRED_METADATA)
                },
                "embedding": embedding_list,
            }
        )

    recomputed_fingerprint = stable_hash(fingerprint_nodes)
    expected_fingerprint = manifest["fingerprints"][
        "logical_index_fingerprint"
    ]
    checks = {
        "chroma_count": len(ids),
        "unique_chroma_ids": len(set(ids)),
        "docstore_count": len(docstore.docs),
        "id_sets_equal": set(ids) == docstore_ids,
        "body_equal_nodes": body_equal,
        "metadata_equal_nodes": metadata_equal,
        "metadata_missing_fields": metadata_missing,
        "empty_text_nodes": empty_text,
        "embedding_missing_or_nonfinite": embedding_missing,
        "embedding_dimensions": sorted(dimensions),
        "physical_pages_covered": len(pages),
        "documents_covered": len({document for document, _ in pages}),
        "expected_logical_fingerprint": expected_fingerprint,
        "recomputed_logical_fingerprint": recomputed_fingerprint,
        "logical_fingerprint_matches": (
            expected_fingerprint == recomputed_fingerprint
        ),
    }
    return checks


def build_review_items() -> list[dict[str, Any]]:
    node_diff = read_csv(NODE_DIFF_PATH)
    duplicates = read_csv(DUPLICATE_PATH)
    node_by_new = {
        row["new_node_id"]: row
        for row in node_diff
        if row.get("new_node_id")
    }

    changed_rows = [
        row for row in node_diff if row["status"] == "content_changed"
    ]
    short_ids = [
        row["node_id_a"]
        for row in duplicates
        if row["candidate_type"] == "short_content"
    ]
    changed_ids = {row["new_node_id"] for row in changed_rows}
    if len(changed_rows) != 7:
        raise RuntimeError(
            f"Expected 7 content-changed review items, got {len(changed_rows)}."
        )
    if len(short_ids) != 15:
        raise RuntimeError(
            f"Expected 15 short-content review items, got {len(short_ids)}."
        )
    if len(set(short_ids)) != 15:
        raise RuntimeError("Short-content review node IDs are not unique.")
    if changed_ids & set(short_ids):
        raise RuntimeError(
            "Content-changed and short-content review sets overlap."
        )

    items: list[dict[str, Any]] = []
    for row in sorted(
        changed_rows,
        key=lambda item: (
            item["document_name"],
            int(item["physical_page_new"]),
            int(item["new_chunk_start_char"]),
        ),
    ):
        items.append(
            {
                "review_type": "content_changed_after_rechunking",
                "document_name": row["document_name"],
                "physical_page": int(row["physical_page_new"]),
                "page_label": row["page_label_new"],
                "old_node_id": row["old_node_id"],
                "new_node_id": row["new_node_id"],
                "detail": (
                    f"old={row['old_chunk_start_char']}-"
                    f"{row['old_chunk_end_char']}; "
                    f"new={row['new_chunk_start_char']}-"
                    f"{row['new_chunk_end_char']}"
                ),
            }
        )

    for new_node_id in short_ids:
        row = node_by_new.get(new_node_id)
        if row is None:
            raise RuntimeError(
                f"Short-content node {new_node_id} has no comparison mapping."
            )
        items.append(
            {
                "review_type": "short_content_or_layout_fragment",
                "document_name": row["document_name"],
                "physical_page": int(row["physical_page_new"]),
                "page_label": row["page_label_new"],
                "old_node_id": row["old_node_id"],
                "new_node_id": new_node_id,
                "detail": row["new_text_preview"],
            }
        )

    if len(items) != 22:
        raise RuntimeError(f"Expected 22 combined review items, got {len(items)}.")
    if len({item["new_node_id"] for item in items}) != 22:
        raise RuntimeError("Combined review items are not unique.")
    for index, item in enumerate(items, start=1):
        item["review_id"] = f"MR-{index:02d}"
        item["review_status"] = "passed"
        item["review_comment"] = REVIEW_COMMENT
    return items


def write_combined_review(
    items: list[dict[str, Any]],
    accepted_at: str,
) -> None:
    rows = "\n".join(
        "| {review_id} | {review_type} | {document_name} | {physical_page} | "
        "{page_label} | `{old_node_id}` | `{new_node_id}` | {detail} | "
        "{review_status} | {review_comment} |".format(**item)
        for item in items
    )
    content = f"""# Combined Manual Review

- Review scope: 22 items
- Content-changed nodes: 7
- Short-content/layout-fragment nodes: 15
- Review status: `passed`
- Passed: 22
- Failed: 0
- Accepted at: `{accepted_at}`
- Review comment: {REVIEW_COMMENT}

The user completed the manual review and confirmed that the 7 content changes
and 15 short-content items have no substantive issues. These nodes are retained;
no index node was changed or deleted as part of this review.

| Review ID | Type | Document | Physical page | Page label | Old node | New node | Detail | Review status | Review comment |
|---|---|---|---:|---|---|---|---|---|---|
{rows}
"""
    COMBINED_REVIEW_PATH.write_text(content, encoding="utf-8")


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise RuntimeError(f"Expected audit text not found: {old[:80]!r}")
    return text.replace(old, new, 1)


def replace_section(
    text: str,
    start: str,
    end: str,
    replacement: str,
) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"Section start not found: {start!r}")
    end_index = text.find(end, start_index + len(start))
    if end_index < 0:
        raise RuntimeError(f"Section end not found after {start!r}")
    return text[:start_index] + replacement + text[end_index:]


def update_audit(
    items: list[dict[str, Any]],
    accepted_at: str,
    checks: dict[str, Any],
) -> None:
    text = AUDIT_PATH.read_text(encoding="utf-8")
    if "`clean_index_structural_verified_pending_manual_review`" in text:
        text = replace_once(
            text,
            "| Verification status | "
            "`clean_index_structural_verified_pending_manual_review` |",
            "| Verification status | `clean_index_verified` |",
        )
        text = replace_once(
            text,
            "| `clean_index_verified` | `False` |",
            "| `clean_index_verified` | `True` |",
        )
        text = replace_once(
            text,
            "| Finding status | structural audit passed, but "
            "short/fragment content remains pending manual review |",
            "| Finding status | structural audit passed; 22/22 manual "
            "review items passed |",
        )
        text = replace_once(
            text,
            "The index does **not** qualify as an unqualified "
            "`clean_index_verified` baseline\nwhile the short/fragment "
            "manual-review queue is non-empty. The storage and source\n"
            "reconstruction checks themselves are reported separately below.",
            "The index satisfies all structural and manual-review conditions "
            "and is accepted as the `clean_index_verified` evaluation baseline. "
            "The review acceptance timestamp is "
            f"`{accepted_at}`; known extraction limitations remain documented "
            "below.",
        )
        final_acceptance = f"""
## Final acceptance

- Status: `clean_index_verified`
- Manual review: {len(items)}/{len(items)} passed
- Review comment: {REVIEW_COMMENT}
- Review inventory: `{COMBINED_REVIEW_PATH}`
- Final report: `{FINAL_REPORT_PATH}`
- Logical fingerprint: `{checks['recomputed_logical_fingerprint']}`
- Fingerprint matches Manifest: `{checks['logical_fingerprint_matches']}`
- Chroma/DocStore IDs equal: `{checks['id_sets_equal']}`
- Body-equal nodes: {checks['body_equal_nodes']}/{checks['chroma_count']}
- Metadata-equal nodes: {checks['metadata_equal_nodes']}/{checks['chroma_count']}
"""
        marker = "\n## Hard-condition results"
        text = replace_once(text, marker, final_acceptance + marker)
    elif "## Final acceptance" not in text:
        raise RuntimeError(
            "Audit is neither in pending state nor already finalized."
        )

    pending_start = "## Pending manual review"
    findings_start = "## Findings by evidence class"
    if pending_start in text:
        text = replace_section(
            text,
            pending_start,
            findings_start,
            f"""## Manual review closure

All 22 review items are present and have `review_status=passed`:

- 7 nodes whose body changed because of verified rechunking boundaries.
- 15 short-content or layout-fragment nodes.

Review comment: {REVIEW_COMMENT}

The full item-level status is recorded in `combined_manual_review.md`. The
previously listed short nodes were retained; no node text or metadata was
modified during review.

""",
        )

    suspected_start = "### Suspected or pending manual review"
    no_problem_start = "### No problem found"
    if suspected_start in text:
        text = replace_section(
            text,
            suspected_start,
            no_problem_start,
            """### Manual review result

- The 7 content-changed/rechunked nodes were reviewed and accepted.
- The 15 short-content nodes were reviewed and accepted.
- No manual-review item was marked failed or unresolved.

""",
        )

    limitations_start = "### Not verified"
    baseline_start = "## Baseline conclusion"
    if limitations_start in text:
        text = replace_section(
            text,
            limitations_start,
            baseline_start,
            """### Known limitations

- OCR and visual semantic completeness are not provided for image-only labels.
- PDF tables are processed as plain `extract_text()` content, without structural
  table reconstruction.
- C920 physical pages 109 and 113 remain sparse text extractions because their
  relevant content is image-oriented.
- Retrieval relevance, RAG answer quality, and no-answer rejection remain out of
  scope for this index-acceptance task.

These limitations are accepted and documented; they do not invalidate the
verified structural baseline.

""",
        )

    if baseline_start in text:
        caveat_marker = "\nDuring diagnostic review"
        baseline_caveat_index = text.find(caveat_marker, text.find(baseline_start))
        if baseline_caveat_index < 0:
            raise RuntimeError("Baseline diagnostic caveat marker not found.")
        text = (
            text[: text.find(baseline_start)]
            + f"""## Baseline conclusion

The clean index is accepted and frozen as `clean_index_verified`. It contains
all 181 physical pages, 240 nodes, matching Chroma/DocStore IDs, matching body
text, equal required metadata, complete 1024-dimensional embeddings, and an
unchanged logical fingerprint. The 22 manually reviewed items all passed.
Image/table extraction limitations remain explicitly documented.
"""
            + text[baseline_caveat_index:]
        )

    recommendations_start = "## Corrective recommendations"
    if recommendations_start in text:
        recommendations_index = text.find(recommendations_start)
        text = (
            text[:recommendations_index]
            + """## Freeze recommendations

1. Treat the logical fingerprint in the Manifest as the frozen index identity.
2. Do not rebuild, mutate, or relabel this baseline during retrieval
   experiments; use a separate child experiment directory for future changes.
3. Preserve the documented image/table extraction limitations when comparing
   future index variants.
4. Keep old `storage/` and all historical results read-only.
"""
        )
    AUDIT_PATH.write_text(text, encoding="utf-8")


def update_manifest(
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
    checks: dict[str, Any],
    accepted_at: str,
) -> dict[str, Any]:
    manifest["status"] = "clean_index_verified"
    manifest["verification_status"] = "clean_index_verified"
    manifest["clean_index_verified"] = True
    manifest["final_acceptance_at"] = accepted_at
    manifest["pending_manual_review_count"] = 0
    manifest["hard_conditions"]["manual_review_passed"] = True
    manifest["manual_review"] = {
        "completed": True,
        "review_status": "passed",
        "review_comment": REVIEW_COMMENT,
        "completed_at": accepted_at,
        "item_count": len(items),
        "passed_count": sum(
            item["review_status"] == "passed" for item in items
        ),
        "failed_count": sum(
            item["review_status"] != "passed" for item in items
        ),
        "content_changed_nodes": sum(
            item["review_type"] == "content_changed_after_rechunking"
            for item in items
        ),
        "short_content_nodes": sum(
            item["review_type"] == "short_content_or_layout_fragment"
            for item in items
        ),
        "review_item_ids_sha256": stable_hash(
            [item["review_id"] for item in items]
        ),
        "report_path": str(COMBINED_REVIEW_PATH),
    }
    manifest["fingerprint_verification"] = {
        "expected": checks["expected_logical_fingerprint"],
        "recomputed": checks["recomputed_logical_fingerprint"],
        "matches": checks["logical_fingerprint_matches"],
        "verified_at": accepted_at,
    }
    manifest["final_storage_verification"] = {
        "chroma_node_count": checks["chroma_count"],
        "docstore_node_count": checks["docstore_count"],
        "unique_chroma_ids": checks["unique_chroma_ids"],
        "id_sets_equal": checks["id_sets_equal"],
        "body_equal_nodes": checks["body_equal_nodes"],
        "metadata_equal_nodes": checks["metadata_equal_nodes"],
        "empty_text_nodes": checks["empty_text_nodes"],
        "missing_or_nonfinite_embeddings": checks[
            "embedding_missing_or_nonfinite"
        ],
        "embedding_dimensions": checks["embedding_dimensions"],
        "documents_covered": checks["documents_covered"],
        "physical_pages_covered": checks["physical_pages_covered"],
        "node_data_modified_during_acceptance": False,
    }
    manifest["comparison"] = {
        "report_path": str(COMPARISON_REPORT_PATH),
        "old_and_clean_node_counts_equal": True,
        "deterministically_paired_nodes": 240,
        "body_equal_paired_nodes": 233,
        "content_changed_paired_nodes": 7,
        "new_nodes": 0,
        "missing_nodes": 0,
        "manual_review_result": "7 content-changed nodes accepted",
    }
    manifest["known_limitations"] = [
        (
            "PDF image-only labels are not converted to searchable text; "
            "C920 physical pages 109 and 113 remain sparse."
        ),
        (
            "PDF tables are extracted as plain text and are not structurally "
            "reconstructed."
        ),
        (
            "OCR and visual semantic completeness are outside the accepted "
            "text-index boundary."
        ),
    ]
    manifest["final_acceptance"] = {
        "status": "clean_index_verified",
        "frozen": True,
        "report_path": str(FINAL_REPORT_PATH),
        "manual_review_report_path": str(COMBINED_REVIEW_PATH),
        "comparison_report_path": str(COMPARISON_REPORT_PATH),
        "accepted_at": accepted_at,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def write_final_report(
    manifest: dict[str, Any],
    items: list[dict[str, Any]],
    checks: dict[str, Any],
    accepted_at: str,
) -> None:
    report = f"""# Clean Index Final Acceptance Report

## Final decision

| Item | Result |
|---|---|
| Status | `clean_index_verified` |
| Frozen | `True` |
| Manual review | {len(items)}/{len(items)} passed |
| Failed review items | 0 |
| Chroma / DocStore nodes | {checks['chroma_count']} / {checks['docstore_count']} |
| Physical pages covered | {checks['physical_pages_covered']} |
| Documents covered | {checks['documents_covered']} |
| Logical fingerprint | `{checks['recomputed_logical_fingerprint']}` |
| Fingerprint matches Manifest | `{checks['logical_fingerprint_matches']}` |
| Accepted at | `{accepted_at}` |

## Structural audit

- Five original PDFs are represented.
- All 181 physical pages have at least one node.
- Chroma and DocStore contain 240 nodes each with identical ID sets.
- Body text is identical for 240/240 nodes.
- Required metadata is identical for 240/240 nodes.
- There are no empty-text nodes or missing/non-finite embeddings.
- All embeddings have dimension 1024.
- No exact duplicate node groups or missing physical pages were found.
- All 46 detected adjacent overlap pairs are retained as expected chunk overlap.

## Old-versus-clean comparison

- Both indexes contain 240 nodes and cover the same physical pages.
- The node_id intersection is 0; IDs are not reusable across the two builds.
- All 240 nodes were paired using deterministic source-page/chunk structure.
- 233 paired bodies are byte-for-byte identical.
- 7 paired bodies changed because metadata-aware slicing moved the chunk
  boundaries; the source PDF, physical page, and page label did not change.
- There are 0 new nodes, 0 missing nodes, and 0 unmatched nodes.
- Filename, normalized document_id, physical page, and page label match for all
  240 pairs.

The detailed comparison is recorded in:

`{COMPARISON_REPORT_PATH}`

## Manual review

- Content-changed nodes reviewed: 7
- Short-content/layout-fragment nodes reviewed: 15
- Total unique review items: {len(items)}
- `review_status=passed`: {sum(item['review_status'] == 'passed' for item in items)}
- `review_status=failed`: {sum(item['review_status'] != 'passed' for item in items)}
- Review comment: {REVIEW_COMMENT}

The complete 22-row review inventory is recorded in:

`{COMBINED_REVIEW_PATH}`

## Fingerprint verification

- Expected fingerprint: `{checks['expected_logical_fingerprint']}`
- Recomputed fingerprint: `{checks['recomputed_logical_fingerprint']}`
- Match: `{checks['logical_fingerprint_matches']}`

The fingerprint was recomputed from the live isolated Chroma/DocStore snapshot
before the Manifest was updated. No index node was written, deleted, or modified.

## Known limitations

- C920 physical pages 109 and 113 contain image-oriented content; text extraction
  remains sparse and image labels are not searchable through the text index.
- PDF tables use plain `extract_text()` content; no table structure is rebuilt.
- OCR and visual semantic completeness are outside this text-index boundary.
- Retrieval relevance and answer quality are not part of this acceptance task.

These limitations were considered during manual acceptance and do not block the
`clean_index_verified` structural baseline status.

## Execution controls

- No index rebuild was performed.
- No Chroma or DocStore node was modified.
- No original PDF was modified.
- No Embedding, Reranker, or LLM API was called.
- No retrieval Benchmark, RAG evaluation, or no-answer test was executed.
- The old index and historical experiment results were not modified.
- No Git commit or push was performed.

## Freeze declaration

The index stored at:

`{manifest['storage']['isolated_root']}`

is frozen as the accepted baseline identified by logical fingerprint:

`{checks['recomputed_logical_fingerprint']}`

Future retrieval experiments should reference this state read-only and use a
separate experiment directory for any later index variant.

Final Manifest:

`{MANIFEST_PATH}`
"""
    FINAL_REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    items = build_review_items()
    checks = load_and_verify_index(manifest)

    hard_conditions_pass = all(manifest["hard_conditions"].values()) and all(
        [
            checks["chroma_count"] == manifest["node_count"],
            checks["docstore_count"] == manifest["node_count"],
            checks["unique_chroma_ids"] == manifest["node_count"],
            checks["id_sets_equal"],
            checks["body_equal_nodes"] == manifest["node_count"],
            checks["metadata_equal_nodes"] == manifest["node_count"],
            checks["empty_text_nodes"] == 0,
            checks["embedding_missing_or_nonfinite"] == 0,
            len(checks["embedding_dimensions"]) == 1,
            checks["embedding_dimensions"] == [
                manifest["embedding_dimension"]
            ],
            checks["documents_covered"] == 5,
            checks["physical_pages_covered"] == 181,
            checks["logical_fingerprint_matches"],
        ]
    )
    if not hard_conditions_pass:
        raise RuntimeError(
            "Final acceptance blocked: a structural, manual-review, or "
            f"fingerprint condition failed: {json.dumps(checks, ensure_ascii=False)}"
        )

    accepted_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    write_combined_review(items, accepted_at)
    update_audit(items, accepted_at, checks)
    updated_manifest = update_manifest(
        manifest,
        items,
        checks,
        accepted_at,
    )
    write_final_report(
        updated_manifest,
        items,
        checks,
        accepted_at,
    )

    print(
        json.dumps(
            {
                "status": updated_manifest["status"],
                "clean_index_verified": updated_manifest[
                    "clean_index_verified"
                ],
                "review_items": len(items),
                "review_passed": sum(
                    item["review_status"] == "passed" for item in items
                ),
                "fingerprint": checks[
                    "recomputed_logical_fingerprint"
                ],
                "fingerprint_matches": checks[
                    "logical_fingerprint_matches"
                ],
                "chroma_nodes": checks["chroma_count"],
                "docstore_nodes": checks["docstore_count"],
                "body_equal_nodes": checks["body_equal_nodes"],
                "metadata_equal_nodes": checks["metadata_equal_nodes"],
                "embedding_dimensions": checks["embedding_dimensions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
