# Clean Index Audit

## Decision

| Item | Value |
|---|---|
| Verification status | `clean_index_verified` |
| `clean_index_verified` | `True` |
| Structural verification | `True` |
| Finding status | structural audit passed; 22/22 manual review items passed |
| Nodes | 240 |
| Physical pages covered | 181/181 |
| Chroma/DocStore ID sets equal | `True` |
| Body-equal nodes | 240/240 |
| Normalized metadata-equal nodes | 240/240 |
| Exact metadata-equal nodes | 240/240 |
| Empty-text nodes | 0 |
| Exact duplicate groups | 0 |
| Near-duplicate candidates | 1 |
| Expected overlap pairs | 46 |
| Short-content nodes | 15 |
| Missing embeddings | 0 |
| Embedding dimensions | `[1024]` |

The index satisfies all structural and manual-review conditions and is accepted as the `clean_index_verified` evaluation baseline. The review acceptance timestamp is `2026-11-12T16:53:14+00:00`; known extraction limitations remain documented below.

## Final acceptance

- Status: `clean_index_verified`
- Manual review: 22/22 passed
- Review comment: 用户已完成人工复核，确认无实质性问题。
- Review inventory: `D:\VScode\word\llamaIndex_rag\evaluation\index_audit\clean_index\combined_manual_review.md`
- Final report: `D:\VScode\word\llamaIndex_rag\evaluation\index_audit\clean_index\final_acceptance_report.md`
- Logical fingerprint: `4595dfd3be078f5a05749869fa2fae3e31c2a5ad6227807382f2b8e92d63221c`
- Fingerprint matches Manifest: `True`
- Chroma/DocStore IDs equal: `True`
- Body-equal nodes: 240/240
- Metadata-equal nodes: 240/240

## Hard-condition results

| Condition | Result |
|---|---|
| `five_pdfs_indexed` | PASS |
| `source_sha256_unchanged` | PASS |
| `all_physical_pages_have_nodes` | PASS |
| `chroma_docstore_ids_equal` | PASS |
| `node_bodies_equal` | PASS |
| `normalized_metadata_equal` | PASS |
| `no_empty_text_nodes` | PASS |
| `no_confirmed_accidental_duplicates` | PASS |
| `embedding_complete_and_finite` | PASS |
| `embedding_dimensions_consistent` | PASS |
| `storage_paths_isolated` | PASS |
| `source_metadata_valid` | PASS |

## Source coverage

| 文档 | 字节 | 物理页 | 节点 | 有节点页 | SHA-256 |
|---|---:|---:|---:|---:|---|
| OBD.pdf | 840,253 | 20 | 24 | 20/20 | `44eb8755776c1782c7a427b9ccbbb05c4f25acec3d330f8075e87a2d0e7b9d4e` |
| 云台.pdf | 5,387,984 | 22 | 29 | 22/22 | `67f60f7617ea1060d40bdbdf87919a4245d118c4be657a829c1807232d11c1ba` |
| 小米路由器.pdf | 1,789,076 | 1 | 12 | 1/1 | `77364d8b6a2ee268e3c1a35cec18bd239bb83e900f5a18163e4fde1526ed5cce` |
| 空气炸锅.pdf | 2,607,162 | 22 | 34 | 22/22 | `a508867ccd58cd7be0c5fb835796f86589d81f535717331048dd9ec646ce02e2` |
| 高清摄像头.pdf | 11,941,902 | 116 | 141 | 116/116 | `f1ecc7a14bd86afdbf1c1384bf06795528a08d44854af544320a8d2e17fea23e` |

Every source path is normalized as a project-relative POSIX path. The
`document_id` is the source page Document ID and maps to physical page `N` with
the stable form `<stored_filename>_part_<N-1>`. All numeric metadata values are
integers; dates and upload time are ISO strings. All five source SHA-256 values
match between build and audit.

## Chroma and DocStore comparison

- Chroma IDs: 240 total, 240 unique.
- DocStore IDs: 240.
- ID sets equal: `True`.
- Body text equal for every compared node: `True`.
- Required metadata normalized equality: `240/240`.
- Exact required metadata equality: `240/240`.
- Source/document/page-label validation errors: 0 nodes.
- Embedding presence and finite-value failures: 0.

## Duplicate and overlap findings

- No exact duplicate groups: `True`.
- No exact cross-page duplicated content: `True`.
- Adjacent overlap pairs: 46 retained as expected chunk overlap.
- Near-duplicate candidates: 1; all remain manual-review candidates unless their classification explicitly says translated/parallel candidate.
- Short-content nodes: 15 retained. They were not removed merely because of length.
- No table truncation was automatically confirmed. PDF tables are not structurally reconstructed by the actual ingestion path.

## C920 sparse-page review

- 高清摄像头.pdf physical page 109 (label `109`): `7` extracted chars, 1 node; visible text `109 USB`. Embedded image extracted from physical page 109 and visually reviewed: it is a small Skype logo; this explains the sparse text '109 USB'.
- 高清摄像头.pdf physical page 113 (label `113`): `3` extracted chars, 1 node; visible text `113`. Embedded image extracted from physical page 113 and visually reviewed: it is a 961x658 Logitech Webcam Software screenshot containing visible labels that text extraction does not capture; this is an image-only content limitation, not an empty node.

Both named sparse pages were checked at the embedded-image level. Page 109 is a
small Skype logo and page 113 is a UI screenshot. Image-only information is not
recoverable as searchable text by the configured PDF parser; this is a known
content-mode limitation, not a duplicate or empty-node defect.

## Manual review closure

All 22 review items are present and have `review_status=passed`:

- 7 nodes whose body changed because of verified rechunking boundaries.
- 15 short-content or layout-fragment nodes.

Review comment: 用户已完成人工复核，确认无实质性问题。

The full item-level status is recorded in `combined_manual_review.md`. The
previously listed short nodes were retained; no node text or metadata was
modified during review.

## Findings by evidence class

### Confirmed

- Five expected PDFs were parsed from original files; all 181 physical pages have at least one node.
- Chroma and DocStore have matching IDs, body text, and normalized required metadata.
- No empty-text nodes and no missing/non-finite embedding vectors.
- No exact duplicate node groups were found.
- The configured overlap pairs exist and are classified as expected overlap.
- C920 pages 109 and 113 are genuinely sparse/image-oriented; page 113 contains visible UI text inside an embedded image that `extract_text()` cannot expose.
- PDF tables are not structurally parsed.

### Manual review result

- The 7 content-changed/rechunked nodes were reviewed and accepted.
- The 15 short-content nodes were reviewed and accepted.
- No manual-review item was marked failed or unresolved.

### No problem found

- No unexplained source file, wrong document hash, wrong file size, duplicate ID, or body mismatch.
- No missing physical page in the clean index.
- No overlap classified as an accidental duplicate.

### Known limitations

- OCR and visual semantic completeness are not provided for image-only labels.
- PDF tables are processed as plain `extract_text()` content, without structural
  table reconstruction.
- C920 physical pages 109 and 113 remain sparse text extractions because their
  relevant content is image-oriented.
- Retrieval relevance, RAG answer quality, and no-answer rejection remain out of
  scope for this index-acceptance task.

These limitations are accepted and documented; they do not invalidate the
verified structural baseline.

## Baseline conclusion

The clean index is accepted and frozen as `clean_index_verified`. It contains
all 181 physical pages, 240 nodes, matching Chroma/DocStore IDs, matching body
text, equal required metadata, complete 1024-dimensional embeddings, and an
unchanged logical fingerprint. The 22 manually reviewed items all passed.
Image/table extraction limitations remain explicitly documented.

During diagnostic review, the old Chroma database was opened for one read-only
metadata inspection. Chroma housekeeping changed its physical SQLite file hash
from the previously recorded `579abb...b767` to `620040...793e`; no collection
write, delete, rebuild, or logical content operation was issued. The old
DocStore hash remained `e800a2...4c3e`. The clean build and audit use only the
isolated paths above, never the old storage path.

## Freeze recommendations

1. Treat the logical fingerprint in the Manifest as the frozen index identity.
2. Do not rebuild, mutate, or relabel this baseline during retrieval
   experiments; use a separate child experiment directory for future changes.
3. Preserve the documented image/table extraction limitations when comparing
   future index variants.
4. Keep old `storage/` and all historical results read-only.
