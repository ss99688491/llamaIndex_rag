# Old Index Integrity Audit

## Audit status

- Audit type: read-only structural audit
- Audit date: `2026-11-13`
- Collection: `knowledge_base`
- Chroma nodes: 240
- DocStore nodes: 240
- Unique Chroma node IDs: 240
- PDF documents: 5
- PDF physical pages: 181
- No Embedding, Reranker, or LLM API was called.
- No source, configuration, Chroma, DocStore, PDF, or historical result file was modified.

Supporting files:

- `old_index_nodes.csv`: all 240 nodes with source, page, metadata, text, hash and duplicate context.
- `source_documents.csv`: five PDFs with size, page count, SHA-1, SHA-256 and node coverage.
- `duplicate_candidates.csv`: duplicate and overlap candidates with classification and recommended action.

## Source documents

| Document | Bytes | Pages | Chroma nodes | DocStore nodes | Physical pages with nodes |
|---|---:|---:|---:|---:|---:|
| OBD.pdf | 840,253 | 20 | 24 | 24 | 20 |
| 云台.pdf | 5,387,984 | 22 | 29 | 29 | 22 |
| 小米路由器.pdf | 1,789,076 | 1 | 12 | 12 | 1 |
| 空气炸锅.pdf | 2,607,162 | 22 | 34 | 34 | 22 |
| 高清摄像头.pdf | 11,941,902 | 116 | 141 | 141 | 116 |

All five stored PDF files still exist in `data/uploads/`. Their metadata
`file_hash` values match the actual SHA-1 values, and their recorded sizes match
the current files. The five SHA-256 values are recorded in
`source_documents.csv`.

Document identity is based on all of:

- `original_name`
- stored `file_name`
- `file_path`
- `file_hash` (SHA-1)
- `file_size`

No unexpected document name or file hash was found.

## Node completeness and attribution

- Every PDFReader page Document has an indexed `doc_id` ending in `_part_N`.
- `N + 1` maps to the physical PDF page because `PDFReader` returns one
  `Document` per page before `SentenceSplitter` creates chunks.
- All physical pages `1..page_count` are represented by at least one node.
- `missing_physical_pages` is empty for all five documents.
- All 240 nodes contain non-empty text.
- All 240 nodes contain the expected source metadata fields.
- All 240 Chroma embeddings are present and have dimension 1024.

Source attribution is substantively correct for every node. There is one
confirmed path-format inconsistency:

- All 29 `云台.pdf` nodes store an absolute `file_path`.
- The other 211 nodes store a relative `file_path`.
- The absolute path points to the correct existing PDF, and the filename,
  SHA-1, size and page metadata are correct. This is a provenance consistency
  issue, not evidence that the nodes came from a different document.

OBD page-label note:

- OBD has 20 physical pages but only page labels `1..19`.
- pypdf assigns label `1` to both the cover and the printed page numbered 1.
- Physical-page coverage was therefore checked using the `_part_N` source
  document index, not page label alone. No physical page is missing.

## Chroma and DocStore comparison

- Chroma IDs and DocStore IDs are identical as sets: PASS.
- Body text is identical for 240/240 nodes.
- All canonical metadata fields exist on both sides for 240/240 nodes.
- Normalized metadata equality is 240/240.
- Exact metadata equality is 223/240.
- The 17 non-exact comparisons are only `uploaded_at` floating-point
  serialization differences between approximately three and seven decimal
  places. No semantic source, page, hash or file-size difference was found.
- Chroma-only values such as `_node_content`, `_node_type`, `doc_id`,
  `document_id` and `ref_doc_id` are expected storage-layer metadata.

## Duplicate and overlap audit

| Candidate type | Count | Assessment |
|---|---:|---|
| Exact normalized-text duplicate groups | 0 | No exact duplicate nodes |
| Exact same content on different pages | 0 | No cross-page copies |
| Empty text nodes | 0 | No empty nodes |
| Expected adjacent chunk overlaps | 45 | All within the same physical page |
| Near-duplicate text candidates | 1 | Swedish and Norwegian C920 sections; legitimate language variants |
| Short/layout or table-like fragments | 14 | Not confirmed duplicates or truncation |

The 45 overlap relationships have:

- minimum shared boundary: 2 characters
- median shared boundary: 51 characters
- maximum shared boundary: 254 characters
- cross-page overlap pairs: 0

These are consistent with the configured `chunk_overlap=64` token setting and
must be retained. They are not duplicate documents or duplicate retrieval
units.

The single near-duplicate is:

- `高清摄像头.pdf` physical pages 51 and 67
- Similarity: approximately `0.874`
- Content: Swedish and Norwegian descriptions of the C920 feature list
- Assessment: translated parallel content, not accidental duplication
- Action: retain

The 14 short-content candidates include title pages, language headers, diagram
labels, a table-of-contents fragment and specification fragments. Two C920
pages have very sparse extracted text:

- physical page 109: only `109 USB`
- physical page 113: only `113`

These are confirmed extraction-sparsity observations, not duplicate nodes.
Because PDF table reconstruction is not used, the remaining 12 short/layout
fragments need visual review before claiming table completeness. No table
truncation is confirmed.

## Slice and parser behavior

Actual current behavior from source and installed reader code:

- `CHUNK_SIZE=512`
- `CHUNK_OVERLAP=64`
- Reader: `llama_index.readers.file.PDFReader`
- PDF extraction: `pypdf` `extract_text()`
- Granularity: one LlamaIndex `Document` per physical PDF page
- `filename_as_id=True`, producing `_part_N` source document IDs
- Chunking: `SentenceSplitter.get_nodes_from_documents([doc])` separately for each page Document
- Metadata-aware chunk sizing: yes; metadata tokens reduce available body chunk size
- Table handling: none detected; no tabula/camelot/table reconstruction

## Findings by evidence class

### Confirmed issues

1. `云台.pdf` stores absolute `file_path` metadata while the other documents
   use relative paths.
2. Seventeen `uploaded_at` values differ only at floating-point serialization
   precision between Chroma and DocStore.
3. C920 physical pages 109 and 113 have extremely sparse extractable text.
4. The ingestion path does not structurally parse tables.

None of these findings indicates a missing or wrongly attributed document node.

### Suspected issues requiring visual review

- Twelve short nodes may be table continuations, diagram labels or layout
  artifacts.
- The two sparse C920 pages may contain information present only as graphics.
  Text-based retrieval cannot recover such information from the current index.

### No problem found

- No duplicate node IDs.
- No empty node text.
- No exact duplicate text groups.
- No exact cross-page duplicated content.
- No missing physical pages.
- No incorrect source document hash or file size.
- No Chroma/DocStore text mismatch.
- No substantive Chroma/DocStore metadata mismatch.
- No unexplained document.

### Not verifiable from this audit

- Visual completeness of image-heavy pages.
- Semantic completeness of tables after `extract_text()`.
- Whether every visible diagram label was extracted.
- Retrieval relevance or answer quality; those require the separately audited
  evaluation datasets and model-based experiments.

## Index snapshot hashes

These hashes document the audited state:

- `storage/chroma/chroma.sqlite3`:
  `579abb2287650a10b97f7afd09586d9b43b6144479b061a1b26545efa610b767`
  (physical SQLite hash at the end of the audit)
- `storage/docstore/docstore.json`:
  `e800a2bb317166e9f98c5a041efe38bd6980481b178d3b4f9dbbe4926edc4c3e`
- `storage/docstore/graph_store.json`:
  `8e0a77744010862225c69da83c585f4f8a42fd551b044ce530dbb1eb6e16742c`
- `storage/docstore/image__vector_store.json`:
  `d17ed74c1649a438e518a8dc56a7772913dfe1ea7a7605bce069c63872431455`
- `storage/docstore/index_store.json`:
  `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`

Stable logical index fingerprints:

- Node + text + canonical metadata + embedding fingerprint:
  `8bd083af46be5b352d16aede3d3725f04718a6a064000bbc16b3f3b58c76218d`
- Sorted node-ID fingerprint:
  `919d1dc525f6e9b8d3ce1ea0b4b87f6f8ec43f65dff4beebef48cd580c65f9d5`
- Sorted document-text fingerprint:
  `f4fa604b27f9c89589ce0c65c8244570d64c7cc8aad055fab92e3d9c8596fca8`

The physical `chroma.sqlite3` file hash can change when Chroma performs SQLite
checkpoint or housekeeping activity on open. Repeated read-only opens produced
different physical hashes while the node IDs, texts, canonical metadata and
logical index fingerprint remained identical. Use the logical fingerprints,
not the SQLite file hash alone, as the content identity.

## Baseline conclusion

The old index is **suitable as a clearly labeled historical comparison
baseline** for retrieval experiments because:

- all five PDFs are represented;
- all physical pages have source nodes;
- IDs, text and canonical metadata are consistent between Chroma and DocStore;
- no exact duplicates or empty nodes were found;
- query retrieval can target every document.

It should **not** be treated as a clean canonical production baseline without
qualification because:

- its DocStore was recovered after a prior ingestion-path defect;
- `云台.pdf` uses a different path convention;
- PDF tables and image-heavy content are not structurally preserved;
- two C920 pages have very sparse extracted text;
- 12 short/layout fragments remain visually unverified.

The correct label remains `old_index_unverified` until an explicit index
decision is made. Future clean-index experiments must use a separate result
directory and may compare against this audited historical snapshot.
