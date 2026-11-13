# Clean Index Build Report

## Build result

- Status: `clean_index_built_pending_audit`
- Audit result: `clean_index_structural_verified_pending_manual_review`
- Build started: `2026-11-12T16:35:48+00:00`
- Build finished: `2026-11-12T16:36:37.923959+00:00`
- Elapsed: `48.958` seconds
- Source PDFs: 5
- Physical pages: 181
- Nodes written: 240
- Embedding provider/model: `dashscope` / `text-embedding-v3`
- Embedding dimension: `[1024]`
- Chroma path: `D:\VScode\word\llamaIndex_rag\evaluation\index_audit\clean_index\storage\chroma`
- DocStore path: `D:\VScode\word\llamaIndex_rag\evaluation\index_audit\clean_index\storage\docstore`
- Old index path used: no

## Source documents

| 文档 | 字节 | 物理页 | 节点 | 有节点页 | SHA-256 |
|---|---:|---:|---:|---:|---|
| OBD.pdf | 840,253 | 20 | 24 | 20/20 | `44eb8755776c1782c7a427b9ccbbb05c4f25acec3d330f8075e87a2d0e7b9d4e` |
| 云台.pdf | 5,387,984 | 22 | 29 | 22/22 | `67f60f7617ea1060d40bdbdf87919a4245d118c4be657a829c1807232d11c1ba` |
| 小米路由器.pdf | 1,789,076 | 1 | 12 | 1/1 | `77364d8b6a2ee268e3c1a35cec18bd239bb83e900f5a18163e4fde1526ed5cce` |
| 空气炸锅.pdf | 2,607,162 | 22 | 34 | 22/22 | `a508867ccd58cd7be0c5fb835796f86589d81f535717331048dd9ec646ce02e2` |
| 高清摄像头.pdf | 11,941,902 | 116 | 141 | 116/116 | `f1ecc7a14bd86afdbf1c1384bf06795528a08d44854af544320a8d2e17fea23e` |

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
