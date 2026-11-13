"""Build an isolated clean RAG index from the five original PDFs.

The script intentionally does not import or call the application's ingestion
entry point. That entry point has a known single-file DocStore persistence
defect. This script performs the verified current parsing, splitting, embedding
and persistence steps while writing only to clean_index/storage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader

from src.config import Settings as AppSettings
from src.settings import _build_embedding


OUTPUT_DIR = Path(__file__).resolve().parent
STORAGE_DIR = OUTPUT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
DOCSTORE_DIR = STORAGE_DIR / "docstore"
BUILD_STATE_PATH = OUTPUT_DIR / "build_state.json"
FAILURE_PATH = OUTPUT_DIR / "build_failure.json"
INPUT_DIR = ROOT / "data" / "uploads"
COLLECTION_NAME = "knowledge_base"
EMBED_BATCH_SIZE = 10

EXPECTED_SOURCES = (
    {
        "original_name": "OBD.pdf",
        "stored_filename": "20261112_221703_OBD.pdf",
        "size_bytes": 840_253,
        "page_count": 20,
        "sha256": "44eb8755776c1782c7a427b9ccbbb05c4f25acec3d330f8075e87a2d0e7b9d4e",
    },
    {
        "original_name": "云台.pdf",
        "stored_filename": "20261112_221703_云台.pdf",
        "size_bytes": 5_387_984,
        "page_count": 22,
        "sha256": "67f60f7617ea1060d40bdbdf87919a4245d118c4be657a829c1807232d11c1ba",
    },
    {
        "original_name": "小米路由器.pdf",
        "stored_filename": "20261112_221703_小米路由器.pdf",
        "size_bytes": 1_789_076,
        "page_count": 1,
        "sha256": "77364d8b6a2ee268e3c1a35cec18bd239bb83e900f5a18163e4fde1526ed5cce",
    },
    {
        "original_name": "空气炸锅.pdf",
        "stored_filename": "20261112_221703_空气炸锅.pdf",
        "size_bytes": 2_607_162,
        "page_count": 22,
        "sha256": "a508867ccd58cd7be0c5fb835796f86589d81f535717331048dd9ec646ce02e2",
    },
    {
        "original_name": "高清摄像头.pdf",
        "stored_filename": "20261112_221703_高清摄像头.pdf",
        "size_bytes": 11_941_902,
        "page_count": 116,
        "sha256": "f1ecc7a14bd86afdbf1c1384bf06795528a08d44854af544320a8d2e17fea23e",
    },
)

EXCLUDED_METADATA_KEYS = [
    "file_name",
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
]

logger = logging.getLogger("clean-index-build")


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def date_from_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, timezone.utc).date().isoformat()


def iso_from_epoch(value: float) -> str:
    return (
        datetime.fromtimestamp(value, timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def sanitize_error(exc: BaseException, secret: str) -> str:
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return message


def ensure_isolated_paths() -> None:
    old_storage = (ROOT / "storage").resolve()
    new_storage = STORAGE_DIR.resolve()
    if new_storage == old_storage:
        raise RuntimeError("The clean index storage resolves to the old storage path.")
    if old_storage in new_storage.parents or new_storage in old_storage.parents:
        raise RuntimeError("Old and new index storage paths overlap.")
    if STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty clean index storage: {STORAGE_DIR}"
        )


def source_inventory(build_time: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for expected in EXPECTED_SOURCES:
        path = INPUT_DIR / expected["stored_filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source PDF: {path}")
        size = path.stat().st_size
        digest = hash_file(path, "sha256")
        sha1 = hash_file(path, "sha1")
        reader = PdfReader(str(path))
        encrypted = reader.is_encrypted
        if encrypted:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                raise RuntimeError(f"Could not decrypt source PDF with empty password: {path}")
        page_count = len(reader.pages)
        page_labels = list(reader.page_labels)
        if size != expected["size_bytes"]:
            raise RuntimeError(
                f"Source size mismatch for {expected['original_name']}: "
                f"{size} != {expected['size_bytes']}"
            )
        if digest != expected["sha256"]:
            raise RuntimeError(
                f"Source SHA-256 mismatch for {expected['original_name']}: "
                f"{digest} != {expected['sha256']}"
            )
        if page_count != expected["page_count"]:
            raise RuntimeError(
                f"Source page-count mismatch for {expected['original_name']}: "
                f"{page_count} != {expected['page_count']}"
            )
        stat = path.stat()
        rows.append(
            {
                **expected,
                "path": path,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha1": sha1,
                "encrypted": encrypted,
                "page_labels": page_labels,
                "creation_date": date_from_epoch(
                    getattr(stat, "st_birthtime", stat.st_ctime)
                ),
                "last_modified_date": date_from_epoch(stat.st_mtime),
                "uploaded_at": build_time,
            }
        )
    return rows


def parse_and_split(
    source: dict[str, Any],
    parser: SentenceSplitter,
) -> tuple[list[Any], list[dict[str, Any]]]:
    path: Path = source["path"]
    documents = PDFReader(return_full_document=False).load_data(file=path)
    if len(documents) != source["page_count"]:
        raise RuntimeError(
            f"PDFReader returned {len(documents)} documents for {path.name}; "
            f"expected {source['page_count']}."
        )

    nodes: list[Any] = []
    page_rows: list[dict[str, Any]] = []
    for page_index, document in enumerate(documents):
        physical_page = page_index + 1
        page_label = str(source["page_labels"][page_index])
        document.id_ = f"{source['stored_filename']}_part_{page_index}"
        document.metadata = {
            "page_label": page_label,
            "file_name": source["stored_filename"],
            "file_path": source["relative_path"],
            "file_type": "application/pdf",
            "file_size": int(source["size_bytes"]),
            "creation_date": source["creation_date"],
            "last_modified_date": source["last_modified_date"],
            "file_hash": source["sha1"],
            "file_sha256": source["sha256"],
            "original_name": source["original_name"],
            "uploaded_at": source["uploaded_at"],
            "source_page": int(physical_page),
            "page_index": int(page_index),
        }
        document.excluded_embed_metadata_keys = list(EXCLUDED_METADATA_KEYS)
        document.excluded_llm_metadata_keys = list(EXCLUDED_METADATA_KEYS)

        page_nodes = parser.get_nodes_from_documents([document], show_progress=False)
        if not page_nodes:
            raise RuntimeError(
                f"SentenceSplitter produced no node for {path.name} page {physical_page}."
            )
        for node in page_nodes:
            if node.ref_doc_id != document.id_:
                raise RuntimeError(
                    f"Node source-document ID mismatch on {path.name} page {physical_page}."
                )
            node.metadata["source_page"] = int(physical_page)
            node.metadata["page_index"] = int(page_index)
            node.excluded_embed_metadata_keys = list(EXCLUDED_METADATA_KEYS)
            node.excluded_llm_metadata_keys = list(EXCLUDED_METADATA_KEYS)
        nodes.extend(page_nodes)
        page_rows.append(
            {
                "stored_filename": source["stored_filename"],
                "original_name": source["original_name"],
                "physical_page": physical_page,
                "page_label": page_label,
                "source_document_id": document.id_,
                "raw_text": document.text or "",
                "raw_text_normalized": normalize_text(document.text or ""),
                "raw_text_chars": len(normalize_text(document.text or "")),
                "node_ids": [node.node_id for node in page_nodes],
                "node_count": len(page_nodes),
            }
        )
    return nodes, page_rows


def build_fingerprints(nodes: list[Any]) -> dict[str, str]:
    ordered = sorted(nodes, key=lambda node: node.node_id)
    node_ids = [node.node_id for node in ordered]
    texts = [normalize_text(node.get_content()) for node in ordered]
    full = [
        {
            "node_id": node.node_id,
            "document_id": node.ref_doc_id,
            "text": normalize_text(node.get_content()),
            "metadata": dict(node.metadata),
            "embedding": list(node.embedding or []),
        }
        for node in ordered
    ]
    return {
        "logical_index_fingerprint": stable_hash(full),
        "node_id_fingerprint": stable_hash(node_ids),
        "document_text_fingerprint": stable_hash(texts),
    }


def write_failure(
    exc: BaseException,
    build_started: str,
    secret: str,
) -> None:
    failure = {
        "status": "clean_index_build_failed",
        "build_started": build_started,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error": sanitize_error(exc, secret),
        "storage_may_be_partial": STORAGE_DIR.exists(),
        "old_index_modified": False,
    }
    FAILURE_PATH.write_text(
        json.dumps(failure, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    started_clock = time.time()
    build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    secret = ""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ensure_isolated_paths()

        cfg = AppSettings()  # type: ignore[call-arg]
        secret = cfg.dashscope_api_key
        if cfg.embedding_provider.strip().lower() != "dashscope":
            raise RuntimeError(
                f"Unexpected embedding provider: {cfg.embedding_provider!r}"
            )
        if cfg.embedding_model != "text-embedding-v3":
            raise RuntimeError(
                f"Unexpected embedding model: {cfg.embedding_model!r}"
            )
        if (cfg.chunk_size, cfg.chunk_overlap) != (512, 64):
            raise RuntimeError(
                "Unexpected slicing configuration: "
                f"{cfg.chunk_size}/{cfg.chunk_overlap}"
            )
        if not secret.strip():
            raise RuntimeError("DashScope embedding credential is not configured.")

        sources = source_inventory(build_time)
        parser = SentenceSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        logger.info("Parsing and splitting %d source PDFs.", len(sources))
        all_nodes: list[Any] = []
        all_pages: list[dict[str, Any]] = []
        for source in sources:
            source_nodes, source_pages = parse_and_split(source, parser)
            logger.info(
                "Prepared %s: pages=%d nodes=%d",
                source["original_name"],
                len(source_pages),
                len(source_nodes),
            )
            all_nodes.extend(source_nodes)
            all_pages.extend(source_pages)
        if not all_nodes:
            raise RuntimeError("No nodes were produced from the source PDFs.")

        dimensions: set[int] = set()
        empty_text_nodes = 0
        for page in all_pages:
            if not page["raw_text_normalized"]:
                empty_text_nodes += 1
        if empty_text_nodes:
            raise RuntimeError(
                f"PDFReader returned empty text for {empty_text_nodes} physical pages."
            )

        logger.info(
            "Creating embedding model and isolated stores: nodes=%d",
            len(all_nodes),
        )
        embed_model = _build_embedding(cfg)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        document_store = SimpleDocumentStore()
        storage_context = StorageContext.from_defaults(
            docstore=document_store,
            vector_store=vector_store,
        )

        total = len(all_nodes)
        for start in range(0, total, EMBED_BATCH_SIZE):
            batch = all_nodes[start : start + EMBED_BATCH_SIZE]
            texts = [
                node.get_content(metadata_mode="all")
                for node in batch
            ]
            embeddings = embed_model.get_text_embedding_batch(
                texts,
                show_progress=False,
            )
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Embedding count mismatch in batch at node {start}: "
                    f"{len(embeddings)} != {len(batch)}"
                )
            for node, embedding in zip(batch, embeddings):
                if embedding is None or len(embedding) == 0:
                    raise RuntimeError(f"Missing embedding for node {node.node_id}.")
                if not all(math.isfinite(float(value)) for value in embedding):
                    raise RuntimeError(f"Non-finite embedding for node {node.node_id}.")
                dimensions.add(len(embedding))
                node.embedding = embedding
            vector_store.add(batch)
            document_store.add_documents(batch)
            logger.info(
                "Embedded and persisted %d/%d nodes.",
                min(start + EMBED_BATCH_SIZE, total),
                total,
            )

        if len(dimensions) != 1:
            raise RuntimeError(f"Inconsistent embedding dimensions: {sorted(dimensions)}")

        # Instantiate and persist the index metadata after manual batch insertion.
        VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=embed_model,
        )
        storage_context.persist(persist_dir=str(DOCSTORE_DIR))

        chroma_count = collection.count()
        if chroma_count != total:
            raise RuntimeError(
                f"Chroma count mismatch after insertion: {chroma_count} != {total}"
            )
        if len(document_store.docs) != total:
            raise RuntimeError(
                "DocStore count mismatch after insertion: "
                f"{len(document_store.docs)} != {total}"
            )

        post_sources = []
        for source in sources:
            post_sha256 = hash_file(source["path"], "sha256")
            if post_sha256 != source["sha256"]:
                raise RuntimeError(
                    f"Source PDF changed during build: {source['original_name']}"
                )
            post_sources.append(
                {
                    "original_name": source["original_name"],
                    "stored_filename": source["stored_filename"],
                    "size_bytes": source["size_bytes"],
                    "page_count": source["page_count"],
                    "sha256": post_sha256,
                }
            )

        fingerprints = build_fingerprints(all_nodes)
        source_stats = []
        for source in sources:
            doc_pages = [
                page
                for page in all_pages
                if page["stored_filename"] == source["stored_filename"]
            ]
            doc_nodes = [
                node
                for node in all_nodes
                if node.metadata.get("original_name") == source["original_name"]
            ]
            source_stats.append(
                {
                    key: value
                    for key, value in source.items()
                    if key
                    not in {
                        "path",
                        "page_labels",
                        "page_count",
                    }
                }
                | {
                    "page_count": int(source["page_count"]),
                    "node_count": len(doc_nodes),
                    "pages_with_nodes": len(
                        {
                            int(node.metadata["source_page"])
                            for node in doc_nodes
                        }
                    ),
                    "extracted_characters": sum(
                        page["raw_text_chars"] for page in doc_pages
                    ),
                }
            )

        state = {
            "status": "clean_index_built_pending_audit",
            "build_time": build_time,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - started_clock, 3),
            "project_root": str(ROOT),
            "pdf_input_dir": str(INPUT_DIR),
            "storage_dir": str(STORAGE_DIR),
            "chroma_dir": str(CHROMA_DIR),
            "docstore_dir": str(DOCSTORE_DIR),
            "collection_name": COLLECTION_NAME,
            "chunk_size": int(cfg.chunk_size),
            "chunk_overlap": int(cfg.chunk_overlap),
            "embedding_provider": cfg.embedding_provider,
            "embedding_model": cfg.embedding_model,
            "embedding_dimension": next(iter(dimensions)),
            "reader": "llama_index.readers.file.PDFReader",
            "pdf_extraction": "pypdf extract_text(); one Document per physical page",
            "chunking": "SentenceSplitter.get_nodes_from_documents per page Document",
            "table_handling": "none; no table reconstruction",
            "total_nodes": int(total),
            "sources": source_stats,
            "source_snapshots_after_build": post_sources,
            "physical_pages": [
                {
                    "stored_filename": page["stored_filename"],
                    "original_name": page["original_name"],
                    "physical_page": page["physical_page"],
                    "page_label": page["page_label"],
                    "source_document_id": page["source_document_id"],
                    "raw_text": page["raw_text"],
                    "raw_text_normalized": page["raw_text_normalized"],
                    "raw_text_chars": page["raw_text_chars"],
                    "node_ids": page["node_ids"],
                    "node_count": page["node_count"],
                }
                for page in all_pages
            ],
            "fingerprints": fingerprints,
        }
        BUILD_STATE_PATH.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Clean index built successfully: nodes=%d dimension=%d fingerprint=%s",
            total,
            next(iter(dimensions)),
            fingerprints["logical_index_fingerprint"],
        )
    except Exception as exc:
        write_failure(exc, build_time, secret)
        logger.error("Clean index build failed: %s", sanitize_error(exc, secret))
        raise


if __name__ == "__main__":
    main()
