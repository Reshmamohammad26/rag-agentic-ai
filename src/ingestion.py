"""PDF -> pages -> chunks -> OpenAI embeddings -> Pinecone.

Run from the project root with:   python -m src.ingestion
(Use `-m` so the `src` package imports resolve; `python src/ingestion.py` will not work.)

Idempotent: every chunk gets a deterministic ID (hash of source + page + index + text),
so re-running overwrites the same vectors instead of duplicating them, and chunks that
already exist in the index are not re-embedded (saves API cost).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
import urllib.request
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)
BATCH_SIZE = 100


def download_pdf(url: str, dest: str) -> Path:
    """Download the PDF once; reuse the local copy afterwards."""
    path = Path(dest)
    if path.exists() and path.stat().st_size > 0:
        logger.info("Using existing PDF at %s", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as resp, open(path, "wb") as out:
        out.write(resp.read())
    return path


def load_pages(pdf_path: Path) -> list[Document]:
    """One Document per PDF page (pypdf). Page numbers are 1-based."""
    reader = PdfReader(str(pdf_path))
    pages: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # skip image-only / blank pages
        pages.append(Document(page_content=text, metadata={"source": pdf_path.name, "page": number}))
    logger.info("Loaded %d non-empty pages", len(pages))
    return pages


def split_pages(pages: list[Document], settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = splitter.split_documents(pages)
    # index each chunk within its page so IDs are stable and unique
    counters: dict[int, int] = {}
    for chunk in chunks:
        page = chunk.metadata["page"]
        chunk.metadata["chunk_index"] = counters.get(page, 0)
        counters[page] = counters.get(page, 0) + 1
    logger.info("Created %d chunks", len(chunks))
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic ID -> re-running ingestion never duplicates vectors."""
    raw = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.metadata['chunk_index']}|{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def get_or_create_index(pc: Pinecone, settings: Settings):
    """Create a serverless cosine index if missing; validate it if it exists."""
    name = settings.pinecone_index_name
    if not pc.has_index(name):
        logger.info("Creating Pinecone index '%s'", name)
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    else:
        desc = pc.describe_index(name)
        if int(desc.dimension) != settings.embedding_dimension:
            raise RuntimeError(
                f"Index '{name}' has dimension {desc.dimension}, but "
                f"{settings.embedding_model} produces {settings.embedding_dimension}. "
                "Use a different PINECONE_INDEX_NAME or delete the index."
            )
        if desc.metric != "cosine":
            raise RuntimeError(f"Index '{name}' uses metric '{desc.metric}', expected 'cosine'.")
    return pc.Index(name)


def ingest(reset: bool = False) -> int:
    settings = get_settings()
    pdf_path = download_pdf(settings.pdf_url, settings.pdf_path)
    chunks = split_pages(load_pages(pdf_path), settings)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = get_or_create_index(pc, settings)
    if reset:
        logger.warning("--reset: deleting all vectors in '%s'", settings.pinecone_index_name)
        try:
            index.delete(delete_all=True)
        except Exception as exc:  # noqa: BLE001 - empty namespace can raise 404
            logger.info("Nothing to delete (%s)", exc)

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )

    written = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        ids = [chunk_id(c) for c in batch]
        existing = set(index.fetch(ids=ids).vectors.keys())
        todo = [(i, c) for i, c in zip(ids, batch) if i not in existing]
        if not todo:
            continue
        vectors = embeddings.embed_documents([c.page_content for _, c in todo])
        index.upsert(
            vectors=[
                {
                    "id": vid,
                    "values": vec,
                    "metadata": {
                        "source": c.metadata["source"],
                        "page": c.metadata["page"],
                        "chunk_index": c.metadata["chunk_index"],
                        "text": c.page_content,
                    },
                }
                for (vid, c), vec in zip(todo, vectors)
            ]
        )
        written += len(todo)
        logger.info("Upserted %d/%d chunks", min(start + BATCH_SIZE, len(chunks)), len(chunks))

    logger.info("Done. %d new vectors written, %d already present.", written, len(chunks) - written)
    return written


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest the Agentic AI PDF into Pinecone.")
    parser.add_argument("--reset", action="store_true", help="Delete all existing vectors first.")
    ingest(reset=parser.parse_args().reset)


if __name__ == "__main__":
    main()
