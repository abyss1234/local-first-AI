import os
import re
import uuid
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import httpx
from qdrant_client import QdrantClient, models

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> List[Tuple[int, int, str]]:
    """
    Simple char-based chunking with overlap.
    Returns list of (start, end, chunk_text).
    """
    text = normalize_text(text)
    if not text:
        return []

    chunks = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


async def ollama_embed(texts: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Calls Ollama /api/embed. input supports string or string[].
    Returns list of vectors. :contentReference[oaicite:3]{index=3}
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {"model": model, "input": texts}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["embeddings"]


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    """
    Create collection if not exists. In Qdrant, a collection stores points(vectors+payload). :contentReference[oaicite:4]{index=4}
    """
    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


@dataclass
class IngestResult:
    points_upserted: int
    files_processed: int


async def ingest_directory(docs_dir: str, collection_name: str = COLLECTION) -> IngestResult:
    """
    Parse ./docs -> chunk -> embed -> upsert to Qdrant (deterministic ids).
    """
    client = qdrant_client()

    # Gather files
    exts = {".txt", ".md"}
    files = []
    for root, _, names in os.walk(docs_dir):
        for name in names:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    files.sort()

    if not files:
        return IngestResult(points_upserted=0, files_processed=0)

    # First embed to learn dimension
    probe_vec = (await ollama_embed(["dimension probe"]))[0]
    dim = len(probe_vec)
    ensure_collection(client, collection_name, dim)

    points: List[models.PointStruct] = []
    total = 0
    files_processed = 0

    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        rel = os.path.relpath(path, docs_dir).replace("\\", "/")
        chunks = chunk_text(text)
        if not chunks:
            continue

        files_processed += 1

        # Batch embed chunks
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_texts = [c[2] for c in batch]
            vecs = await ollama_embed(batch_texts)

            for (start, end, chunk), vec in zip(batch, vecs):
                chunk_idx = i + batch.index((start, end, chunk))
                # deterministic id so ingest is repeatable
                pid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{rel}::chunk::{chunk_idx}"))

                snippet = chunk[:240].replace("\n", " ")
                payload = {
                    "file": rel,
                    "chunk_id": chunk_idx,
                    "start": start,
                    "end": end,
                    "text": chunk,
                    "snippet": snippet,
                }
                points.append(models.PointStruct(id=pid, vector=vec, payload=payload))
                total += 1

    # Upsert
    if points:
        client.upsert(collection_name=collection_name, points=points)  # points are vectors + payload :contentReference[oaicite:5]{index=5}

    return IngestResult(points_upserted=total, files_processed=files_processed)


import time

async def retrieve_with_stats(query: str, top_k: int = 5, collection_name: str = COLLECTION):
    t0 = time.perf_counter()
    t_embed0 = time.perf_counter()
    qvec = (await ollama_embed([query]))[0]
    embed_ms = (time.perf_counter() - t_embed0) * 1000.0
    client = qdrant_client()

    t_q0 = time.perf_counter()
    hits = client.search(
        collection_name=collection_name,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,
    )
    qdrant_ms = (time.perf_counter() - t_q0) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0

    stats = {
        "top_k": top_k,
        "hits_raw": len(hits),
        "embed_ms": round(embed_ms, 2),
        "qdrant_ms": round(qdrant_ms, 2),
        "retrieve_total_ms": round(total_ms, 2),
    }
    return hits,stats

async def retrieve(query: str, top_k: int = 5, collection_name: str = COLLECTION):
    hits, _ = await retrieve_with_stats(query, top_k, collection_name)
    return hits


# async def retrieve(query: str, top_k: int = 5, collection_name: str = COLLECTION):
#     client = qdrant_client()
#     qvec = (await ollama_embed([query]))[0]

#     hits = client.search(
#         collection_name=collection_name,
#         query_vector=qvec,
#         limit=top_k,
#         with_payload=True,
#     )
#     return hits
