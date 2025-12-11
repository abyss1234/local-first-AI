import os
import asyncio
# from httpx import Client, ASGITransport
from httpx import AsyncClient, ASGITransport

# import os, sys
# sys.path.insert(0, "/app")
from rag import ingest_directory, retrieve
from main import app

DOCS_DIR = "/app/docs"

def test_ingest_writes_points():
    # assumes docs exist and services are up
    res = asyncio.run(ingest_directory(DOCS_DIR))
    assert res.files_processed >= 1
    assert res.points_upserted >= 1

def test_retrieve_returns_results():
    hits = asyncio.run(retrieve("What is RAG?", top_k=3))
    assert len(hits) >= 1
    assert (hits[0].payload or {}).get("file") is not None

def test_chat_rag_endpoint_200():
    async def run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post("/chat_rag", json={"message": "What is RAG?", "top_k": 3})
            assert r.status_code == 200
            data = r.json()
            assert "answer" in data
            assert "citations" in data
            assert isinstance(data["citations"], list)

    asyncio.run(run())
