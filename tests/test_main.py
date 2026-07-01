from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api import security
from api.searcher import Searcher


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build a TestClient against an app wired to an isolated, empty index.

    NOTE: api.main.upload_document() calls ingest_document() without passing
    index_path/metadata_path, and ingest_document's index_path/metadata_path
    defaults are bound to ingestor.INDEX_PATH/METADATA_PATH at *function
    definition time* -- so monkeypatching those module constants afterward
    has no effect on the already-defined default. To keep the Searcher
    instance and the ingestion target in sync during tests (as they are in
    production, where both resolve to "data/..."), we wrap ingest_document
    with explicit paths and patch main_module's reference to it.
    """
    import functools

    import api.ingestor as ingestor_module
    import api.main as main_module

    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.json"

    pinned_ingest = functools.partial(
        ingestor_module.ingest_document, index_path=index_path, metadata_path=metadata_path
    )
    monkeypatch.setattr(main_module, "ingest_document", pinned_ingest)

    empty_searcher = Searcher(index_path=index_path, metadata_path=metadata_path)
    monkeypatch.setattr(main_module, "searcher", empty_searcher)
    monkeypatch.setattr(main_module, "UPLOADS_DIR", tmp_path / "uploads")
    security._request_times.clear()

    return TestClient(main_module.app)


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_search_requires_query_param(client):
    response = client.get("/search")
    assert response.status_code == 400
    assert "Query is required" in response.json()["detail"]


def test_search_rejects_empty_query(client):
    response = client.get("/search", params={"q": "   "})
    assert response.status_code == 400


def test_search_rejects_invalid_mode(client):
    response = client.get("/search", params={"q": "hello", "mode": "nonsense"})
    assert response.status_code == 400
    assert "Invalid search mode" in response.json()["detail"]


def test_search_against_empty_index_returns_empty_results(client):
    response = client.get("/search", params={"q": "hello world", "mode": "keyword"})
    assert response.status_code == 200
    body = response.json()
    assert body["keyword_results"] == []
    assert body["semantic_results"] == []


def test_search_rate_limit_enforced(client):
    for _ in range(10):
        resp = client.get("/search", params={"q": "hello", "mode": "keyword"})
        assert resp.status_code == 200

    resp = client.get("/search", params={"q": "hello", "mode": "keyword"})
    assert resp.status_code == 429


def test_upload_rejects_disallowed_extension(client):
    response = client.post(
        "/upload",
        files={"file": ("malware.exe", b"fake binary content", "application/octet-stream")},
    )
    assert response.status_code == 400


def test_upload_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_indexes_text_file_and_makes_it_searchable(client, monkeypatch):
    # Avoid pulling in sentence-transformers/faiss for embeddings; stub with a
    # deterministic fake so we can assert on end-to-end wiring instead of ML output.
    import api.ingestor as ingestor_module

    class FakeModel:
        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingestor_module, "get_embedding_model", lambda: FakeModel())

    content = b"This is a sufficiently long useful sentence for chunking and search."
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["chunks_indexed"] >= 1
    assert "doc_id" in body

    search_response = client.get("/search", params={"q": "useful sentence", "mode": "keyword"})
    assert search_response.status_code == 200
    results = search_response.json()["keyword_results"]
    assert any("useful sentence" in r["text"] for r in results)


def test_upload_cleans_up_temp_file_after_indexing(client, monkeypatch, tmp_path):
    import api.ingestor as ingestor_module

    class FakeModel:
        def encode(self, texts):
            if isinstance(texts, str):
                texts = [texts]
            return [[1.0] for _ in texts]

    monkeypatch.setattr(ingestor_module, "get_embedding_model", lambda: FakeModel())

    content = b"Another useful sentence long enough to survive cleaning."
    client.post("/upload", files={"file": ("doc.txt", content, "text/plain")})

    import api.main as main_module

    leftover_files = list(main_module.UPLOADS_DIR.glob("*")) if main_module.UPLOADS_DIR.exists() else []
    assert leftover_files == []


def test_upload_returns_500_with_generic_message_on_unexpected_error(client, monkeypatch):
    import api.main as main_module

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(main_module, "ingest_document", boom)

    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"long enough valid content here", "text/plain")},
    )
    assert response.status_code == 500
    # Internal error details should not leak to the client.
    assert "disk on fire" not in response.text
