import json

from api.ingestor import save_metadata
from api.searcher import Searcher, SimpleBM25, build_bm25


def _write_metadata(tmp_path, items):
    metadata_path = tmp_path / "metadata.json"
    save_metadata(items, metadata_path)
    return metadata_path


def test_searcher_with_no_metadata_and_no_index_returns_empty_results(tmp_path):
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=tmp_path / "missing.json")

    keyword_results, keyword_ms = searcher.keyword_search("anything")
    semantic_results, semantic_ms = searcher.semantic_search("anything")

    assert keyword_results == []
    assert semantic_results == []
    assert keyword_ms >= 0
    assert semantic_ms >= 0


def test_keyword_search_returns_matching_chunk(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"vector_id": 0, "doc_id": "a", "chunk_id": 0, "text": "neural networks train with gradients"},
                {"vector_id": 1, "doc_id": "b", "chunk_id": 0, "text": "payments and invoices need reconciliation"},
            ]
        ),
        encoding="utf-8",
    )

    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)
    results, elapsed = searcher.keyword_search("neural gradients")

    assert elapsed >= 0
    assert results[0]["doc_id"] == "a"


def test_keyword_search_ranks_by_relevance(tmp_path):
    metadata_path = _write_metadata(
        tmp_path,
        [
            {"vector_id": 0, "doc_id": "low", "chunk_id": 0, "text": "cats are nice"},
            {"vector_id": 1, "doc_id": "high", "chunk_id": 0, "text": "python python python programming"},
            {"vector_id": 2, "doc_id": "none", "chunk_id": 0, "text": "totally unrelated content"},
        ],
    )
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)

    results, _ = searcher.keyword_search("python", k=3)

    assert results[0]["doc_id"] == "high"


def test_keyword_search_respects_k_limit(tmp_path):
    items = [{"vector_id": i, "doc_id": f"doc{i}", "chunk_id": 0, "text": "shared keyword text"} for i in range(10)]
    metadata_path = _write_metadata(tmp_path, items)
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)

    results, _ = searcher.keyword_search("shared keyword", k=3)

    assert len(results) == 3


def test_keyword_search_is_case_insensitive(tmp_path):
    metadata_path = _write_metadata(
        tmp_path,
        [{"vector_id": 0, "doc_id": "a", "chunk_id": 0, "text": "GRADIENT Descent Optimizer"}],
    )
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)

    results, _ = searcher.keyword_search("gradient descent")

    assert len(results) == 1
    assert results[0]["doc_id"] == "a"


def test_search_mode_semantic_only_skips_keyword(tmp_path, monkeypatch):
    metadata_path = _write_metadata(
        tmp_path,
        [{"vector_id": 0, "doc_id": "a", "chunk_id": 0, "text": "some text"}],
    )
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)
    monkeypatch.setattr(searcher, "keyword_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    response = searcher.search("query", mode="semantic")

    assert response["keyword_results"] == []
    assert response["keyword_time_taken_ms"] is None


def test_search_mode_keyword_only_skips_semantic(tmp_path, monkeypatch):
    metadata_path = _write_metadata(
        tmp_path,
        [{"vector_id": 0, "doc_id": "a", "chunk_id": 0, "text": "python programming"}],
    )
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)
    monkeypatch.setattr(searcher, "semantic_search", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    response = searcher.search("python", mode="keyword")

    assert response["semantic_results"] == []
    assert response["semantic_time_taken_ms"] is None
    assert response["keyword_results"][0]["doc_id"] == "a"


def test_search_mode_both_runs_semantic_and_keyword_concurrently(tmp_path):
    metadata_path = _write_metadata(
        tmp_path,
        [{"vector_id": 0, "doc_id": "a", "chunk_id": 0, "text": "python programming"}],
    )
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)
    # No FAISS index on disk -> semantic_search short-circuits to empty without needing a model.
    response = searcher.search("python", mode="both")

    assert response["keyword_results"][0]["doc_id"] == "a"
    assert response["semantic_results"] == []
    assert response["keyword_time_taken_ms"] is not None
    assert response["semantic_time_taken_ms"] is not None


def test_refresh_picks_up_metadata_written_after_construction(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    save_metadata([], metadata_path)
    searcher = Searcher(index_path=tmp_path / "missing.index", metadata_path=metadata_path)

    assert searcher.keyword_search("anything")[0] == []

    save_metadata(
        [{"vector_id": 0, "doc_id": "new", "chunk_id": 0, "text": "freshly added document"}],
        metadata_path,
    )
    searcher.refresh()

    results, _ = searcher.keyword_search("freshly added")
    assert results[0]["doc_id"] == "new"


def test_build_bm25_falls_back_to_simple_bm25_when_rank_bm25_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    bm25 = build_bm25([["python", "code"], ["cats", "dogs"]])
    assert isinstance(bm25, SimpleBM25)


def test_simple_bm25_counts_term_frequency():
    bm25 = SimpleBM25([["python", "python", "code"], ["cats", "dogs"]])

    scores = bm25.get_scores(["python"])

    assert scores[0] == 2.0
    assert scores[1] == 0.0

