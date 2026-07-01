import json

from api.searcher import Searcher


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

