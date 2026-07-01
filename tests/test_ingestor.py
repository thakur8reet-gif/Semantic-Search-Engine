import json
from pathlib import Path

import pytest

from api.ingestor import (
    _split_sentences,
    chunk_text,
    clean_text,
    extract_text,
    ingest_document,
    load_metadata,
    save_metadata,
)


def test_clean_text_removes_short_lines_and_collapses_whitespace():
    text = "Header\nThis is a meaningful line with enough length.\n\nAnother useful line for indexing."

    assert clean_text(text) == "This is a meaningful line with enough length. Another useful line for indexing."


def test_chunk_text_uses_word_boundaries_and_overlap():
    chunks = chunk_text("alpha beta gamma delta epsilon zeta", chunk_size=22, overlap=10)

    assert chunks == ["alpha beta gamma delta", "delta epsilon zeta"]
    assert all(len(chunk) <= 22 for chunk in chunks)
    assert set(chunks[0].split()) & set(chunks[1].split())


def test_extract_text_reads_utf8_text_file(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text("A useful document body.", encoding="utf-8")

    assert extract_text(path, ".txt") == "A useful document body."


def test_extract_text_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "doc.docx"
    path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_text(path, ".docx")


def test_extract_text_ignores_undecodable_bytes(tmp_path: Path):
    path = tmp_path / "doc.md"
    path.write_bytes(b"valid text \xff\xfe more text")

    # errors="ignore" means this should not raise, just drop bad bytes
    result = extract_text(path, ".md")
    assert "valid text" in result


def test_clean_text_drops_all_short_lines_returns_empty_string():
    assert clean_text("short\nalso short\nno") == ""


def test_chunk_text_empty_string_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0)


def test_chunk_text_rejects_overlap_gte_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=10, overlap=10)


def test_chunk_text_single_word_longer_than_chunk_size_still_included():
    # A pathological case: one "word" (e.g. a long URL/hash) exceeds chunk_size.
    # The function should still emit it rather than looping forever or dropping it.
    long_word = "a" * 50
    chunks = chunk_text(f"short {long_word} short", chunk_size=10, overlap=2)
    assert any(long_word in c for c in chunks)


def test_chunk_text_terminates_for_dense_short_text():
    # Regression guard against infinite loops in the sliding-window logic.
    text = " ".join(["word"] * 200)
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) > 0
    assert len(chunks) < 200  # sanity: shouldn't produce one chunk per word


def test_chunk_text_splits_on_sentence_boundaries_not_mid_sentence():
    # This is the behavior fix: chunks should start/end at sentence
    # boundaries where possible, instead of at an arbitrary word-count cutoff
    # that can land mid-sentence.
    text = (
        "Alpha sentence is here for testing purposes today. "
        "Beta sentence follows right after that one. "
        "Gamma sentence wraps up this short paragraph nicely."
    )
    chunks = chunk_text(text, chunk_size=60, overlap=10)

    for chunk in chunks:
        # Every chunk should start with a capital letter following a
        # sentence, not a lowercase word fragment from mid-sentence.
        assert chunk[0].isupper(), f"Chunk starts mid-sentence: {chunk!r}"
        # Every chunk should end with sentence-ending punctuation.
        assert chunk.rstrip()[-1] in ".!?", f"Chunk ends mid-sentence: {chunk!r}"


def test_chunk_text_keeps_short_sentences_together_in_one_chunk():
    text = "Short one. Short two. Short three."
    chunks = chunk_text(text, chunk_size=200, overlap=20)

    # All three sentences comfortably fit in one chunk_size=200 chunk.
    assert len(chunks) == 1
    assert chunks[0] == "Short one. Short two. Short three."


def test_chunk_text_overlap_carries_whole_sentences_between_chunks():
    text = (
        "First sentence here for the test. "
        "Second sentence continues the thought. "
        "Third sentence pushes past the limit now. "
        "Fourth sentence should land in the next chunk."
    )
    chunks = chunk_text(text, chunk_size=90, overlap=40)

    assert len(chunks) >= 2
    # The overlap sentence carried into chunk two should be a complete
    # sentence (not a word fragment) shared with the end of chunk one.
    first_chunk_sentences = set(_split_sentences(chunks[0]))
    second_chunk_sentences = set(_split_sentences(chunks[1]))
    assert first_chunk_sentences & second_chunk_sentences


def test_chunk_text_falls_back_to_word_split_for_oversized_single_sentence():
    # A "sentence" with no punctuation at all (e.g. a long unbroken line)
    # that alone exceeds chunk_size must still get split, not emitted whole.
    long_word = "a" * 50
    text = f"start {long_word} end with no period at all here to force overflow"
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert all(len(c) <= max(20, len(long_word)) for c in chunks)
    assert any(long_word in c for c in chunks)


def test_load_metadata_returns_empty_list_when_file_missing(tmp_path: Path):
    assert load_metadata(tmp_path / "missing.json") == []


def test_save_and_load_metadata_round_trip(tmp_path: Path):
    metadata_path = tmp_path / "nested" / "metadata.json"
    data = [{"vector_id": 0, "doc_id": "x", "chunk_id": 0, "text": "hello"}]

    save_metadata(data, metadata_path)

    assert metadata_path.exists()
    assert load_metadata(metadata_path) == data
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == data


def test_ingest_document_raises_when_no_searchable_text(tmp_path: Path):
    path = tmp_path / "empty.txt"
    path.write_text("short\nlines\nonly", encoding="utf-8")  # all filtered by clean_text

    with pytest.raises(ValueError, match="No searchable text"):
        ingest_document(
            path,
            ".txt",
            model=_FakeModel(),
            index_path=tmp_path / "faiss.index",
            metadata_path=tmp_path / "metadata.json",
        )


def test_ingest_document_end_to_end_with_fake_model(tmp_path: Path):
    path = tmp_path / "doc.txt"
    path.write_text(
        "This is a perfectly reasonable sentence for indexing purposes today.",
        encoding="utf-8",
    )

    result = ingest_document(
        path,
        ".txt",
        model=_FakeModel(),
        index_path=tmp_path / "faiss.index",
        metadata_path=tmp_path / "metadata.json",
    )

    assert result["chunks_indexed"] >= 1
    assert "doc_id" in result

    metadata = load_metadata(tmp_path / "metadata.json")
    assert len(metadata) == result["chunks_indexed"]
    assert (tmp_path / "faiss.index").exists()


def test_ingest_document_appends_to_existing_index(tmp_path: Path):
    index_path = tmp_path / "faiss.index"
    metadata_path = tmp_path / "metadata.json"

    path_a = tmp_path / "a.txt"
    path_a.write_text("First document with a reasonably long sentence.", encoding="utf-8")
    path_b = tmp_path / "b.txt"
    path_b.write_text("Second document with another reasonably long sentence.", encoding="utf-8")

    ingest_document(path_a, ".txt", model=_FakeModel(), index_path=index_path, metadata_path=metadata_path)
    ingest_document(path_b, ".txt", model=_FakeModel(), index_path=index_path, metadata_path=metadata_path)

    metadata = load_metadata(metadata_path)
    doc_ids = {item["doc_id"] for item in metadata}
    assert len(doc_ids) == 2
    # vector_ids should be sequential across both ingests, not reset
    assert sorted(item["vector_id"] for item in metadata) == list(range(len(metadata)))


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer so tests don't need the real model."""

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[float(len(t) % 7), float(len(t) % 5)] for t in texts]
