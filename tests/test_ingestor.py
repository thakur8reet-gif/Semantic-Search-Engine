from pathlib import Path

from api.ingestor import chunk_text, clean_text, extract_text


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
