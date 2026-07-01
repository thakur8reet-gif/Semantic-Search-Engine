from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Protocol


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MODEL_NAME = "all-MiniLM-L6-v2"
DATA_DIR = Path("data")
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.json"


class EmbeddingModel(Protocol):
    def encode(self, texts: list[str] | str):
        ...


_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def extract_text(path: Path, ext: str) -> str:
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError(f"Unsupported extension: {ext}")


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    useful_lines = [line for line in lines if len(line) > 20]
    cleaned = " ".join(" ".join(useful_lines).split())
    return cleaned


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    start_word = 0

    while start_word < len(words):
        chunk_words: list[str] = []
        current_length = 0
        word_index = start_word

        while word_index < len(words):
            word = words[word_index]
            next_length = len(word) if not chunk_words else current_length + 1 + len(word)
            if chunk_words and next_length > chunk_size:
                break
            chunk_words.append(word)
            current_length = next_length
            word_index += 1

        if not chunk_words:
            chunk_words.append(words[word_index])
            word_index += 1

        chunks.append(" ".join(chunk_words))
        if word_index >= len(words):
            break

        overlap_words: list[str] = []
        overlap_length = 0
        previous_index = word_index - 1
        while previous_index >= start_word:
            word = words[previous_index]
            next_length = len(word) if not overlap_words else overlap_length + 1 + len(word)
            if next_length > overlap:
                break
            overlap_words.insert(0, word)
            overlap_length = next_length
            previous_index -= 1

        start_word = max(previous_index + 1, start_word + 1)
    return chunks


def load_metadata(metadata_path: Path = METADATA_PATH) -> list[dict]:
    if not metadata_path.exists():
        return []
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_metadata(metadata: list[dict], metadata_path: Path = METADATA_PATH) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _as_float32_array(values):
    import numpy as np

    return np.asarray(values, dtype="float32")


def ingest_document(
    path: Path,
    ext: str,
    *,
    doc_id: str | None = None,
    model: EmbeddingModel | None = None,
    index_path: Path = INDEX_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict:
    import faiss

    doc_id = doc_id or str(uuid.uuid4())
    raw_text = extract_text(path, ext)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)

    if not chunks:
        raise ValueError("No searchable text found in document")

    embedding_model = model or get_embedding_model()
    embeddings = _as_float32_array(embedding_model.encode(chunks))
    faiss.normalize_L2(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index = faiss.read_index(str(index_path))
    else:
        index = faiss.IndexFlatIP(embeddings.shape[1])

    start_position = index.ntotal
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    metadata = load_metadata(metadata_path)
    for offset, chunk in enumerate(chunks):
        metadata.append(
            {
                "vector_id": start_position + offset,
                "doc_id": doc_id,
                "chunk_id": offset,
                "text": chunk,
            }
        )
    save_metadata(metadata, metadata_path)

    return {"doc_id": doc_id, "chunks_indexed": len(chunks)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest a document into FAISS.")
    parser.add_argument("path", help="Path to a .txt, .md, or .pdf document")
    args = parser.parse_args()

    input_path = Path(args.path)
    result = ingest_document(input_path, input_path.suffix.lower())
    print(result)
