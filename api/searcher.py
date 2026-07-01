from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

from api.ingestor import INDEX_PATH, METADATA_PATH, get_embedding_model, load_metadata


class Searcher:
    def __init__(self, index_path: Path = INDEX_PATH, metadata_path: Path = METADATA_PATH):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata: list[dict] = []
        self.bm25 = None
        self.refresh()

    def refresh(self) -> None:
        self.metadata = load_metadata(self.metadata_path)
        tokenized_chunks = [item["text"].lower().split() for item in self.metadata]
        self.bm25 = build_bm25(tokenized_chunks) if tokenized_chunks else None

        if self.index_path.exists():
            import faiss

            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

    def semantic_search(self, query: str, k: int = 5) -> tuple[list[dict], int]:
        start = time.perf_counter()
        if self.index is None or not self.metadata:
            return [], elapsed_ms(start)

        import faiss
        import numpy as np

        model = get_embedding_model()
        query_embedding = np.asarray(model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, min(k, len(self.metadata)))

        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = self.metadata[int(idx)]
            results.append(
                {
                    "doc_id": item["doc_id"],
                    "chunk_id": item["chunk_id"],
                    "score": float(score),
                    "text": item["text"],
                }
            )

        return results, elapsed_ms(start)

    def keyword_search(self, query: str, k: int = 5) -> tuple[list[dict], int]:
        start = time.perf_counter()
        if self.bm25 is None or not self.metadata:
            return [], elapsed_ms(start)

        scores = self.bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]

        results: list[dict] = []
        for idx in ranked_indices:
            item = self.metadata[idx]
            results.append(
                {
                    "doc_id": item["doc_id"],
                    "chunk_id": item["chunk_id"],
                    "score": float(scores[idx]),
                    "text": item["text"],
                }
            )

        return results, elapsed_ms(start)

    def search(self, query: str, mode: str = "both", k: int = 5) -> dict:
        response = {
            "semantic_results": [],
            "keyword_results": [],
            "semantic_time_taken_ms": None,
            "keyword_time_taken_ms": None,
        }

        if mode == "semantic":
            response["semantic_results"], response["semantic_time_taken_ms"] = self.semantic_search(query, k)
            return response

        if mode == "keyword":
            response["keyword_results"], response["keyword_time_taken_ms"] = self.keyword_search(query, k)
            return response

        with ThreadPoolExecutor(max_workers=2) as executor:
            semantic_future = executor.submit(self.semantic_search, query, k)
            keyword_future = executor.submit(self.keyword_search, query, k)
            response["semantic_results"], response["semantic_time_taken_ms"] = semantic_future.result()
            response["keyword_results"], response["keyword_time_taken_ms"] = keyword_future.result()

        return response


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def build_bm25(tokenized_chunks: list[list[str]]):
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_chunks)
    except ModuleNotFoundError:
        return SimpleBM25(tokenized_chunks)


class SimpleBM25:
    """Small fallback for local tests; install rank-bm25 for the real app."""

    def __init__(self, tokenized_chunks: list[list[str]]):
        self.documents = tokenized_chunks
        self.term_counts = [Counter(doc) for doc in tokenized_chunks]

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for counts in self.term_counts:
            scores.append(float(sum(counts[token] for token in query_tokens)))
        return scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search indexed documents.")
    parser.add_argument("query")
    parser.add_argument("--mode", default="both", choices=["semantic", "keyword", "both"])
    args = parser.parse_args()

    print(Searcher().search(args.query, args.mode))
