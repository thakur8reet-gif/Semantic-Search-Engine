from pydantic import BaseModel


class SearchResult(BaseModel):
    doc_id: str
    chunk_id: int
    score: float
    text: str


class SearchResponse(BaseModel):
    semantic_results: list[SearchResult] = []
    keyword_results: list[SearchResult] = []
    semantic_time_taken_ms: int | None = None
    keyword_time_taken_ms: int | None = None


class UploadResponse(BaseModel):
    doc_id: str
    chunks_indexed: int
    message: str

