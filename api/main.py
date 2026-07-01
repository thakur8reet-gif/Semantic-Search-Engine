from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.ingestor import ingest_document
from api.models import SearchResponse, UploadResponse
from api.searcher import Searcher
from api.security import check_rate_limit, cleanup_file, sanitize_query, save_upload_file, validate_search_mode


UPLOADS_DIR = Path("uploads")

app = FastAPI(title="Semantic Search Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

searcher = Searcher()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile) -> UploadResponse:
    saved_path: Path | None = None
    try:
        saved_path, ext = await save_upload_file(file, UPLOADS_DIR)
        result = ingest_document(saved_path, ext)
        searcher.refresh()
        return UploadResponse(
            doc_id=result["doc_id"],
            chunks_indexed=result["chunks_indexed"],
            message="Document indexed successfully",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Upload failed. Try again.") from exc
    finally:
        if saved_path is not None:
            cleanup_file(saved_path)


@app.get("/search", response_model=SearchResponse)
def search(request: Request, q: str | None = None, mode: str = "both") -> SearchResponse:
    try:
        check_rate_limit(request)
        query = sanitize_query(q)
        search_mode = validate_search_mode(mode)
        return SearchResponse(**searcher.search(query, search_mode))
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail="Search failed. Try again.") from exc

