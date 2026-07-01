from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import HTTPException, Request, UploadFile


ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 200 * 1024 * 1024
RATE_LIMIT_COUNT = 10
RATE_LIMIT_WINDOW_SECONDS = 60

_request_times: dict[str, list[float]] = defaultdict(list)


def sanitize_query(query: str | None) -> str:
    if query is None:
        raise HTTPException(status_code=400, detail="Query is required")

    cleaned = query.replace("\x00", "")
    cleaned = " ".join(cleaned.split())

    if not cleaned:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(cleaned) > 500:
        raise HTTPException(status_code=400, detail="Query is too long")

    return cleaned


def validate_search_mode(mode: str) -> str:
    normalized = mode.lower().strip()
    if normalized not in {"semantic", "keyword", "both"}:
        raise HTTPException(status_code=400, detail="Invalid search mode")
    return normalized


def check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    recent = [t for t in _request_times[ip] if t > window_start]
    if len(recent) >= RATE_LIMIT_COUNT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    recent.append(now)
    _request_times[ip] = recent


def extension_for_filename(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return ext


def validate_file_bytes(contents: bytes, ext: str) -> None:
    if not contents:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")

    if ext == ".pdf" and not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    if ext in {".txt", ".md"}:
        sample = contents[:4096]
        if b"\x00" in sample:
            raise HTTPException(status_code=400, detail="Invalid text file")
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text files must be UTF-8") from exc


async def save_upload_file(upload: UploadFile, uploads_dir: Path) -> tuple[Path, str]:
    ext = extension_for_filename(upload.filename)
    contents = await upload.read()
    validate_file_bytes(contents, ext)

    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4()}{ext}"
    save_path = (uploads_dir / safe_name).resolve()
    uploads_root = uploads_dir.resolve()

    try:
        save_path.relative_to(uploads_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload path") from exc

    save_path.write_bytes(contents)
    return save_path, ext


def cleanup_file(path: Path) -> None:
    try:
        if path.exists():
            os.remove(path)
    except OSError:
        pass
