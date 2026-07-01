import io
import time

import pytest
from fastapi import HTTPException, UploadFile

from api import security


# ---- sanitize_query ----

def test_sanitize_query_strips_null_bytes_and_collapses_whitespace():
    assert security.sanitize_query("hello\x00   world\n\tfoo") == "hello world foo"


def test_sanitize_query_rejects_none():
    with pytest.raises(HTTPException) as exc:
        security.sanitize_query(None)
    assert exc.value.status_code == 400


def test_sanitize_query_rejects_empty_after_cleaning():
    with pytest.raises(HTTPException) as exc:
        security.sanitize_query("   \x00  ")
    assert exc.value.status_code == 400


def test_sanitize_query_rejects_overlong_query():
    with pytest.raises(HTTPException) as exc:
        security.sanitize_query("a" * 501)
    assert exc.value.status_code == 400


def test_sanitize_query_accepts_exactly_500_chars():
    assert security.sanitize_query("a" * 500) == "a" * 500


# ---- validate_search_mode ----

@pytest.mark.parametrize("raw,expected", [("semantic", "semantic"), (" KEYWORD ", "keyword"), ("Both", "both")])
def test_validate_search_mode_normalizes_case_and_whitespace(raw, expected):
    assert security.validate_search_mode(raw) == expected


def test_validate_search_mode_rejects_unknown_mode():
    with pytest.raises(HTTPException) as exc:
        security.validate_search_mode("hybrid")
    assert exc.value.status_code == 400


# ---- extension_for_filename ----

def test_extension_for_filename_accepts_allowed_types():
    assert security.extension_for_filename("notes.TXT") == ".txt"
    assert security.extension_for_filename("report.pdf") == ".pdf"


def test_extension_for_filename_rejects_disallowed_type():
    with pytest.raises(HTTPException) as exc:
        security.extension_for_filename("virus.exe")
    assert exc.value.status_code == 400


def test_extension_for_filename_rejects_missing_filename():
    with pytest.raises(HTTPException):
        security.extension_for_filename(None)


def test_extension_for_filename_rejects_double_extension_bypass_attempt():
    # A classic upload-filter bypass: "shell.php.txt" should resolve on the
    # last suffix only, which IS allowed here (.txt) -- documents current
    # behavior so a regression to a more permissive check gets caught.
    assert security.extension_for_filename("shell.php.txt") == ".txt"


# ---- validate_file_bytes ----

def test_validate_file_bytes_rejects_empty_file():
    with pytest.raises(HTTPException) as exc:
        security.validate_file_bytes(b"", ".txt")
    assert exc.value.status_code == 400


def test_validate_file_bytes_rejects_oversized_file():
    oversized = b"a" * (security.MAX_FILE_SIZE + 1)
    with pytest.raises(HTTPException) as exc:
        security.validate_file_bytes(oversized, ".txt")
    assert exc.value.status_code == 413


def test_validate_file_bytes_accepts_file_at_exact_size_limit():
    at_limit = b"a" * security.MAX_FILE_SIZE
    security.validate_file_bytes(at_limit, ".txt")  # should not raise


def test_validate_file_bytes_rejects_pdf_without_pdf_magic_bytes():
    with pytest.raises(HTTPException) as exc:
        security.validate_file_bytes(b"not a real pdf", ".pdf")
    assert exc.value.status_code == 400


def test_validate_file_bytes_accepts_pdf_with_correct_magic_bytes():
    security.validate_file_bytes(b"%PDF-1.4 rest of file", ".pdf")  # should not raise


def test_validate_file_bytes_rejects_text_file_with_null_bytes():
    with pytest.raises(HTTPException) as exc:
        security.validate_file_bytes(b"hello\x00world", ".txt")
    assert exc.value.status_code == 400


def test_validate_file_bytes_rejects_non_utf8_text_file():
    with pytest.raises(HTTPException) as exc:
        security.validate_file_bytes(b"\xff\xfe\x00\x01bad encoding", ".txt")
    assert exc.value.status_code == 400


def test_validate_file_bytes_accepts_valid_utf8_text():
    security.validate_file_bytes("hello world \u00e9\u00e8".encode("utf-8"), ".md")  # should not raise


# ---- check_rate_limit ----

class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host="1.2.3.4"):
        self.client = _FakeClient(host)


def test_check_rate_limit_allows_requests_under_the_limit():
    security._request_times.clear()
    req = _FakeRequest("10.0.0.1")
    for _ in range(security.RATE_LIMIT_COUNT):
        security.check_rate_limit(req)  # should not raise


def test_check_rate_limit_blocks_requests_over_the_limit():
    security._request_times.clear()
    req = _FakeRequest("10.0.0.2")
    for _ in range(security.RATE_LIMIT_COUNT):
        security.check_rate_limit(req)

    with pytest.raises(HTTPException) as exc:
        security.check_rate_limit(req)
    assert exc.value.status_code == 429


def test_check_rate_limit_tracks_ips_independently():
    security._request_times.clear()
    req_a = _FakeRequest("10.0.0.3")
    req_b = _FakeRequest("10.0.0.4")
    for _ in range(security.RATE_LIMIT_COUNT):
        security.check_rate_limit(req_a)

    security.check_rate_limit(req_b)  # different IP, should not raise


def test_check_rate_limit_expires_old_requests(monkeypatch):
    security._request_times.clear()
    req = _FakeRequest("10.0.0.5")

    fake_now = [1_000_000.0]
    monkeypatch.setattr(security.time, "time", lambda: fake_now[0])

    for _ in range(security.RATE_LIMIT_COUNT):
        security.check_rate_limit(req)

    # Jump past the rate limit window; old timestamps should no longer count.
    fake_now[0] += security.RATE_LIMIT_WINDOW_SECONDS + 1
    security.check_rate_limit(req)  # should not raise


def test_check_rate_limit_handles_missing_client():
    security._request_times.clear()
    req = _FakeRequest.__new__(_FakeRequest)
    req.client = None
    security.check_rate_limit(req)  # falls back to "unknown" bucket, should not raise


# ---- save_upload_file / cleanup_file (path traversal + persistence) ----

def _make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(content))


@pytest.mark.asyncio
async def test_save_upload_file_writes_inside_uploads_dir(tmp_path):
    uploads_dir = tmp_path / "uploads"
    upload = _make_upload("notes.txt", b"hello world, this is fine")

    saved_path, ext = await security.save_upload_file(upload, uploads_dir)

    assert ext == ".txt"
    assert saved_path.exists()
    assert saved_path.parent.resolve() == uploads_dir.resolve()
    assert saved_path.read_bytes() == b"hello world, this is fine"


@pytest.mark.asyncio
async def test_save_upload_file_generates_random_safe_filename(tmp_path):
    # Guards against path traversal via crafted filenames like "../../evil.txt"
    uploads_dir = tmp_path / "uploads"
    upload = _make_upload("../../../etc/evil.txt", b"data")

    saved_path, _ = await security.save_upload_file(upload, uploads_dir)

    assert saved_path.parent.resolve() == uploads_dir.resolve()
    assert ".." not in saved_path.name


@pytest.mark.asyncio
async def test_save_upload_file_rejects_disallowed_extension(tmp_path):
    uploads_dir = tmp_path / "uploads"
    upload = _make_upload("malware.exe", b"data")

    with pytest.raises(HTTPException) as exc:
        await security.save_upload_file(upload, uploads_dir)
    assert exc.value.status_code == 400


def test_cleanup_file_removes_existing_file(tmp_path):
    path = tmp_path / "file.txt"
    path.write_text("data")
    security.cleanup_file(path)
    assert not path.exists()


def test_cleanup_file_is_silent_when_file_missing(tmp_path):
    path = tmp_path / "does_not_exist.txt"
    security.cleanup_file(path)  # should not raise

