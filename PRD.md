# Product Requirements Document: Semantic Search Engine

## 1. Overview

The Semantic Search Engine lets users upload documents and search them using both semantic retrieval and keyword retrieval. The main product value is the side-by-side comparison of semantic search with BM25 keyword search, making the differences between meaning-based and exact-term retrieval visible.

## 2. Goals

- Let users upload `.txt`, `.md`, and `.pdf` documents.
- Convert document text into searchable chunks.
- Store semantic embeddings in FAISS.
- Store readable chunk metadata separately.
- Support semantic search, keyword search, and comparison mode.
- Show relevance scores and search timing for each retrieval method.
- Provide a simple Streamlit UI for upload and search.

## 3. Non-Goals

- Multi-user authentication.
- Cloud storage.
- Long-term document management.
- Advanced access control.
- Production-grade distributed vector storage.
- Rich PDF layout preservation.

## 4. Target Users

- Students building an AI/data portfolio project.
- Recruiters or interviewers evaluating information retrieval understanding.
- Developers learning the difference between semantic and keyword search.

## 5. Core User Stories

- As a user, I can upload a supported document so its content becomes searchable.
- As a user, I can enter a query and receive semantic matches based on meaning.
- As a user, I can enter a query and receive BM25 keyword matches based on exact terms.
- As a user, I can compare semantic and keyword results side by side.
- As a user, I can see scores and response times to understand result quality and performance.

## 6. Functional Requirements

### Document Upload

- The API must expose `POST /upload`.
- The upload endpoint must accept only `.txt`, `.md`, and `.pdf`.
- The backend must validate the file extension and basic file signature.
- The backend must reject empty files.
- The backend must reject files larger than 200 MB.
- Uploaded files must be saved temporarily using UUID filenames.
- Raw uploaded files must be deleted after ingestion.

### Text Extraction

- Text and Markdown files must be read as UTF-8.
- PDF files must be extracted using `pypdf`.
- Extracted text must be cleaned before chunking.
- Cleaning must remove short lines and collapse whitespace.

### Chunking

- Text must be split into chunks of 800 characters.
- Adjacent chunks must overlap by 100 characters.
- Empty chunks must not be indexed.

### Embedding And Indexing

- Embeddings must use `sentence-transformers/all-MiniLM-L6-v2`.
- The model should be loaded lazily and reused.
- Embeddings must be converted to `float32`.
- Embeddings must be L2-normalized.
- Vectors must be stored in FAISS using `IndexFlatIP`.
- Metadata must be stored in `data/metadata.json`.
- FAISS index must be stored in `data/faiss.index`.

### Search

- The API must expose `GET /search?q=...&mode=semantic|keyword|both`.
- Queries must be sanitized by removing null bytes and normalizing whitespace.
- Empty queries must be rejected.
- Queries longer than 500 characters must be rejected.
- Invalid search modes must be rejected.
- Semantic search must return top 5 FAISS matches.
- Keyword search must return top 5 BM25 matches.
- Comparison mode must return both result sets in one response.
- Search responses must include timing in milliseconds for each requested search mode.

### Streamlit UI

- The UI must allow users to upload supported documents.
- The UI must allow users to enter a search query.
- The UI must provide mode selection: `both`, `semantic`, or `keyword`.
- In comparison mode, semantic and keyword results must appear side by side.
- Results must show score, text, and timing.

## 7. API Contract

### `POST /upload`

Request:

```text
multipart/form-data
file=<uploaded file>
```

Successful response:

```json
{
  "doc_id": "uuid",
  "chunks_indexed": 12,
  "message": "Document indexed successfully"
}
```

### `GET /search`

Query parameters:

```text
q=<query string>
mode=semantic|keyword|both
```

Successful response:

```json
{
  "semantic_results": [
    {
      "doc_id": "uuid",
      "chunk_id": 0,
      "score": 0.82,
      "text": "matching chunk text"
    }
  ],
  "keyword_results": [
    {
      "doc_id": "uuid",
      "chunk_id": 1,
      "score": 14.2,
      "text": "matching chunk text"
    }
  ],
  "semantic_time_taken_ms": 43,
  "keyword_time_taken_ms": 2
}
```

## 8. Security Requirements

- Never trust user-provided filenames.
- Store uploads using UUID filenames only.
- Prevent path traversal by resolving paths inside the upload directory.
- Reject unsupported file extensions.
- Validate basic file signatures.
- Reject oversized files before ingestion.
- Sanitize search queries.
- Apply simple per-IP rate limiting.
- Restrict CORS to the Streamlit frontend origin.
- Return generic error messages for unexpected failures.
- Log internal errors server-side only.

## 9. Data Requirements

- `data/faiss.index` stores vectors only.
- `data/metadata.json` maps vector positions to readable chunks.
- Generated data files must not be committed to Git.
- Uploaded raw files must not persist after ingestion.

## 10. Success Metrics

- A supported document can be uploaded and indexed successfully.
- A semantic query returns relevant meaning-based chunks.
- A keyword query returns exact-term-focused chunks.
- Comparison mode shows visibly different ranking behavior when appropriate.
- Search results return within an acceptable local development range.
- Tests pass with `pytest`.

## 11. Acceptance Criteria

- `pytest` passes.
- `POST /upload` indexes `.txt`, `.md`, and valid `.pdf` files.
- Invalid file types are rejected.
- Empty or oversized files are rejected.
- `GET /search` rejects empty, oversized, and invalid-mode queries.
- Semantic search returns top results with cosine scores.
- Keyword search returns top results with BM25 scores.
- Streamlit UI can upload documents and display search results.
- `README.md` explains setup, testing, and running the app.

## 12. Milestones

### Milestone 1: Ingestion

- Create project structure.
- Implement text extraction.
- Implement cleaning and chunking.
- Add ingestion tests.

### Milestone 2: Indexing

- Add sentence-transformer embeddings.
- Add FAISS storage.
- Add metadata storage.

### Milestone 3: Search

- Add semantic search.
- Add BM25 keyword search.
- Add comparison mode with timing.

### Milestone 4: API

- Add FastAPI upload endpoint.
- Add FastAPI search endpoint.
- Add validation, rate limiting, and CORS.

### Milestone 5: UI

- Add Streamlit upload flow.
- Add query input and mode selector.
- Add side-by-side result comparison.
