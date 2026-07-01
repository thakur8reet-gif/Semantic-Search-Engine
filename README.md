# Semantic Search Engine

Document search project with side-by-side semantic search and keyword search.

## Features

- Upload `.txt`, `.md`, and `.pdf` documents through FastAPI.
- Validate extension, file size, and file signature before ingestion.
- Chunk text into 800-character windows with 100-character overlap.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`.
- Store normalized vectors in FAISS and readable chunks in `data/metadata.json`.
- Search using semantic FAISS retrieval, BM25 keyword retrieval, or both.
- Compare semantic and keyword results in a Streamlit two-column UI.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run The API

```bash
uvicorn api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to test upload and search.

## Run The UI

```bash
streamlit run ui/app.py
```

## Test

```bash
pytest
```

## Project Layout

```text
api/
  main.py
  ingestor.py
  searcher.py
  security.py
  models.py
data/
uploads/
ui/
  app.py
tests/
```

Generated files under `data/` and raw uploads are intentionally gitignored because they can be large and may contain user data.
