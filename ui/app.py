import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Semantic Search", layout="wide")
st.title("Semantic Search")

with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload .txt, .md, or .pdf", type=["txt", "md", "pdf"])
    if uploaded_file and st.button("Index document", type="primary"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
        if response.ok:
            data = response.json()
            st.success(f"Indexed {data['chunks_indexed']} chunks")
        else:
            st.error(response.json().get("detail", "Upload failed"))

query = st.text_input("Search query")
mode = st.segmented_control("Mode", ["both", "semantic", "keyword"], default="both")

if st.button("Search", type="primary", disabled=not query):
    response = requests.get(f"{API_BASE_URL}/search", params={"q": query, "mode": mode}, timeout=60)
    if not response.ok:
        st.error(response.json().get("detail", "Search failed"))
    else:
        data = response.json()
        semantic_col, keyword_col = st.columns(2)

        with semantic_col:
            st.subheader("Semantic Results")
            if data["semantic_time_taken_ms"] is not None:
                st.caption(f"Time: {data['semantic_time_taken_ms']} ms")
            for result in data["semantic_results"]:
                st.metric("Cosine score", f"{result['score']:.3f}")
                st.write(result["text"])
                st.divider()

        with keyword_col:
            st.subheader("Keyword Results")
            if data["keyword_time_taken_ms"] is not None:
                st.caption(f"Time: {data['keyword_time_taken_ms']} ms")
            for result in data["keyword_results"]:
                st.metric("BM25 score", f"{result['score']:.3f}")
                st.write(result["text"])
                st.divider()

