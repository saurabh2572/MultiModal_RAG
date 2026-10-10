"""Local Streamlit chat UI for the LangGraph online RAG workflow."""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st

from multimodal_rag.online_graph import OnlineRAG
from multimodal_rag.settings import Settings


st.set_page_config(page_title="Multimodal Document RAG", page_icon="📚", layout="wide")
st.title("Multimodal Document RAG")
st.caption("Hybrid search across document text, tables, and figures, with page-level citations.")

settings = Settings.from_env()
if not settings.access_token:
    st.error("Set RAG_ACCESS_TOKEN in .env before using the app.")
    st.stop()

with st.sidebar:
    st.subheader("Access")
    if st.session_state.get("authenticated"):
        st.success("Authenticated")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
    else:
        with st.form("login"):
            token = st.text_input("Access token", type="password")
            submitted = st.form_submit_button("Unlock")
        if submitted:
            if settings.authenticate(token):
                st.session_state["authenticated"] = True
                st.session_state["access_token"] = token
                st.rerun()
            else:
                st.error("Invalid access token")
    st.divider()
    st.write(f"LLM: `{settings.llm_provider}` / `{settings.llm_model}`")
    st.write("LangSmith tracing: " + ("on" if settings.tracing_enabled else "off"))
    if st.session_state.get("authenticated") and st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()

if not st.session_state.get("authenticated"):
    st.info("Enter the access token in the sidebar to start chatting.")
    st.stop()


@st.cache_resource(show_spinner="Loading retrieval indexes and models...")
def get_pipeline(config: Settings) -> OnlineRAG:
    config.validate_runtime()
    return OnlineRAG(config)


try:
    pipeline = get_pipeline(settings)
except (FileNotFoundError, ValueError, RuntimeError) as exc:
    st.error(str(exc))
    st.info("Build the vector database and BM25 index first; see README.md for commands.")
    st.stop()

messages = st.session_state.setdefault("messages", [])


def show_images(citations: list[dict]) -> None:
    """Render only image files attached to retrieved evidence, never LLM-provided URLs."""
    for source in citations:
        image_path = source.get("image_path")
        if source.get("modality") != "image" or not image_path:
            continue
        path = Path(image_path)
        if not path.is_file():
            continue
        try:
            st.image(
                path.read_bytes(),
                caption=f"Source [{source['citation']}] — {source['source']}, page {source['page_label']}",
                width="stretch",
            )
        except OSError:
            st.warning(f"Source [{source['citation']}] image could not be read.")


def show_sources(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for source in citations:
            st.markdown(
                f"**[{source['citation']}] {source['source']} — PDF page {source['page']}** "
                f"(printed page {source['page_label']}; {source['modality']}; {source['section']})"
            )
            st.caption(source["snippet"])


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_images(message.get("citations", []))
            show_sources(message.get("citations", []))

if question := st.chat_input("Ask about the annual report"):
    history = [{"role": message["role"], "content": message["content"]} for message in messages]
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching text, tables, and figures..."):
            try:
                result = asyncio.run(pipeline.ask(question, st.session_state["access_token"], history))
            except Exception as exc:
                st.error(f"Query failed: {type(exc).__name__}: {exc}")
            else:
                st.markdown(result["answer"])
                show_images(result["citations"])
                show_sources(result["citations"])
                messages.append({"role": "assistant", "content": result["answer"], "citations": result["citations"]})
