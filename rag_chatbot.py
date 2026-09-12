import os
import re
from pathlib import Path

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ABOUT_ME_CANDIDATES = (APP_DIR / "about_myself.txt", APP_DIR / "about_me.txt")
ABOUT_ME_PATH = next((path for path in ABOUT_ME_CANDIDATES if path.exists()), ABOUT_ME_CANDIDATES[0])
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 5
CHUNK_OVERLAP = 2
TOP_K = 3
SYSTEM_INSTRUCTION = (
    "Answer ONLY using this context about Raj. "
    "If the answer is not in the context, say you don’t have that information."
)
WELCOME_MESSAGE = (
    "Hi — I’m Raj’s personal Q&A bot. Ask about his payments work, family, "
    "sports, gaming, or the two cats. I’ll answer from his bio and show the "
    "source chunks I used."
)
EXAMPLE_QUESTIONS = [
    "What are your hobbies?",
    "What do you do for work?",
    "Do you have any pets?",
]
CUSTOM_CSS = """
<style>
@import url("https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap");

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 24%),
        linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
}

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 6rem;
    max-width: 760px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stDecoration"] {display: none;}
[data-testid="stHeader"] {display: none;}
[data-testid="stStatusWidget"] {display: none;}

h1 {
    letter-spacing: -0.03em;
    font-weight: 700 !important;
    color: #0f172a;
}

[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(15, 23, 42, 0.06);
    border-radius: 18px;
    padding: 0.85rem 1rem;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
}

[data-testid="stChatInput"] {
    border-radius: 18px;
}

[data-testid="stExpander"] {
    background: #f8fafc;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 14px;
}

.source-card {
    background: #ffffff;
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    color: #334155;
    line-height: 1.55;
}

.source-label {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #2563eb;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: #475569;
    font-size: 1.05rem;
    margin: -0.4rem 0 1.2rem 0;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid rgba(37, 99, 235, 0.18);
    background: #ffffff;
    color: #1e3a8a;
    font-weight: 500;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
}

.stButton > button:hover {
    border-color: rgba(37, 99, 235, 0.4);
    color: #1d4ed8;
}
</style>
"""


def split_into_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def chunk_sentences(
    sentences: list[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    if not sentences:
        return []

    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(sentences), step):
        window = sentences[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(sentences):
            break
    return chunks


def embed_chunks(client: OpenAI, chunks: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def load_chunks_and_embeddings(client: OpenAI) -> tuple[list[str], list[list[float]]]:
    text = ABOUT_ME_PATH.read_text(encoding="utf-8")
    sentences = split_into_sentences(text)
    chunks = chunk_sentences(sentences)
    embeddings = embed_chunks(client, chunks)
    return chunks, embeddings


def embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return response.data[0].embedding


def cosine_similarity(chunk_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_embedding)
    return (chunk_embeddings @ query_embedding) / (chunk_norms.flatten() * query_norm)


def top_k_chunks(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunks: list[str],
    k: int = TOP_K,
) -> list[str]:
    query = np.array(query_embedding, dtype=float)
    matrix = np.array(chunk_embeddings, dtype=float)
    scores = cosine_similarity(matrix, query)
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunks[int(index)] for index in top_indices]


def answer_from_context(client: OpenAI, question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}",
            },
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content or ""


def render_sources(sources: list[str]) -> None:
    with st.expander("Sources"):
        for index, chunk in enumerate(sources, start=1):
            st.markdown(
                f'<div class="source-card"><div class="source-label">Chunk {index}</div>{chunk}</div>',
                unsafe_allow_html=True,
            )


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            render_sources(message["sources"])


def default_messages() -> list[dict]:
    return [{"role": "assistant", "content": WELCOME_MESSAGE, "sources": []}]


def handle_user_prompt(prompt: str, client: OpenAI | None) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    sources: list[str] = []
    if not client or not st.session_state.embeddings:
        response = "I can’t answer yet because the knowledge base is not indexed."
    else:
        with st.spinner("Thinking..."):
            query_embedding = embed_query(client, prompt)
            sources = top_k_chunks(
                query_embedding,
                st.session_state.embeddings,
                st.session_state.chunks,
            )
            response = answer_from_context(client, prompt, sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(response)
        if sources:
            render_sources(sources)


st.set_page_config(
    page_title="Ask Me Anything About Raj Reddy",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("🤖 Ask Me Anything About Raj Reddy")
st.markdown(
    '<p class="hero-subtitle">A private Q&amp;A bot trained on Raj’s bio.</p>',
    unsafe_allow_html=True,
)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
about_me_exists = ABOUT_ME_PATH.exists()

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is missing. Add it to a `.env` file in this folder "
        "(OPENAI_API_KEY=your-key), then refresh the app."
    )

if not about_me_exists:
    st.error(
        f"`{ABOUT_ME_PATH.name}` was not found. Place `about_myself.txt` "
        "(or `about_me.txt`) next to `rag_chatbot.py`, then refresh the app."
    )

if "chunks" not in st.session_state or "embeddings" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.embeddings = []
    if client and about_me_exists:
        try:
            with st.spinner("Indexing Raj’s bio..."):
                st.session_state.chunks, st.session_state.embeddings = (
                    load_chunks_and_embeddings(client)
                )
        except FileNotFoundError:
            st.error(
                f"`{ABOUT_ME_PATH.name}` was not found. Place `about_myself.txt` "
                "next to `rag_chatbot.py`, then refresh the app."
            )

if "messages" not in st.session_state:
    st.session_state.messages = default_messages()

example_cols = st.columns(3)
for column, question in zip(example_cols, EXAMPLE_QUESTIONS):
    if column.button(question, use_container_width=True):
        st.session_state.pending_prompt = question
        st.rerun()

if st.button("Clear chat"):
    st.session_state.messages = default_messages()
    st.session_state.pop("pending_prompt", None)
    st.rerun()

for message in st.session_state.messages:
    render_message(message)

typed_prompt = st.chat_input("Ask me anything about Raj Reddy...")
prompt = st.session_state.pop("pending_prompt", None) or typed_prompt
if prompt:
    handle_user_prompt(prompt, client)
