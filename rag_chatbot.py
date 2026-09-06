import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Ask Me Anything About Raj Reddy", page_icon="🤖")
st.title("🤖 Ask Me Anything About Raj Reddy")

if not OPENAI_API_KEY:
    st.warning("OPENAI_API_KEY is not set. Add it to your .env file.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about Raj Reddy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Echo the user's message for now (RAG / LLM coming next)
    response = prompt
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
