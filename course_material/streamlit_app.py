import streamlit as st
import openai
import os
from dotenv import load_dotenv
from pathlib import Path
import PyPDF2
from openai import OpenAI

# Load OpenAI API key on Streamlit Cloud
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    # Check current directory and up to two parent directories for .env
    for path in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
        dotenv_file = path / ".env"
        if dotenv_file.exists():
            load_dotenv(dotenv_file)
            break

client = OpenAI()

with open("./UNOClassicManualGerman.pdf", "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

# Basic prompt for context
developer_prompt = f"""
# Identity

You are a game assistant for the UNO game and answer questions from the user about rules of game.

# Instructions

1. Only answer based on the context of the manual.
2. If there is no answer or not a definite answer to the question in the manual, say it so
3. Answer always in the language of the user question

# Manual
{text}
"""


def get_openai_response(messages):
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini", messages=messages, temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# Streamlit UI
st.title("UNO Manual Chatbot 🧠🎴")
st.write("Ask me anything about UNO rules!")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4.1-mini"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "developer", "content": developer_prompt}]

# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Your question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
