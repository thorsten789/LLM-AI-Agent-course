import pickle
import time
import numpy as np
import openai
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
import getpass
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_FILE = os.path.join(BASE_DIR, "source_text_vectorstore.pkl")


def embed_text(text, max_retries=3, backoff_seconds=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=text,
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds * attempt)


def extract_title(text: str) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("# ").strip()
    return first_line


def build_vectorstore(textfilenames):
    texts = []
    titles = []
    embeddings = []
    for fn in textfilenames:
        path = os.path.join(BASE_DIR, "Source_text", fn)
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        texts.append(text)
        titles.append(extract_title(text))
        embeddings.append(embed_text(text))
    embeddings = np.vstack(embeddings)
    vectorstore = {
        "filenames": textfilenames,
        "titles": titles,
        "texts": texts,
        "embeddings": embeddings,
    }
    with open(VECTORSTORE_FILE, "wb") as file:
        pickle.dump(vectorstore, file)
    print(f"Saved {len(embeddings)} embeddings as new vectorstore.\n")
    return vectorstore


def load_vectorstore(textfilenames):
    if not os.path.exists(VECTORSTORE_FILE):
        return None
    with open(VECTORSTORE_FILE, "rb") as file:
        vectorstore = pickle.load(file)
    if vectorstore.get("filenames") != textfilenames:
        return None
    print(f"Loaded {vectorstore['embeddings'].shape[0]} embeddings from existing vectorstore.\n")
    return vectorstore


def select_best_source(query):
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return None, None
    query_embedding = embed_text(query)
    index_embeddings = vectorstore["embeddings"]
    dot_products = index_embeddings.dot(query_embedding)
    query_norm = np.linalg.norm(query_embedding)
    index_norms = np.linalg.norm(index_embeddings, axis=1)
    similarities = dot_products / (index_norms * query_norm + 1e-12)
    best_index = int(np.argmax(similarities))
    return (
        vectorstore["titles"][best_index],
        vectorstore["texts"][best_index],
    )

# One-time initialization for this Streamlit session
if "app_initialized" not in st.session_state:
    if "OPENAI_API_KEY" not in os.environ:
        if os.path.exists("../secrets/OpenAI_ReadToken_1.txt"):
            with open("../secrets/OpenAI_ReadToken_1.txt", "r") as file:
                os.environ["OPENAI_API_KEY"] = file.read()
        else:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

    openai_api_key = os.environ["OPENAI_API_KEY"]
    openai.api_key = openai_api_key
    st.session_state["client"] = init_chat_model(
        "openai:gpt-4.1-mini",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    # Put all filenames of the texts in the Source_text folder into a sorted list
    textfilenames = sorted([f for f in os.listdir(os.path.join(BASE_DIR, "Source_text")) if f.endswith(".txt")])
    print(f"Found {len(textfilenames)} text files in Source_text folder.\n")

    # note: list index = the number the filename begins with - 1

    # Load or build the source-text vectorstore once at startup.
    vectorstore = load_vectorstore(textfilenames)
    if vectorstore is None:
        vectorstore = build_vectorstore(textfilenames)
    st.session_state["vectorstore"] = vectorstore

    # start with text 01
    text = vectorstore["texts"][0]

    # Basic prompt template for context
    prompt_template = """# Identity

You impersonate the person of Zarathustra according to the Source text given below, thus giving the user the opportunity to learn about his ideas and teachings.

# Instructions

1. Always answer only with original passages from the Source text. Do not add, paraphrase, summarize, modify, or invent any words or ideas.
2. Each single passage must be a verbatim excerpt from the Source text and must not be longer than 400 characters.
3. You may include up to three original passages in one answer. If you move from one part of the Source text to a different part, always insert the exact marker "[...]" between the passages.
4. Do not write any explanation, interpretation, or filler text. The response must consist exclusively of the selected exact quote(s) from the Source text.
5. Always select the best-fitting original quote from the chosen Source text for the new user input. Under these rules, the answer must still be provided using the selected text; do not respond with "No suitable passage found."

# The Source text to start with is the following:
{source_text}"""

    st.session_state["developer_prompt_template"] = prompt_template
    st.session_state["developer_prompt"] = prompt_template.format(source_text=text)

    st.session_state["openai_model"] = "gpt-4.1-mini"
    st.session_state["messages"] = [{"role": "developer", "content": st.session_state["developer_prompt"]}]
    st.session_state["app_initialized"] = True

client = st.session_state["client"]
developer_prompt = st.session_state["developer_prompt"]


def build_system_prompt(source_text: str) -> str:
    prompt_template = st.session_state["developer_prompt_template"]
    return prompt_template.format(source_text=source_text)


def build_history_messages(messages):
    role_map = {
        "system": SystemMessage,
        "developer": SystemMessage,
        "user": HumanMessage,
        "assistant": AIMessage,
    }
    return [
        role_map.get(m["role"], HumanMessage)(content=m["content"])
        for m in messages
        if m["role"] != "developer"
    ]


def get_openai_response(system_prompt: str, history_messages):
    try:
        chain_messages = [SystemMessage(content=system_prompt)] + history_messages
        response = client.invoke(chain_messages)
        return response.content.strip()
    except Exception as e:
        return f"Error: {e}"


# Streamlit UI
st.title("Sprich mit Zarathustra")
st.write("Ein Bot für Alle und Keinen")

# Set a default model if needed for older sessions
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4.1-mini"

# Initialize chat history if needed for older sessions
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "developer", "content": developer_prompt}]

# Display chat messages from history on app rerun
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("role") == "assistant" and "source_title" in message:
            st.markdown(f"*(Aus dem Kapitel: {message['source_title']})*")

# Accept user input
if prompt := st.chat_input("Eingabe:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    source_title, source_text = select_best_source(prompt)
    if source_title is None:
        source_text = st.session_state["developer_prompt"]
        source_title = "Unknown source"
    system_prompt = build_system_prompt(source_text)
    history_messages = build_history_messages(st.session_state.messages)

    with st.chat_message("assistant"):
        assistant_response = get_openai_response(system_prompt, history_messages)
        st.markdown(assistant_response)
        st.markdown(f"*(Aus dem Kapitel: {source_title})*")
    st.session_state.messages.append({"role": "assistant", "content": assistant_response, "source_title": source_title})
