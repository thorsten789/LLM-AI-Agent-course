import pickle
import random
import re
import time
import numpy as np
import openai
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
import getpass
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE_FILE = os.path.join(BASE_DIR, "source_text_vectorstore.pkl")


def retrieve_source_text_for_query(query: str) -> dict | None:
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore is None:
        return None

    texts = vectorstore.get("texts", [])
    if not texts:
        return None

    embeddings = vectorstore.get("embeddings")
    if embeddings is None:
        return {"text": texts[-1], "title": vectorstore.get("titles", ["Unknown source"])[-1]}

    query_embedding = embed_text(query)
    index_embeddings = np.asarray(embeddings, dtype=np.float32)
    if index_embeddings.ndim == 1:
        index_embeddings = index_embeddings.reshape(1, -1)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)

    dot_products = index_embeddings.dot(query_embedding.T).ravel()
    query_norm = np.linalg.norm(query_embedding)
    index_norms = np.linalg.norm(index_embeddings, axis=1)
    similarities = dot_products / (index_norms * query_norm + 1e-12)
    best_index = int(np.argmax(similarities))
    return {
        "text": texts[best_index],
        "title": vectorstore.get("titles", ["Unknown source"])[best_index],
    }


def build_retrieval_query(text: str) -> str:
    words = re.findall(r"\b[\wäöüÄÖÜß]+\b", text.lower())
    unique_words = sorted(set(words))

    if not unique_words:
        return text.strip()

    if len(unique_words) < 10:
        selected_words = unique_words
    else:
        sample_size = max(4, min(20, int(len(unique_words) * 0.05)))
        selected_words = random.sample(unique_words, sample_size)

    return " ".join(selected_words)


@tool
def new_text_tool(query: str) -> str:
    """Return a new source-text selection and chapter title based on the previous quote."""
    print("Called new_text_tool.\n")
    retrieval_query = build_retrieval_query(query)
    result = retrieve_source_text_for_query(retrieval_query)
    if result is None:
        return "No source text available."
    return f"Chapter: {result['title']}\n{result['text']}"


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

    best_result = retrieve_source_text_for_query(query)
    if best_result is None:
        return None, None

    return best_result["title"], best_result["text"]

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

1. Return only verbatim passages from the Source text. Do not add, paraphrase, summarize, explain, modify, or invent anything.
2. Each passage must be an exact excerpt from the Source text and must not be longer than 400 characters. It must be continuous and complete, with no omissions, ellipses, or jumps inside the passage.
3. You may include up to three passages in one answer. If you use more than one, each new passage must come from a different chapter. A new passage is allowed only after you finish the current passage and call new_text_tool once before selecting the next one.
4. For each passage, use the most recently received source text. This may be the initial text shown below or a later text returned by the tool. Do not use older text, earlier conversation context, or a merged combination of chunks. Keep each source text block separate.
5. Do not reuse a sentence that was already used in this answer. If a tool result repeats a source text you already used in this answer, choose a different passage from the latest available source text. If that is not possible, stop there and do not add anything else.
6. Before you add a second or third passage, call new_text_tool once for each additional passage. Do this immediately after choosing the previous passage and before choosing the next one.
7. When you call new_text_tool, pass the previous quote as a plain string with no chapter label, no commentary, and no surrounding explanation. The tool will return a new source selection based on that quote. The tool result contains a chapter header followed by the actual text. Use only the text after the header as the relevant source text for the next passage.
8. Keep every quoted passage distinct. Never repeat a sentence or passage within the same answer.
9. The response must consist exclusively of the selected exact quote(s). Do not include any prose, interpretation, filler text, or chapter labels inside the quote content.
10. Do not use the marker "[...]" or any other placeholder. Do not write chapter attribution inside the response body. The application will render the chapter origin automatically after each quote paragraph.
11. Put each quote in its own paragraph and separate multiple quotes with a blank line.
12. Always choose the most fitting quote from the selected Source text for the user's latest request. If no suitable passage is found, do not answer with a placeholder such as "No suitable passage found."

# Relevant source text
The relevant source text for your current step is determined as follows:
- For the first passage in this answer, it is the text shown below.
- After you call new_text_tool, the most recent tool result becomes the relevant source text for the next passage.
Use only that relevant source text for your next passage selection. For the first passage, it is the text shown below. For later passages, it is the most recent tool result, from its first character to its last character. Do not use any earlier chunk, any earlier message, or any merged combination of chunks.
This is the relevant source text for the first passage in this answer:

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


def split_quote_blocks(content: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", content.strip()) if block.strip()]


def infer_quote_label(quote: str, fallback_label: str | None = None) -> str:
    vectorstore = st.session_state.get("vectorstore")
    if vectorstore:
        titles = vectorstore.get("titles", [])
        texts = vectorstore.get("texts", [])
        for title, source_text in zip(titles, texts):
            if quote and quote in source_text:
                return title

    retrieved = retrieve_source_text_for_query(quote)
    if retrieved and retrieved.get("title"):
        return retrieved["title"]

    return fallback_label or ""


def build_quote_labels_for_content(content: str, fallback_label: str | None = None) -> list[str]:
    if content.startswith("Error:"):
        return []

    blocks = split_quote_blocks(content)
    if not blocks:
        return []

    return [infer_quote_label(block, fallback_label=fallback_label) for block in blocks]


def render_assistant_message(message):
    content = message.get("content", "")
    quote_labels = message.get("quote_labels", [])

    if not quote_labels:
        st.markdown(content)
        return

    blocks = split_quote_blocks(content)
    if not blocks:
        st.markdown(content)
        return

    for i, block in enumerate(blocks):
        st.markdown(block)
        if i < len(quote_labels):
            label = quote_labels[i]
            if label:
                st.markdown(f"*(Aus dem Kapitel: {label})*")
        if i < len(blocks) - 1:
            st.write("")


def get_openai_response(system_prompt: str, history_messages, initial_quote_label: str | None = None):
    try:
        chain_messages = [SystemMessage(content=system_prompt)] + history_messages
        bound_model = client.bind_tools([new_text_tool])
        quote_labels = [initial_quote_label] if initial_quote_label else []

        for _ in range(3):
            response = bound_model.invoke(chain_messages)
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls is None:
                tool_calls = getattr(response, "additional_kwargs", {}).get("tool_calls", [])

            if not tool_calls:
                return response.content.strip(), quote_labels

            chain_messages.append(response)
            for tool_call in tool_calls:
                tool_name = getattr(tool_call, "name", None) or (tool_call.get("name") if isinstance(tool_call, dict) else None)
                tool_id = getattr(tool_call, "id", None) or (tool_call.get("id") if isinstance(tool_call, dict) else None)
                if tool_name != "new_text_tool":
                    chain_messages.append(
                        ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id, name=tool_name)
                    )
                    continue

                tool_args = getattr(tool_call, "args", {}) or {}
                if not isinstance(tool_args, dict):
                    tool_args = {"query": str(tool_args or "")}
                tool_output = new_text_tool.invoke({"query": tool_args.get("query", "")})
                chain_messages.append(
                    ToolMessage(content=str(tool_output), tool_call_id=tool_id, name=tool_name)
                )
                if isinstance(tool_output, str):
                    first_line = tool_output.splitlines()[0].strip() if tool_output else ""
                    if first_line.startswith("Chapter:"):
                        quote_labels.append(first_line.replace("Chapter:", "", 1).strip())

        return response.content.strip(), quote_labels
    except Exception as e:
        return f"Error: {e}", quote_labels if 'quote_labels' in locals() else []


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
        if message.get("role") == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

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
        assistant_response, _ = get_openai_response(
            system_prompt,
            history_messages,
            source_title,
        )
        quote_labels = build_quote_labels_for_content(assistant_response, fallback_label=source_title)
        render_assistant_message({
            "role": "assistant",
            "content": assistant_response,
            "quote_labels": quote_labels,
        })
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_response,
        "quote_labels": quote_labels,
    })
