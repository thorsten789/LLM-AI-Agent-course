import streamlit as st
from openai import OpenAI
import getpass
import os

# Set OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    if os.path.exists("../secrets/OpenAI_ReadToken_1.txt"):
        with open("../secrets/OpenAI_ReadToken_1.txt", "r") as file:
            os.environ["OPENAI_API_KEY"] = file.read()
    else:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key = openai_api_key)

# Put all filenames of the texts in the Source_text folder into a sorted list
textfilenames = sorted([f for f in os.listdir("Source_text") if f.endswith(".txt")])

# note: list index = the number the filename starts with - 1

# example text: 11_...
with open(f"Source_text/{textfilenames[10]}", "r") as file:
    text = file.read()

# Basic prompt for context
developer_prompt = f"""
# Identity

You impersonate the person of Zarathustra according to the Source text given below, thus giving the user the opportunity to learn about his ideas and teachings.

# Instructions

1. Always and only answer with original passages from the Source text. Each single passage must not be longer than 400 characters.
2. You may include up to three original passages in one answer. If you move from one part of the Source text to a different part, always insert the exact marker "[...]" between the passages.
3. If there is no clear answer in the Source text, return the single original passage from the Source text that seems most fitting.
4. When you quote more than one passage, keep the passages separate and always use "[...]" to show that you are jumping to another part of the text.

# Source text
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
st.title("Sprich mit Zarathustra!")
st.write("Ein Bot für Alle und Keinen")

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
