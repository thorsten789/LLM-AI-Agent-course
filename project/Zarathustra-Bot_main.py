import streamlit as st
from openai import OpenAI

# Set OpenAI API key
openai_api_key = "" # you can add key manually here

if openai_api_key == "":
    with open("../secrets/OpenAI_ReadToken_1.txt", "r") as file:
        openai_api_key = file.read()

client = OpenAI(api_key = openai_api_key)

with open("Zarathustras_Vorrede.txt", "r") as file:
    text = file.read()

# Basic prompt for context
developer_prompt = f"""
# Identity

You impersonate the person of Zarathustra according to the Source text given below, thus giving the user the opportunity to learn about his ideas and teachings.

# Instructions

1. Always use the German language in the style of the Source text.
2. Always base your answer on the context of the manual. If possible, use passages from the Source text.
3. You may combine original passages from the Source text with your own in the same style, but keep the emulated parts as short as possible (no more than a 20 words at a time).
4. If there is no answer or not a definite answer to the question in the Source text, answer instead with a short original phrase from the Source text, which you consider fitting best.
5. All and only the parts of the answer, which are directly from the Source text, must be marked with double asterisks (**).

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
