import os
import streamlit as st
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from textwrap import wrap


# -----------------------------
# Groq API Setup
# -----------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# -----------------------------
# Extract PDF Text
# -----------------------------
def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        content = page.extract_text()

        if content:
            text += content + "\n"

    return text



# -----------------------------
# Chunking
# -----------------------------
def create_chunks(text):

    return wrap(
        text,
        width=700
    )



# -----------------------------
# Create FAISS Index
# -----------------------------
def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index



# -----------------------------
# Retrieve Relevant Context
# -----------------------------
def search_context(index, chunks, question, k=4):

    query = embedding_model.encode(
        [question]
    )

    query = np.array(query).astype("float32")


    distances, ids = index.search(
        query,
        k
    )


    results = []

    for idx in ids[0]:

        if idx < len(chunks):
            results.append(
                chunks[idx]
            )

    return "\n\n".join(results)



# -----------------------------
# Groq Answer
# -----------------------------
def ask_groq(question, context):

    prompt = f"""
You are a PDF RAG assistant.

Answer only from the PDF context.

PDF Context:
{context}

Question:
{question}

If the answer is not present in the PDF, say:
"I could not find this information in the uploaded PDF."
"""


    try:

        response = client.chat.completions.create(

            model="openai/gpt-oss-120b",

            messages=[
                {
                    "role": "system",
                    "content": "Answer using retrieved PDF information."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2
        )

        return response.choices[0].message.content


    except Exception as e:

        return f"Groq API Error: {str(e)}"



# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄"
)


st.title("📄 Groq + FAISS PDF RAG Chatbot")


if not os.environ.get("GROQ_API_KEY"):

    st.error(
        "GROQ_API_KEY is missing. Add it in Streamlit Secrets."
    )

    st.stop()



uploaded_pdf = st.file_uploader(
    "Upload PDF file",
    type=["pdf"]
)


if uploaded_pdf:

    with st.spinner("Processing PDF..."):

        text = extract_pdf_text(
            uploaded_pdf
        )

        if not text.strip():

            st.error(
                "No readable text found in this PDF."
            )

            st.stop()


        chunks = create_chunks(
            text
        )

        index = create_vector_database(
            chunks
        )


    st.success(
        f"PDF processed successfully. {len(chunks)} chunks created."
    )


    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    if question:

        with st.chat_message("user"):
            st.write(question)


        with st.spinner("Generating answer..."):

            context = search_context(
                index,
                chunks,
                question
            )

            answer = ask_groq(
                question,
                context
            )


        with st.chat_message("assistant"):
            st.write(answer)


else:

    st.info(
        "Upload a PDF to start chatting."
    )
