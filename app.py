import os
import streamlit as st
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from textwrap import wrap


# -----------------------------
# Groq API Setup
# -----------------------------
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# -----------------------------
# Load Text Documents
# -----------------------------
def load_documents():

    documents = []
    folder = "documents"

    if not os.path.exists(folder):
        return documents

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        if filename.lower().endswith(".txt"):

            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

    return documents


# -----------------------------
# Simple Chunking
# -----------------------------
def create_chunks(documents):

    chunks = []

    for document in documents:

        document_chunks = wrap(
            document,
            width=500
        )

        chunks.extend(document_chunks)

    return chunks



# -----------------------------
# Create FAISS Database
# -----------------------------
@st.cache_resource
def create_faiss_database():

    documents = load_documents()

    if not documents:
        return None, []

    chunks = create_chunks(documents)

    vectors = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    vectors = np.array(vectors).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(vectors)

    return index, chunks



index, chunks = create_faiss_database()



# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_context(question, k=3):

    if index is None:
        return "No documents available."

    query_vector = embedding_model.encode(
        [question]
    )

    query_vector = np.array(
        query_vector
    ).astype("float32")


    distances, results = index.search(
        query_vector,
        k
    )


    context = []

    for item in results[0]:

        if item < len(chunks):
            context.append(
                chunks[item]
            )


    return "\n\n".join(context)



# -----------------------------
# Generate Answer using Groq
# -----------------------------
def generate_answer(question, context):

    prompt = f"""
You are a RAG chatbot.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not found in the context, say:
"I don't have enough information."
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "You answer from retrieved documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )


    return response.choices[0].message.content



# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Groq FAISS RAG Chatbot",
    page_icon="🤖"
)


st.title("🤖 Groq + FAISS RAG Chatbot")


if not os.environ.get("GROQ_API_KEY"):

    st.error(
        "Missing GROQ_API_KEY. Add it in Streamlit Cloud Secrets."
    )

    st.stop()


if not chunks:

    st.warning(
        "No .txt files found. Add files inside the documents folder."
    )


question = st.chat_input(
    "Ask your question..."
)


if question:

    with st.chat_message("user"):
        st.write(question)


    with st.spinner("Searching knowledge base..."):

        context = retrieve_context(question)

        answer = generate_answer(
            question,
            context
        )


    with st.chat_message("assistant"):
        st.write(answer)
