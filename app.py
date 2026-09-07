import os
import streamlit as st
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


def load_documents():
    docs = []
    folder = "documents"

    if not os.path.exists(folder):
        return []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())

    return docs


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for doc in documents:
        chunks.extend(splitter.split_text(doc))

    return chunks


@st.cache_resource
def create_vector_database():

    documents = load_documents()

    if not documents:
        return None, None

    chunks = create_chunks(documents)

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return index, chunks


index, chunks = create_vector_database()


def search_context(question, k=3):

    if index is None:
        return ""

    query_embedding = embedding_model.encode([question])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, ids = index.search(query_embedding, k)

    results = []

    for i in ids[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return "\n\n".join(results)


def ask_groq(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer only using the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in context, say:
"I don't have enough information."
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "Answer using retrieved documents."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return completion.choices[0].message.content


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖"
)

st.title("🤖 Groq + FAISS RAG Chatbot")

st.write(
    "Place your knowledge files inside the documents folder."
)

question = st.chat_input("Ask something...")

if question:

    with st.chat_message("user"):
        st.write(question)

    context = search_context(question)

    answer = ask_groq(question, context)

    with st.chat_message("assistant"):
        st.write(answer)
