import os
import streamlit as st
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# Groq API
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
# Load Documents
# -----------------------------
def load_documents():

    documents = []
    folder = "documents"

    if not os.path.exists(folder):
        return documents

    for filename in os.listdir(folder):

        filepath = os.path.join(folder, filename)

        if filename.endswith(".txt"):

            with open(filepath, "r", encoding="utf-8") as file:
                documents.append(file.read())

    return documents



# -----------------------------
# Chunking
# -----------------------------
def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = []

    for document in documents:
        chunks.extend(
            splitter.split_text(document)
        )

    return chunks



# -----------------------------
# FAISS Vector Database
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
# Retrieve Context
# -----------------------------
def retrieve_context(question, k=3):

    if index is None:
        return "No documents found."

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
# Groq Generation
# -----------------------------
def generate_answer(question, context):

    prompt = f"""
You are a RAG AI assistant.

Use only the context below to answer.

Context:
{context}

User Question:
{question}

If the answer is not present in the context,
reply:
"I don't have enough information."
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": "Answer accurately using retrieved information."
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
# Streamlit Interface
# -----------------------------
st.set_page_config(
    page_title="Groq RAG Chatbot",
    page_icon="🤖"
)


st.title("🤖 Groq + FAISS RAG Chatbot")


if not os.environ.get("GROQ_API_KEY"):
    st.warning(
        "Please add GROQ_API_KEY in Streamlit Secrets."
    )


question = st.chat_input(
    "Ask your question..."
)


if question:

    with st.chat_message("user"):
        st.write(question)


    with st.spinner("Searching knowledge base..."):

        context = retrieve_context(
            question
        )

        answer = generate_answer(
            question,
            context
        )


    with st.chat_message("assistant"):
        st.write(answer)
