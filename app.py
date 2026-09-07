import os
import streamlit as st
import faiss
import numpy as np

from groq import Groq
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from textwrap import wrap


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
# Extract PDF Text
# -----------------------------
def extract_pdf_text(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text



# -----------------------------
# Chunking
# -----------------------------
def create_chunks(text):

    chunks = wrap(
        text,
        width=700
    )

    return chunks



# -----------------------------
# Create FAISS Index
# -----------------------------
@st.cache_resource
def create_vector_database(chunks):

    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index



# -----------------------------
# Search PDF Knowledge
# -----------------------------
def search_context(index, chunks, question, k=4):

    query_embedding = embedding_model.encode(
        [question]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    distances, results = index.search(
        query_embedding,
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
# Groq Answer
# -----------------------------
def ask_groq(question, context):

    prompt = f"""
You are a PDF RAG assistant.

Answer the user's question only from the provided PDF context.

PDF Context:
{context}

Question:
{question}

If the information is not available in the PDF, say:
"I could not find this information in the uploaded PDF."
"""


    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

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



# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄"
)


st.title("📄 Groq + FAISS PDF RAG Chatbot")


if not os.environ.get("GROQ_API_KEY"):

    st.error(
        "Please add GROQ_API_KEY in Streamlit Secrets."
    )

    st.stop()


uploaded_pdf = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_pdf:

    with st.spinner("Reading PDF and creating knowledge base..."):

        pdf_text = extract_pdf_text(
            uploaded_pdf
        )

        if pdf_text.strip():

            chunks = create_chunks(
                pdf_text
            )

            index = create_vector_database(
                chunks
            )

            st.success(
                f"PDF processed successfully. {len(chunks)} chunks created."
            )


            question = st.chat_input(
                "Ask anything about this PDF..."
            )


            if question:

                with st.chat_message("user"):
                    st.write(question)


                with st.spinner("Searching PDF..."):

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

            st.warning(
                "Could not extract text from this PDF."
            )

else:

    st.info(
        "Upload a PDF file to start chatting."
    )
