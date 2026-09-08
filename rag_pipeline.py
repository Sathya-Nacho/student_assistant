# rag_pipeline.py
# RAG pipeline for College Student Assistant

from dotenv import load_dotenv
import os
import numpy as np
import google.generativeai as genai
from data.college_data import college_documents

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

EMBED_MODEL = "models/gemini-embedding-001"


def get_embedding(text: str) -> np.ndarray:
    """Create an embedding for the given text."""
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    return np.array(result["embedding"])


def build_index(documents):
    """Create embeddings for all college documents."""
    print(f"Building embeddings for {len(documents)} chunks...")

    for doc in documents:
        doc["embedding"] = get_embedding(doc["text"])

    print("Done.")
    return documents


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def retrieve(query: str, documents, top_k: int = 3):
    """Retrieve the most relevant college documents."""
    query_emb = get_embedding(query)

    scored = [
        (
            cosine_similarity(query_emb, doc["embedding"]),
            doc
        )
        for doc in documents
    ]

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        doc for score, doc in scored[:top_k]
    ]


def generate_answer(query, results):
    """Generate an answer using retrieved college information."""

    context = "\n".join(
        [doc["text"] for doc in results]
    )

    prompt = f"""
You are a helpful college student assistant.

Answer the user's question using ONLY the college
information provided below.

College information:
{context}

User question:
{query}

Give a clear, simple and helpful answer.
If the information is not available in the college
information, say that the information is not available.
"""

    model = genai.GenerativeModel("gemini-3.6-flash")

    response = model.generate_content(prompt)

    return response.text


if __name__ == "__main__":

    # Step 1: Build embeddings
    indexed_docs = build_index(college_documents)

    # Step 2: Ask multiple questions
    print("\nCollege Student Assistant")
    print("Type 'exit' to stop the program.")

    while True:

        test_query = input("\nAsk your college question: ")

        if test_query.lower() == "exit":
            print("\nThank you!")
            break

        # Step 3: Retrieve relevant documents
        results = retrieve(
            test_query,
            indexed_docs,
            top_k=2
        )

        # Step 4: Generate answer
        answer = generate_answer(
            test_query,
            results
        )

        print("\nAI Assistant Answer:")
        print(answer)