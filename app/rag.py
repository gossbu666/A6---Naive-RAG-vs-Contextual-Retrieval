"""
RAG backend logic — shared by the Flask app.
Retriever : BAAI/bge-small-en-v1.5 (local, sentence-transformers)
Generator : llama-3.1-8b-instant (Groq free tier)
"""
import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load .env from project root (one level up from app/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GEN_MODEL        = "llama-3.1-8b-instant"
CHROMA_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")

embed_model = SentenceTransformer(EMBED_MODEL_NAME)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    contextual_collection = chroma_client.get_collection("contextual_rag")
    print(f"✅ Loaded contextual_rag ({contextual_collection.count()} chunks)")
except Exception as e:
    contextual_collection = None
    print(f"⚠️  contextual_rag not found — run the notebook first. ({e})")


def get_embedding(text: str) -> list:
    return embed_model.encode(text, normalize_embeddings=True).tolist()


def retrieve(query: str, top_k: int = 3) -> list:
    if contextual_collection is None:
        return []
    q_emb = get_embedding(query)
    results = contextual_collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "distances"]
    )
    chunks = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        chunks.append({"text": doc, "score": round(1 - dist, 4)})
    return chunks


def generate_answer(question: str, context_chunks: list) -> str:
    context = "\n\n".join(
        [f"[Source {i+1} (similarity {c['score']:.3f})]:\n{c['text']}"
         for i, c in enumerate(context_chunks)]
    )
    prompt = f"""You are a helpful assistant for students studying Natural Language Processing.
Answer the question using ONLY the provided context. Be concise and accurate.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}
Answer:"""

    response = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400
    )
    return response.choices[0].message.content.strip()


def ask(question: str, top_k: int = 3) -> dict:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return {
            "answer": "⚠️ Vector database not loaded. Please run the notebook first.",
            "sources": []
        }
    answer = generate_answer(question, chunks)
    return {"answer": answer, "sources": chunks}
