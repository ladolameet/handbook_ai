from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import requests
import os
from dotenv import load_dotenv

# ================================================
# Load API Key
# ================================================
load_dotenv("google_key.env")
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise Exception("❌ GOOGLE_API_KEY not found in google_key.env")

# Google API URLs
EMBED_URL = f"https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedText?key={API_KEY}"
GEN_URL   = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# ================================================
# Load handbook
# ================================================
df = pd.read_csv("handbook_final.csv")
texts = [f"Page {row['page']}: {row['chunk']}" for _, row in df.iterrows()]

# ================================================
# Helper: Generate embedding for text
# ================================================
def embed_text(t):
    body = {
        "model": "text-embedding-004",
        "input": {"text": t}
    }

    res = requests.post(EMBED_URL, json=body)
    data = res.json()

    # Debug: print API errors
    if "embedding" not in data:
        print("❌ EMBED ERROR:", data)
        raise Exception("Embedding API Error")

    return np.array(data["embedding"]["values"], dtype=float)

# ================================================
# Build embeddings IN MEMORY (no npy file)
# ================================================
print("⏳ Generating embeddings (first-time only)...")

embeddings = np.array([embed_text(t) for t in texts])

print("✅ Embeddings ready:", embeddings.shape)

# ================================================
# FastAPI setup
# ================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# ================================================
# Find similar chunks
# ================================================
def search_similar(query, k=4):
    q_emb = embed_text(query)
    sims = embeddings @ q_emb / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    )
    top = sims.argsort()[-k:][::-1]
    return [texts[i] for i in top]

# ================================================
# RAG Answer
# ================================================
def rag_answer(query):
    context = "\n".join(search_similar(query))

    prompt = f"""
Use the following handbook context to answer the question:

Context:
{context}

Question:
{query}

Answer clearly and correctly.
"""

    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    res = requests.post(GEN_URL, json=body)
    data = res.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        print("❌ GEMINI ERROR:", data)
        return "⚠ Gemini API Error"

# ================================================
# API endpoint
# ================================================
@app.post("/chat")
async def chat(data: Query):
    return {"answer": rag_answer(data.query)}
