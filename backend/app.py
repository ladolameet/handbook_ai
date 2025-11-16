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

# New Google embedding model
EMBED_URL = "https://generativelanguage.googleapis.com/v1/models/text-embedding-004:embedContent"
GEN_URL   = "https://generativelanguage.googleapis.com/v1/models/gemini-2.0-flash:generateContent"

# ================================================
# Load handbook and generate embeddings in memory
# ================================================
df = pd.read_csv("handbook_final.csv")
texts = [f"Page {row['page']}: {row['chunk']}" for _, row in df.iterrows()]

def embed_text(t):
    body = {
        "model": "text-embedding-004",
        "input": t
    }
    res = requests.post(f"{EMBED_URL}?key={API_KEY}", json=body)
    data = res.json()
    return np.array(data["data"][0]["embedding"], dtype=float)

# Generate embeddings once on server start
embeddings = np.array([embed_text(t) for t in texts])

print("Embeddings loaded in memory:", embeddings.shape)

# ================================================
# FastAPI Setup
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
# RAG answer
# ================================================
def rag_answer(query):
    context = "\n".join(search_similar(query))

    prompt = f"""
Use this handbook context to answer:

Context:
{context}

Question:
{query}

Answer clearly.
"""

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    res = requests.post(f"{GEN_URL}?key={API_KEY}", json=body)

    try:
        return res.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "⚠ Gemini API error"

# ================================================
# Endpoint
# ================================================
@app.post("/chat")
async def chat(data: Query):
    return {"answer": rag_answer(data.query)}
