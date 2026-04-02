import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Duplicate Claim Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse("static/index.html")

# Load data and embeddings on startup
data = pd.read_csv("../data/claims.csv")
data["text"] = data["title"] + " " + data["description"]

embeddings = np.load("../model/claims_embeddings.npy")

model = SentenceTransformer("all-MiniLM-L6-v2")

THRESHOLD = 0.75


@app.get("/check")
def check_duplicate(text: str = Query(..., description="Claim text to check for duplicates")):
    query_embedding = model.encode([text])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score >= THRESHOLD:
        matched_claim = data.iloc[best_idx]
        return {
            "is_duplicate": True,
            "similarity_score": round(best_score, 4),
            "matched_claim": {
                "id": int(matched_claim["id"]),
                "title": matched_claim["title"],
                "description": matched_claim["description"],
            },
        }

    return {
        "is_duplicate": False,
        "similarity_score": round(best_score, 4),
        "matched_claim": None,
    }


@app.get("/claims")
def list_claims():
    return data[["id", "title", "description"]].to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok", "total_claims": len(data)}
