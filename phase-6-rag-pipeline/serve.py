# 선택: pip install fastapi uvicorn 후 'uvicorn serve:app --reload' 로 실행.

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI

import documents
import pipeline


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "bge-m3"
COLLECTION_NAME = "phase6_support"
MAX_SIZE = 90
TOP_K = 3


def build_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_URL,
        model_name=EMBED_MODEL,
    )
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )
    pipeline.ingest(collection, documents.DOC, MAX_SIZE)
    return collection


app = FastAPI()
collection = build_collection()


@app.get("/ask")
def ask_get(q: str):
    return pipeline.answer(collection, q, TOP_K)


@app.post("/ask")
def ask_post(q: str):
    return pipeline.answer(collection, q, TOP_K)