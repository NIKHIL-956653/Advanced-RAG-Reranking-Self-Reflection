import faiss
import numpy as np
import pickle
import os
from embeddings import get_embedding, get_embeddings_batch

FAISS_INDEX_PATH = "faiss_index/index.faiss"
DOCUMENTS_PATH = "faiss_index/documents.pkl"

def create_index():
    os.makedirs("faiss_index", exist_ok=True)
    dimension = 384
    return faiss.IndexFlatL2(dimension)

def save_index(index, documents):
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "wb") as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents!")

def load_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None, []
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)
    return index, documents

def add_documents(documents: list):
    if os.path.exists(FAISS_INDEX_PATH):
        index, existing_docs = load_index()
    else:
        index = create_index()
        existing_docs = []
    texts = [doc["content"] for doc in documents]
    embeddings = get_embeddings_batch(texts)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)
    existing_docs.extend(documents)
    save_index(index, existing_docs)
    return index, existing_docs

if __name__ == "__main__":
    sample_docs = [
        {"id": "DEMO-1", "title": "User Authentication",
         "content": "Implement secure login with JWT tokens, bcrypt password hashing, role-based access control, FastAPI endpoints, MongoDB storage."},
        {"id": "DEMO-2", "title": "Payment Gateway",
         "content": "Integrate Stripe payment gateway with webhook support, subscription billing, payment confirmation emails, FastAPI, PostgreSQL."},
        {"id": "DEMO-3", "title": "Notification System",
         "content": "Build WebSocket real-time notifications, Redis pub/sub, push notifications, email alerts, FastAPI WebSockets."},
        {"id": "DEMO-4", "title": "Search Feature",
         "content": "Implement full-text search with Elasticsearch, fuzzy search, faceted filtering, search suggestions, FastAPI."},
        {"id": "DEMO-5", "title": "File Upload Storage",
         "content": "Secure file upload with AWS S3, file validation, virus scanning, CDN delivery, FastAPI, boto3."},
    ]
    print("Adding documents to FAISS...")
    index, docs = add_documents(sample_docs)
    print(f"Total: {index.ntotal} documents stored!")