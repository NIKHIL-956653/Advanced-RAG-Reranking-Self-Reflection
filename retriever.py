import faiss
import numpy as np
from vector_store import load_index
from embeddings import get_embedding

def search_similar(query: str, top_k: int = 10) -> list:
    index, documents = load_index()
    if not index:
        return []

    query_embedding = get_embedding(query)
    query_embedding = np.array([query_embedding]).astype('float32')

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            doc = documents[idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "similarity_score": float(1 / (1 + distances[0][i])),
                "distance": float(distances[0][i])
            })
    return results