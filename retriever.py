import numpy as np
from embeddings import get_embedding
from vector_store import load_index


def search_similar(query: str, top_k: int = 5) -> list:
    """FAISS similarity search — returns top_k documents for the query."""
    index, documents = load_index()
    if index is None or index.ntotal == 0:
        raise RuntimeError(
            "FAISS index is empty. Run: python vector_store.py"
        )

    query_vec = np.array([get_embedding(query)]).astype("float32")
    k = min(top_k, index.ntotal)
    distances, indices = index.search(query_vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        doc = dict(documents[idx])
        doc["distance"] = float(dist)
        doc["score"] = 1.0 / (1.0 + float(dist))
        results.append(doc)
    return results


if __name__ == "__main__":
    for d in search_similar("secure login with JWT", top_k=3):
        print(f"{d['score']:.3f}  {d['title']}")