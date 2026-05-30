import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def rerank_documents(query: str, documents: list, top_k: int = 3) -> list:
    """
    Re-rank retrieved documents using LLM scoring.
    Better than raw vector similarity!
    """
    if not documents:
        return []

    print(f"\nRe-ranking {len(documents)} documents...")

    scored_docs = []

    for doc in documents:
        prompt = f"""Score how relevant this document is to the query.
        
Query: {query}

Document Title: {doc['title']}
Document Content: {doc['content']}

Score from 0.0 to 1.0 where:
1.0 = perfectly relevant
0.0 = completely irrelevant

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 10
                }
            )
            score_text = response.json()["choices"][0]["message"]["content"].strip()
            score = float(score_text)
            score = max(0.0, min(1.0, score))
        except:
            score = doc['similarity_score']

        scored_docs.append({
            **doc,
            "rerank_score": score
        })
        print(f"  [{doc['id']}] {doc['title']} → Rerank Score: {score:.4f}")

    # Sort by rerank score
    scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

    print(f"\nTop {top_k} after re-ranking:")
    for doc in scored_docs[:top_k]:
        print(f"  ✅ [{doc['id']}] {doc['title']} → {doc['rerank_score']:.4f}")

    return scored_docs[:top_k]


if __name__ == "__main__":
    from retriever import search_similar

    query = "Build secure login system with JWT"

    print("Step 1: FAISS retrieval (top 5)...")
    docs = search_similar(query, top_k=5)
    for d in docs:
        print(f"  [{d['id']}] {d['title']} → {d['similarity_score']:.4f}")

    print("\nStep 2: Re-ranking...")
    reranked = rerank_documents(query, docs, top_k=3)

    print("\nFinal Result:")
    for d in reranked:
        print(f"  🏆 [{d['id']}] {d['title']} → {d['rerank_score']:.4f}")
        