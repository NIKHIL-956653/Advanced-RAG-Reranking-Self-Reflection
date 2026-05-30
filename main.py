import os
import json
from dotenv import load_dotenv
from retriever import search_similar
from reranker import rerank_documents
from self_reflection import generate_with_reflection
from critique import generate_with_critique

load_dotenv()

def run_advanced_rag_pipeline(query: str) -> dict:
    """
    Full Advanced RAG Pipeline:
    1. FAISS retrieval
    2. Re-ranking
    3. Generate + Self-reflection
    4. LLM Critique
    """
    print(f"\n{'='*60}")
    print(f"🚀 ADVANCED RAG PIPELINE")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Step 1 — FAISS Retrieval
    print("\n📚 Step 1: FAISS Retrieval (top 10)...")
    docs = search_similar(query, top_k=5)
    print(f"Retrieved {len(docs)} documents")

    # Step 2 — Re-ranking
    print("\n🔄 Step 2: Re-ranking...")
    reranked = rerank_documents(query, docs, top_k=3)

    # Step 3 — Generate + Self-reflection
    print("\n🤔 Step 3: Generate + Self-reflection...")
    reflection_result = generate_with_reflection(
        query, reranked, max_attempts=2
    )

    # Step 4 — LLM Critique
    print("\n🎯 Step 4: LLM Critique...")
    critique_result = generate_with_critique(
        query,
        reflection_result['document'],
        reranked,
        max_attempts=2
    )

    # Final result
    final_result = {
        "query": query,
        "retrieved_docs": [d['title'] for d in reranked],
        "rerank_scores": [d['rerank_score'] for d in reranked],
        "self_reflection_passed": reflection_result['passed'],
        "reflection_attempts": reflection_result['attempts'],
        "critique_score": critique_result['score'],
        "critique_approved": critique_result['approved'],
        "critique_attempts": critique_result['attempts'],
        "final_document": critique_result['document']
    }

    # Save output
    filename = f"output_{query[:20].replace(' ', '_')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {query}\n\n")
        f.write(f"**Critique Score:** {critique_result['score']}/10\n\n")
        f.write(f"**Self-reflection:** {'✅ Passed' if reflection_result['passed'] else '⚠️ Max attempts'}\n\n")
        f.write("---\n\n")
        f.write(critique_result['document'])

    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE!")
    print(f"📚 Retrieved: {[d['title'] for d in reranked]}")
    print(f"🤔 Self-reflection: {'✅ PASSED' if reflection_result['passed'] else '⚠️ MAX ATTEMPTS'}")
    print(f"🎯 Critique Score: {critique_result['score']}/10")
    print(f"✅ Approved: {'YES' if critique_result['approved'] else 'NO'}")
    print(f"💾 Saved: {filename}")
    print(f"{'='*60}")

    return final_result


if __name__ == "__main__":
    queries = [
        "Build secure login system with JWT",
        "Integrate payment gateway with webhooks",
    ]

    for query in queries:
        result = run_advanced_rag_pipeline(query)
        print(f"\nSummary for '{query}':")
        print(f"  Score: {result['critique_score']}/10")
        print(f"  Approved: {result['critique_approved']}")