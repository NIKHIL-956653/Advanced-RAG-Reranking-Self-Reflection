import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def llm_critique(query: str, document: str) -> dict:
    """
    Second LLM call acts as critic!
    Scores document 1-10 and gives feedback.
    This is the quality gate!
    """
    critique_prompt = f"""You are a senior technical documentation expert.

QUERY: {query}

DOCUMENT TO REVIEW:
{document}

Review this document and provide:
1. A score from 1-10 (10 = perfect)
2. What's good about it
3. What needs improvement

Respond in this EXACT format:
SCORE: [number 1-10]
STRENGTHS: [what is good]
IMPROVEMENTS: [what to fix]
VERDICT: APPROVED or REJECTED
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [
                {"role": "user", "content": critique_prompt}
            ]
        }
    )

    critique = response.json()["choices"][0]["message"]["content"]

    # Extract score
    score = 7  # default
    for line in critique.split('\n'):
        if line.startswith('SCORE:'):
            try:
                score = int(line.split(':')[1].strip())
            except:
                pass

    approved = score >= 7

    return {
        "score": score,
        "critique": critique,
        "approved": approved
    }


def generate_with_critique(query: str, document: str,
                            context_docs: list,
                            max_attempts: int = 2) -> dict:
    """
    Full critique loop!
    If score < 7 → regenerate with feedback
    If score >= 7 → approved!
    """
    from self_reflection import generate_document

    current_doc = document
    feedback = ""

    for attempt in range(1, max_attempts + 1):
        print(f"\nCritique attempt {attempt}...")
        result = llm_critique(query, current_doc)

        print(f"Score: {result['score']}/10")
        print(f"Verdict: {'✅ APPROVED' if result['approved'] else '❌ REJECTED'}")

        if result['approved']:
            return {
                "document": current_doc,
                "score": result['score'],
                "critique": result['critique'],
                "attempts": attempt,
                "approved": True
            }
        else:
            print(f"Score too low ({result['score']}/10) — regenerating...")
            feedback = result['critique']

            # Regenerate with feedback
            context = "\n\n".join([
                f"[{doc['id']}] {doc['title']}:\n{doc['content']}"
                for doc in context_docs
            ])

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a senior technical writer. Improve the document based on feedback."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nPrevious document:\n{current_doc}\n\nFeedback:\n{feedback}\n\nRewrite improved version for: {query}"
                        }
                    ]
                }
            )
            current_doc = response.json()["choices"][0]["message"]["content"]

    return {
        "document": current_doc,
        "score": result['score'],
        "critique": result['critique'],
        "attempts": max_attempts,
        "approved": False
    }


if __name__ == "__main__":
    from retriever import search_similar
    from reranker import rerank_documents
    from self_reflection import generate_with_reflection

    query = "Build secure login system with JWT"

    print("Step 1: Retrieve + Rerank...")
    docs = search_similar(query, top_k=5)
    reranked = rerank_documents(query, docs, top_k=3)

    print("\nStep 2: Generate + Self-reflection...")
    reflection_result = generate_with_reflection(query, reranked)
    document = reflection_result['document']

    print("\nStep 3: LLM Critique...")
    critique_result = generate_with_critique(
        query, document, reranked
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"Score: {critique_result['score']}/10")
    print(f"Approved: {'✅ YES' if critique_result['approved'] else '❌ NO'}")
    print(f"Attempts: {critique_result['attempts']}")
    print(f"{'='*60}")