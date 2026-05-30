import requests
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def generate_document(query: str, context_docs: list) -> str:
    """Generate initial document using LLM"""
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
                    "content": "You are a senior technical writer. Write clear structured technical documentation based ONLY on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nWrite technical documentation for: {query}"
                }
            ]
        }
    )
    return response.json()["choices"][0]["message"]["content"]


def self_reflect(query: str, document: str, context_docs: list) -> dict:
    """
    LLM reviews its own output!
    Checks if every claim is supported by context.
    """
    context = "\n\n".join([
        f"[{doc['id']}] {doc['title']}:\n{doc['content']}"
        for doc in context_docs
    ])

    reflection_prompt = f"""You are a strict document reviewer.

ORIGINAL QUERY: {query}

CONTEXT PROVIDED:
{context}

GENERATED DOCUMENT:
{document}

Review the generated document carefully:
1. Is every claim supported by the provided context?
2. Are there any hallucinated facts not in context?
3. Does it fully answer the query?

Respond in this exact format:
PASSED: yes or no
ISSUES: list any issues found or "none"
SUGGESTIONS: improvements or "none"
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
                {"role": "user", "content": reflection_prompt}
            ]
        }
    )

    reflection = response.json()["choices"][0]["message"]["content"]
    passed = "passed: yes" in reflection.lower() or "passed:yes" in reflection.lower().replace(" ", "")

    return {
        "passed": passed,
        "reflection": reflection
    }


def generate_with_reflection(query: str, context_docs: list,
                              max_attempts: int = 2) -> dict:
    """
    Generate document with self-reflection loop!
    Regenerates if self-reflection fails.
    """
    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}: Generating document...")
        document = generate_document(query, context_docs)

        print(f"Attempt {attempt}: Self-reflecting...")
        result = self_reflect(query, document, context_docs)

        print(f"Self-reflection result: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")

        if result['passed']:
            print("✅ Document passed self-reflection!")
            return {
                "document": document,
                "attempts": attempt,
                "passed": True,
                "reflection": result['reflection']
            }
        else:
            print(f"❌ Issues found: {result['reflection']}")
            if attempt < max_attempts:
                print("Regenerating with feedback...")

    print("⚠️ Max attempts reached — returning best version")
    return {
        "document": document,
        "attempts": max_attempts,
        "passed": False,
        "reflection": result['reflection']
    }


if __name__ == "__main__":
    from retriever import search_similar
    from reranker import rerank_documents

    query = "Build secure login system with JWT"

    print("Step 1: Retrieving documents...")
    docs = search_similar(query, top_k=5)

    print("Step 2: Re-ranking...")
    reranked = rerank_documents(query, docs, top_k=3)

    print("Step 3: Generating with self-reflection...")
    result = generate_with_reflection(query, reranked)

    print(f"\n{'='*60}")
    print(f"Final document (attempts: {result['attempts']}):")
    print(f"Self-reflection: {'✅ PASSED' if result['passed'] else '❌ FAILED'}")
    print(f"{'='*60}")
    print(result['document'][:500] + "...")