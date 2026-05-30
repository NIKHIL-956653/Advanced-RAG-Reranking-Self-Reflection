# Advanced RAG — Re-ranking, Self-Reflection & LLM Critique
### Production-grade RAG pipeline with automated quality control

A 4-stage Advanced RAG pipeline that retrieves historical documents,
re-ranks them for true relevance, generates documentation with
self-reflection, and validates quality through LLM critique —
acting as an inbuilt feedback loop for content quality.

## The Problem This Solves
Basic RAG:
Query → FAISS → LLM → Document
Problem: No quality control!
LLM might hallucinate!
Wrong docs retrieved!
Advanced RAG (this project):
Query → FAISS → Re-ranking → Generate
→ Self-reflection → LLM Critique
→ Approved Document! ✅

## 4-Stage Pipeline
Stage 1 — FAISS Retrieval
Query → converted to vector →
search FAISS → top 10 similar docs
Stage 2 — Re-ranking
LLM scores each retrieved doc
for TRUE relevance (0.0 to 1.0)
Returns best top 3!
Stage 3 — Self-reflection
LLM generates document
LLM checks its OWN output:
"Is everything from the context?"
"Any hallucinations?"
If issues → regenerate!
Stage 4 — LLM Critique
Second LLM scores document 1-10
Lists strengths and improvements
Score < 7 → regenerate with feedback
Score >= 7 → APPROVED! ✅

## Tech Stack

- **FAISS** — Fast vector similarity search
- **Sentence Transformers** — all-MiniLM-L6-v2 embeddings
- **OpenRouter** — Gemini 2.0 Flash (re-ranking + generation)
- **Streamlit** — Interactive pipeline UI
- **Python** — Core language
- **python-dotenv** — Environment management

## Project Structure
Advanced-RAG-Reranking-Self-Reflection/
├── embeddings.py       # Text → 384-dim vectors
├── vector_store.py     # Store/load FAISS index
├── retriever.py        # Search similar documents
├── reranker.py         # LLM-based re-ranking
├── self_reflection.py  # Generate + self-check
├── critique.py         # LLM quality gate (1-10)
├── main.py             # Full pipeline runner
├── streamlit_app.py    # Interactive UI
├── faiss_index/        # Stored vectors (auto-generated)
├── requirements.txt
├── .gitignore
└── .env                # API keys (not committed)

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/NIKHIL-956653/Advanced-RAG-Reranking-Self-Reflection.git
cd Advanced-RAG-Reranking-Self-Reflection
```

### 2. Create virtual environment
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create .env file
OPENROUTER_API_KEY=your_openrouter_key

### 5. Store sample documents in FAISS
```bash
python vector_store.py
```

## Run

### Full pipeline (terminal)
```bash
python main.py
```

### Interactive UI
```bash
streamlit run streamlit_app.py
```

## How Each Stage Works

### Stage 1 — FAISS Retrieval
```python
# Convert query to vector
query_vector = get_embedding("Build login system")
# Search FAISS for similar vectors
results = index.search(query_vector, top_k=10)
```

### Stage 2 — Re-ranking
FAISS retrieves by vector distance (fast)
Re-ranker asks LLM: "Score relevance 0-1"
Re-sort by LLM score (accurate!)
Result:
DEMO-1 Authentication → 1.0000 ✅
DEMO-5 File Upload    → 0.1000
DEMO-3 Notifications  → 0.0000

### Stage 3 — Self-reflection
LLM generates document using context
LLM reviews its own output:
"PASSED: yes
ISSUES: none
SUGGESTIONS: none"
→ Document approved! ✅

### Stage 4 — LLM Critique
Critic LLM scores the document:
"SCORE: 7
STRENGTHS: Clear structure, accurate JWT details
IMPROVEMENTS: Add error handling examples
VERDICT: APPROVED"
→ Final document delivered! ✅

## Real Results

| Query | Top Doc Retrieved | Rerank Score | Critique Score | Approved |
|-------|------------------|--------------|----------------|----------|
| Build login with JWT | User Authentication | 1.0000 | 7/10 | ✅ YES |
| Integrate payment gateway | Payment Gateway | 0.8500 | 7/10 | ✅ YES |
| Send notifications | Notification System | 0.6500 | 7/10 | ✅ YES |

## Add Your Own Documents

Use the Streamlit sidebar:
1. Enter Document ID (e.g. DEMO-6)
2. Enter Title
3. Enter Content
4. Click "Add to FAISS"

Document instantly searchable! 🚀

## Why Advanced RAG > Basic RAG

| Feature | Basic RAG | Advanced RAG |
|---------|-----------|--------------|
| Retrieval | Vector similarity | Vector + LLM re-ranking |
| Quality check | None | Self-reflection loop |
| Quality gate | None | LLM critique (min 7/10) |
| Hallucinations | Possible | Caught by self-reflection |
| Output quality | Good | Excellent! |

## Interview Answer

"We built a 4-stage Advanced RAG pipeline:
Stage 1 — FAISS retrieves top-10 similar documents.
Stage 2 — LLM re-ranker scores each for true relevance,
           returning the best top-3.
Stage 3 — Self-reflection: LLM checks its own output
           for hallucinations, regenerates if issues found.
Stage 4 — LLM critique scores 1-10, regenerates with
           feedback if below 7.
This gives consistent high-quality output with zero
hallucinations and minimum 7/10 quality score."

## Author

**Nikhil Chandra Sairam Tokala**
AI/ML Engineer | GenAI Engineer | DevOps
Dubai, UAE
[LinkedIn](https://linkedin.com/in/nikhil-chandra-133ncsr200233) |
[GitHub](https://github.com/NIKHIL-956653)
