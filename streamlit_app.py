import streamlit as st
from retriever import search_similar
from reranker import rerank_documents
from self_reflection import generate_with_reflection
from critique import generate_with_critique

st.set_page_config(
    page_title="Advanced RAG Pipeline",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Advanced RAG Pipeline")
st.caption("Re-ranking + Self-reflection + LLM Critique")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Pipeline Settings")
    top_k_retrieve = st.slider("FAISS Retrieve (top-k)", 3, 10, 5)
    top_k_rerank = st.slider("After Re-ranking (top-k)", 1, 5, 3)
    max_attempts = st.slider("Max Reflection Attempts", 1, 3, 2)
    st.divider()
    st.markdown("""
    ### Pipeline Steps:
    1. 📚 FAISS Retrieval
    2. 🔄 Re-ranking
    3. 🤔 Self-reflection
    4. 🎯 LLM Critique
    """)
    st.divider()
    st.header("➕ Add Document to FAISS")
    new_id = st.text_input("Document ID", placeholder="DEMO-6")
    new_title = st.text_input("Document Title", placeholder="Feature name")
    new_content = st.text_area("Document Content", placeholder="Describe the feature...")
    if st.button("Add to FAISS"):
        if new_id and new_title and new_content:
            from vector_store import add_documents
            add_documents([{
                "id": new_id,
                "title": new_title,
                "content": new_content
            }])
            st.success(f"✅ Added {new_id} to FAISS!")
        else:
            st.warning("Fill all fields!")

# Main area
query = st.text_input(
    "Enter your query:",
    placeholder="Build secure login system with JWT..."
)

if st.button("🚀 Run Advanced RAG Pipeline"):
    if query:
        # Step 1
        with st.status("📚 Step 1: FAISS Retrieval...") as status:
            docs = search_similar(query, top_k=top_k_retrieve)
            status.update(
                label=f"📚 Retrieved {len(docs)} documents!",
                state="complete"
            )

        # Step 2
        with st.status("🔄 Step 2: Re-ranking...") as status:
            reranked = rerank_documents(query, docs, top_k=top_k_rerank)
            status.update(
                label=f"🔄 Re-ranked! Top doc: {reranked[0]['title']}",
                state="complete"
            )

        # Show reranked docs
        st.markdown("### 📚 Retrieved & Re-ranked Documents:")
        for doc in reranked:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**[{doc['id']}] {doc['title']}**")
            with col2:
                score = doc['rerank_score']
                color = "🟢" if score > 0.5 else "🟡" if score > 0.2 else "🔴"
                st.write(f"{color} {score:.4f}")

        st.divider()

        # Step 3
        with st.status("🤔 Step 3: Generating + Self-reflection...") as status:
            reflection_result = generate_with_reflection(
                query, reranked, max_attempts=max_attempts
            )
            label = "✅ Self-reflection PASSED!" if reflection_result['passed'] \
                else "⚠️ Max attempts reached"
            status.update(label=label, state="complete")

        # Step 4
        with st.status("🎯 Step 4: LLM Critique...") as status:
            critique_result = generate_with_critique(
                query,
                reflection_result['document'],
                reranked,
                max_attempts=max_attempts
            )
            status.update(
                label=f"🎯 Score: {critique_result['score']}/10 — {'✅ APPROVED' if critique_result['approved'] else '❌ REJECTED'}",
                state="complete"
            )

        st.divider()

        # Results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Critique Score", f"{critique_result['score']}/10")
        with col2:
            st.metric("🤔 Self-reflection",
                      "✅ Passed" if reflection_result['passed'] else "⚠️ Failed")
        with col3:
            st.metric("✅ Approved",
                      "YES" if critique_result['approved'] else "NO")
        with col4:
            st.metric("🔄 Attempts", critique_result['attempts'])

        st.markdown("### 📄 Final Document:")
        st.markdown(critique_result['document'])

        st.download_button(
            "💾 Download Document",
            critique_result['document'],
            file_name=f"{query[:20].replace(' ', '_')}.md",
            mime="text/markdown"
        )

        # Critique details
        with st.expander("🎯 Full Critique Report"):
            st.text(critique_result['critique'])

    else:
        st.warning("Enter a query first!")