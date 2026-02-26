"""
FinGuard AI - Observable RAG Customer Support Assistant
Streamlit UI with Real-time Observability Dashboard
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Use new orchestrator architecture
from rag_orchestrator import RAGOrchestrator

# Page config
st.set_page_config(
    page_title="FinGuard AI Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        color: #28a745;
    }
    .warning-metric {
        color: #ffc107;
    }
    .error-metric {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_orchestrator' not in st.session_state:
    with st.spinner("🔧 Initializing FinGuard AI..."):
        try:
            st.session_state.rag_orchestrator = RAGOrchestrator()
            st.session_state.indexed = False
            st.session_state.messages = []
            st.session_state.query_count = 0
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/chatbot.png", width=80)
    st.markdown("# FinGuard AI")
    st.markdown("**Observable RAG Support**")
    st.markdown("---")

    # Index documents
    st.subheader("📚 Knowledge Base")

    if not st.session_state.indexed:
        if st.button("📥 Load Policies", use_container_width=True):
            with st.spinner("Loading and indexing policies..."):
                data_path = os.path.join(os.path.dirname(__file__), "../data/fintech_policies.txt")

                if os.path.exists(data_path):
                    result = st.session_state.rag_orchestrator.index_documents(data_path)

                    if result['success']:
                        st.session_state.indexed = True
                        st.success(f"✅ Indexed {result['chunks_created']} chunks in {result['total_time_ms']:.0f}ms")
                    else:
                        st.error(f"❌ Indexing failed: {result.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ Policy file not found: {data_path}")
    else:
        st.success("✅ Knowledge base loaded")

        # Collection info
        collection_info = st.session_state.rag_orchestrator.get_collection_info()
        st.info(f"📄 **{collection_info['document_count']}** documents indexed")

    st.markdown("---")

    # Session stats
    st.subheader("📊 Session Stats")

    if st.session_state.query_count > 0:
        stats = st.session_state.rag_orchestrator.get_session_stats()

        st.metric("Total Queries", stats['total_queries'])
        st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")
        st.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")
        st.metric("Success Rate", f"{stats['success_rate']:.0f}%")

        if stats['hallucination_rate'] > 0:
            st.metric("Hallucination Rate", f"{stats['hallucination_rate']:.1f}%", delta=None, delta_color="inverse")
    else:
        st.info("No queries yet")

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.rerun()

# Main content
st.markdown('<div class="main-header">🛡️ FinGuard AI Support Assistant</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Powered by Google Gemini with Full Observability
</div>
""", unsafe_allow_html=True)

# Check if indexed
if not st.session_state.indexed:
    st.warning("⚠️ Please load the knowledge base from the sidebar first!")
    st.info("👈 Click **'Load Policies'** in the sidebar to get started")
    st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show metrics for assistant messages
        if message["role"] == "assistant" and "metrics" in message:
            with st.expander("📊 Query Metrics", expanded=False):
                metrics = message["metrics"]

                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("⏱️ Latency", f"{metrics['total_latency_ms']:.0f}ms")
                    st.caption(f"Embed: {metrics['embedding_ms']:.0f}ms")
                    st.caption(f"Search: {metrics['search_ms']:.0f}ms")
                    st.caption(f"LLM: {metrics['llm_ms']:.0f}ms")

                with col2:
                    st.metric("💰 Cost", f"${metrics['total_cost_usd']:.6f}")
                    st.caption(f"Tokens: {metrics['total_tokens']}")
                    st.caption(f"Input: {metrics['input_tokens']}")
                    st.caption(f"Output: {metrics['output_tokens']}")

                with col3:
                    confidence_pct = metrics['confidence'] * 100
                    color = "success-metric" if confidence_pct >= 80 else "warning-metric"
                    st.metric("🎯 Confidence", f"{confidence_pct:.0f}%")
                    st.caption(f"Grounding: {metrics['grounding_score']*100:.0f}%")
                    st.caption(metrics['status'][:20])

                with col4:
                    st.metric("📄 Sources", len(metrics['retrieved_docs']))
                    st.caption(f"Avg Score: {metrics['avg_relevance']:.2f}")
                    st.caption(f"Top Score: {metrics['top_relevance']:.2f}")

                # Show retrieved documents
                st.markdown("**📚 Retrieved Sources:**")
                for i, doc in enumerate(metrics['retrieved_docs'][:3], 1):
                    with st.container():
                        st.markdown(f"**Source {i}** (Relevance: {doc['relevance_score']:.2f})")
                        st.text(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])

# Chat input
if prompt := st.chat_input("Ask me anything about FinGuard policies..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response, metrics = st.session_state.rag_orchestrator.query(prompt)

            st.markdown(response)
            st.session_state.query_count += 1

            # Show inline metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"⏱️ {metrics['total_latency_ms']:.0f}ms")
            with col2:
                st.caption(f"💰 ${metrics['total_cost_usd']:.4f}")
            with col3:
                confidence_pct = metrics['confidence'] * 100
                emoji = "✅" if confidence_pct >= 80 else "⚠️"
                st.caption(f"{emoji} {confidence_pct:.0f}% confidence")

    # Add assistant message with metrics
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🔬 **Observable RAG**")
    st.caption("Every query is tracked and measured")
with col2:
    st.markdown("🤖 **Powered by Gemini**")
    st.caption("Google's latest AI model")
with col3:
    st.markdown("📊 **Real-time Metrics**")
    st.caption("Latency, cost, and quality tracking")

# Example queries (only show if chat is empty)
if len(st.session_state.messages) == 0:
    st.markdown("### 💡 Try asking:")
    example_queries = [
        "Why was my payment declined?",
        "How long do refunds take?",
        "What are forex charges?",
        "Why was I charged twice?",
        "What are the transaction limits?"
    ]

    cols = st.columns(3)
    for i, query in enumerate(example_queries):
        with cols[i % 3]:
            if st.button(query, key=f"example_{i}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
