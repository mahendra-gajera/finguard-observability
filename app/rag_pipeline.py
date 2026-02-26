"""
RAG Pipeline - Complete Observable RAG Implementation
"""

import os
import sys
from typing import Dict, Tuple
from dotenv import load_dotenv

from embeddings import GeminiEmbeddings
from vector_store import VectorStore
from llm import GeminiLLM
from observability import (
    ObservabilityTracker,
    detect_hallucination,
    calculate_relevance_score
)

load_dotenv()

# Safe print function for Windows emoji issues
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        # Fallback: remove emojis
        print(text.encode('ascii', 'ignore').decode('ascii'))


class ObservableRAGPipeline:
    """Complete RAG pipeline with full observability"""

    def __init__(self):
        """Initialize RAG components"""
        safe_print("🔧 Initializing RAG Pipeline...")

        # Initialize components
        self.embeddings = GeminiEmbeddings()
        self.vector_store = VectorStore(self.embeddings)
        self.llm = GeminiLLM()

        # Observability
        self.tracker = ObservabilityTracker()

        # Settings
        self.top_k = int(os.getenv("TOP_K_RESULTS", "3"))

        safe_print("✅ RAG Pipeline ready!")

    def index_documents(self, file_path: str) -> Dict:
        """Load and index documents from file"""
        safe_print(f"\n📚 Loading documents from {file_path}...")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple chunking by sections (split by ## headers)
            chunks = []
            current_chunk = ""
            lines = content.split('\n')

            for line in lines:
                if line.startswith('##') and current_chunk:
                    # Save previous chunk
                    if len(current_chunk.strip()) > 50:
                        chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'

            # Add last chunk
            if len(current_chunk.strip()) > 50:
                chunks.append(current_chunk.strip())

            safe_print(f"📄 Created {len(chunks)} chunks from document")

            # Create metadata
            metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]

            # Index documents
            metrics = self.vector_store.add_documents(chunks, metadatas)

            safe_print(f"✅ Indexed {len(chunks)} chunks successfully!")
            safe_print(f"⏱️ Indexing took {metrics['total_time_ms']:.0f}ms")

            return {
                "success": True,
                "chunks_created": len(chunks),
                **metrics
            }

        except Exception as e:
            safe_print(f"❌ Error indexing documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def query(self, user_query: str) -> Tuple[str, Dict]:
        """
        Process query with full observability

        Returns:
            Tuple[response_text, complete_metrics_dict]
        """
        # Create trace
        trace = self.tracker.create_trace(user_query)

        try:
            # Step 1: Search vector store
            retrieved_docs, search_metrics = self.vector_store.similarity_search(
                user_query,
                k=self.top_k
            )
            self.tracker.add_span(trace, "search", search_metrics)

            if not retrieved_docs:
                return "I don't have enough information to answer that question. Please contact customer support.", {
                    "error": "No relevant documents found",
                    "confidence": 0
                }

            # Step 2: Build context
            context = "\n\n".join([
                f"[Source {i+1}]\n{doc['content']}"
                for i, doc in enumerate(retrieved_docs)
            ])

            # Step 3: Generate response with LLM
            response, llm_metrics = self.llm.generate_with_context(user_query, context)
            self.tracker.add_span(trace, "llm", llm_metrics)

            # Step 4: Quality checks
            hallucination_check = detect_hallucination(response, retrieved_docs)
            relevance_check = calculate_relevance_score(user_query, retrieved_docs)

            quality_metrics = {
                **hallucination_check,
                **relevance_check
            }

            # Complete trace
            self.tracker.complete_trace(trace, response, quality_metrics)

            # Prepare complete metrics
            complete_metrics = {
                **trace["metrics"],
                "retrieved_docs": retrieved_docs,
                "trace_id": trace["trace_id"]
            }

            return response, complete_metrics

        except Exception as e:
            trace["status"] = "failed"
            trace["error"] = str(e)

            return f"Sorry, I encountered an error: {str(e)}", {
                "error": str(e),
                "trace_id": trace["trace_id"],
                "success": False
            }

    def get_session_stats(self) -> Dict:
        """Get current session statistics"""
        return self.tracker.get_session_metrics()

    def get_collection_info(self) -> Dict:
        """Get vector store statistics"""
        return self.vector_store.get_collection_stats()


if __name__ == "__main__":
    # Test RAG pipeline
    print("=" * 60)
    print("Testing Observable RAG Pipeline")
    print("=" * 60)

    # Initialize pipeline
    rag = ObservableRAGPipeline()

    # Index documents
    data_file = "../data/fintech_policies.txt"
    if os.path.exists(data_file):
        index_result = rag.index_documents(data_file)
        safe_print(f"\n📊 Indexing Result: {index_result['success']}")
    else:
        safe_print(f"⚠️ Data file not found: {data_file}")

    # Test queries
    test_queries = [
        "Why was my payment declined?",
        "How long do refunds take?",
        "What are forex charges?"
    ]

    for i, query in enumerate(test_queries, 1):
        safe_print(f"\n{'=' * 60}")
        safe_print(f"Query {i}: {query}")
        safe_print('=' * 60)

        response, metrics = rag.query(query)

        safe_print(f"\n💬 Response:\n{response[:200]}...")

        safe_print(f"\n📊 Metrics:")
        safe_print(f"   ⏱️ Total Latency: {metrics['total_latency_ms']:.0f}ms")
        safe_print(f"   💰 Cost: ${metrics['total_cost_usd']:.4f}")
        safe_print(f"   🔍 Confidence: {metrics['confidence']*100:.0f}%")
        safe_print(f"   ✓ Hallucination: {metrics['status']}")
        safe_print(f"   📄 Sources: {len(metrics['retrieved_docs'])} documents")

    # Session stats
    safe_print(f"\n{'=' * 60}")
    safe_print("Session Statistics")
    safe_print('=' * 60)
    stats = rag.get_session_stats()
    for key, value in stats.items():
        safe_print(f"   {key}: {value}")
