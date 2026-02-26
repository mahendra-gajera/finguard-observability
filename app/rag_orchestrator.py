"""
RAG Orchestrator - Coordinates AI Service and Observability Service
"""

import os
import sys
import io
from typing import Dict, Tuple

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass  # Ignore if already wrapped or not available

# Add services to path
sys.path.insert(0, os.path.dirname(__file__))

from services.ai_service import AIService
from services.observability_service import ObservabilityService


class RAGOrchestrator:
    """
    RAG Orchestrator - Coordinates AI and Observability Services

    Responsibilities:
    - Coordinate query flow between services
    - Ensure proper trace lifecycle
    - Build context from retrieved documents
    - Return unified response + metrics

    Clean Separation:
    - Delegates AI operations to AIService
    - Delegates monitoring to ObservabilityService
    - No direct AI or observability logic here
    """

    def __init__(self):
        """Initialize orchestrator with both services"""
        self.ai_service = AIService()
        self.obs_service = ObservabilityService()

    def query(self, user_query: str) -> Tuple[str, Dict]:
        """
        Process user query with full observability

        Args:
            user_query: User's question

        Returns:
            tuple: (response_text, complete_metrics)

        Flow:
            1. Start observability trace
            2. Generate embedding (AI)
            3. Record embedding span (Obs)
            4. Search documents (AI)
            5. Record search span (Obs)
            6. Generate response (AI)
            7. Record generation span (Obs)
            8. Analyze quality (Obs)
            9. Complete trace (Obs)
            10. Return response + metrics
        """
        try:
            # Step 1: Start trace
            trace_id = self.obs_service.start_trace(user_query)

            # Step 2: Generate embedding
            query_vector, embed_metrics = self.ai_service.embed_text(user_query)

            if not embed_metrics.get('success'):
                error_msg = f"Embedding failed: {embed_metrics.get('error', 'Unknown error')}"
                return error_msg, {"error": error_msg, "trace_id": trace_id}

            # Step 3: Record embedding span
            self.obs_service.record_span(trace_id, "embedding", embed_metrics)

            # Step 4: Search documents
            retrieved_docs, search_metrics = self.ai_service.search_documents(query_vector)

            if not search_metrics.get('success'):
                error_msg = f"Search failed: {search_metrics.get('error', 'Unknown error')}"
                return error_msg, {"error": error_msg, "trace_id": trace_id}

            # Step 5: Record search span
            self.obs_service.record_span(trace_id, "search", search_metrics)

            # Step 6: Build context from retrieved documents
            context = self._build_context(retrieved_docs)

            # Step 7: Generate response
            response, gen_metrics = self.ai_service.generate_response(user_query, context)

            if not gen_metrics.get('success'):
                error_msg = f"Generation failed: {gen_metrics.get('error', 'Unknown error')}"
                return error_msg, {"error": error_msg, "trace_id": trace_id}

            # Step 8: Record generation span
            self.obs_service.record_span(trace_id, "generation", gen_metrics)

            # Step 9: Analyze quality
            quality_metrics = self.obs_service.analyze_quality(response, retrieved_docs)

            # Step 10: Complete trace
            final_metrics = self.obs_service.complete_trace(trace_id, quality_metrics)

            return response, final_metrics

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return error_msg, {"error": str(e)}

    def index_documents(self, file_path: str) -> Dict:
        """
        Index documents for RAG

        Args:
            file_path: Path to document file

        Returns:
            dict: Indexing results
        """
        return self.ai_service.index_documents(file_path)

    def get_session_stats(self) -> Dict:
        """
        Get session statistics

        Returns:
            dict: Aggregated session metrics
        """
        return self.obs_service.get_session_stats()

    def get_collection_info(self) -> Dict:
        """
        Get vector store collection information

        Returns:
            dict: Collection statistics
        """
        return self.ai_service.vector_store.get_collection_stats()

    def _build_context(self, retrieved_docs: list) -> str:
        """
        Build context string from retrieved documents

        Args:
            retrieved_docs: List of document dicts

        Returns:
            str: Formatted context
        """
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Source {i}]\n{doc['content']}")

        return "\n\n".join(context_parts)


# Test the orchestrator
if __name__ == "__main__":
    print("Testing RAG Orchestrator...")
    print("=" * 70)

    orchestrator = RAGOrchestrator()

    # Test 1: Index documents
    print("\n1. Indexing Documents...")
    data_file = os.path.join(os.path.dirname(__file__), "..", "data", "fintech_policies.txt")

    if os.path.exists(data_file):
        result = orchestrator.index_documents(data_file)
        print(f"   ✅ Indexed {result.get('chunks_created', 0)} chunks")
    else:
        print(f"   ⚠️ Data file not found: {data_file}")

    # Test 2: Query with observability
    print("\n2. Testing Query with Full Observability...")
    test_queries = [
        "Why was my payment declined?",
        "How long do refunds take?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        print("   " + "-" * 60)

        response, metrics = orchestrator.query(query)

        print(f"   Response: {response[:80]}...")
        print(f"   Latency: {metrics.get('total_latency_ms', 0):.0f}ms")
        print(f"   Cost: ${metrics.get('total_cost_usd', 0):.6f}")
        print(f"   Confidence: {metrics.get('confidence', 0)*100:.0f}%")
        print(f"   Hallucination: {metrics.get('hallucination_detected', False)}")

    # Test 3: Session stats
    print("\n3. Session Statistics...")
    stats = orchestrator.get_session_stats()
    print(f"   Total queries: {stats.get('total_queries', 0)}")
    print(f"   Avg latency: {stats.get('avg_latency_ms', 0):.0f}ms")
    print(f"   Total cost: ${stats.get('total_cost_usd', 0):.4f}")
    print(f"   Success rate: {stats.get('success_rate', 0)}%")

    print("\n" + "=" * 70)
    print("✅ RAG Orchestrator test complete!")
