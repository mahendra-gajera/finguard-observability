"""
Observability Service - Pure Monitoring and Tracking
Handles traces, quality checks, and metrics without AI operations
"""

import time
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from observability import detect_hallucination, calculate_relevance_score


class ObservabilityService:
    """
    Observability Service - Provides monitoring and tracking

    Responsibilities:
    - Create and manage traces
    - Track operation spans
    - Analyze response quality
    - Detect hallucinations
    - Calculate confidence scores
    - Aggregate session metrics

    Does NOT handle:
    - AI/ML operations
    - API calls to AI providers
    - Vector search
    - Embedding generation
    """

    def __init__(self):
        """Initialize observability tracking"""
        self.traces = []
        self.session_start = time.time()
        self.cost_per_input_token = float(os.getenv("GEMINI_INPUT_COST", "0.00001")) / 1000
        self.cost_per_output_token = float(os.getenv("GEMINI_OUTPUT_COST", "0.00003")) / 1000

    def start_trace(self, query: str) -> str:
        """
        Start a new trace for a query

        Args:
            query: User's query text

        Returns:
            str: Unique trace ID
        """
        trace_id = f"trace_{len(self.traces) + 1}_{int(time.time())}"

        trace = {
            "trace_id": trace_id,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time(),
            "spans": {},
            "metrics": {},
            "status": "in_progress"
        }

        self.traces.append(trace)
        return trace_id

    def record_span(self, trace_id: str, operation: str, metrics: Dict):
        """
        Record an operation span in the trace

        Args:
            trace_id: Trace identifier
            operation: Operation name (e.g., "embedding", "search", "generation")
            metrics: Metrics from the operation
        """
        trace = self._get_trace(trace_id)
        if not trace:
            return

        trace["spans"][operation] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_quality(self, response: str, retrieved_docs: List[Dict]) -> Dict:
        """
        Analyze response quality

        Args:
            response: Generated response text
            retrieved_docs: Documents used for generation

        Returns:
            dict: Quality metrics
                - confidence: 0-1 score
                - grounding_score: 0-1 score
                - hallucination_detected: boolean
                - status: descriptive status
                - retrieved_docs_count: number of sources
                - avg_relevance: average relevance of docs
        """
        # Hallucination detection
        hallucination_result = detect_hallucination(response, retrieved_docs)

        # Relevance scoring
        relevance_result = calculate_relevance_score("", retrieved_docs)

        return {
            "confidence": hallucination_result['confidence'],
            "grounding_score": hallucination_result['grounding_score'],
            "hallucination_detected": hallucination_result['hallucination_detected'],
            "status": hallucination_result['status'],
            "retrieved_docs_count": len(retrieved_docs),
            "avg_relevance": relevance_result.get('avg_relevance', 0),
            "retrieved_docs": retrieved_docs
        }

    def complete_trace(self, trace_id: str, quality_metrics: Dict = None) -> Dict:
        """
        Complete a trace and return final metrics

        Args:
            trace_id: Trace identifier
            quality_metrics: Optional quality metrics to include

        Returns:
            dict: Complete trace metrics
        """
        trace = self._get_trace(trace_id)
        if not trace:
            return {}

        total_time = (time.time() - trace["start_time"]) * 1000

        # Extract metrics from spans
        embedding_time = 0
        search_time = 0
        llm_time = 0
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        top_relevance = 0
        results_count = 0

        # Get embedding metrics
        if "embedding" in trace["spans"]:
            embedding_time = trace["spans"]["embedding"]["metrics"].get("duration_ms", 0)

        # Get search metrics
        if "search" in trace["spans"]:
            search_metrics = trace["spans"]["search"]["metrics"]
            search_time = search_metrics.get("duration_ms", 0)
            top_relevance = search_metrics.get("top_relevance", 0)
            results_count = search_metrics.get("results_count", 0)

        # Get generation metrics
        if "generation" in trace["spans"]:
            gen_metrics = trace["spans"]["generation"]["metrics"]
            llm_time = gen_metrics.get("duration_ms", 0)
            input_tokens = gen_metrics.get("input_tokens", 0)
            output_tokens = gen_metrics.get("output_tokens", 0)
            total_tokens = gen_metrics.get("total_tokens", 0)

        # Calculate cost
        input_cost = input_tokens * self.cost_per_input_token
        output_cost = output_tokens * self.cost_per_output_token
        total_cost = input_cost + output_cost

        # Build complete metrics
        complete_metrics = {
            "trace_id": trace_id,
            "query": trace["query"],
            "total_latency_ms": round(total_time, 2),
            "embedding_ms": round(embedding_time, 2),
            "search_ms": round(search_time, 2),
            "llm_ms": round(llm_time, 2),
            "other_ms": round(total_time - embedding_time - search_time - llm_time, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "top_relevance": round(top_relevance, 3),
            "results_count": results_count
        }

        # Add quality metrics if provided
        if quality_metrics:
            complete_metrics.update(quality_metrics)

        # Update trace
        trace["metrics"] = complete_metrics
        trace["status"] = "completed"
        trace["end_time"] = time.time()

        return complete_metrics

    def get_session_stats(self) -> Dict:
        """
        Get aggregated session statistics

        Returns:
            dict: Session-level metrics
                - total_queries: Count
                - avg_latency_ms: Average
                - p95_latency_ms: 95th percentile
                - total_cost_usd: Total cost
                - avg_cost_per_query: Average cost
                - hallucination_rate: Percentage
                - success_rate: Percentage
                - session_duration_sec: Duration
        """
        if not self.traces:
            return {
                "total_queries": 0,
                "avg_latency_ms": 0,
                "total_cost_usd": 0,
                "success_rate": 0
            }

        completed_traces = [t for t in self.traces if t["status"] == "completed"]

        if not completed_traces:
            return {
                "total_queries": len(self.traces),
                "avg_latency_ms": 0,
                "total_cost_usd": 0,
                "success_rate": 0
            }

        latencies = [t["metrics"]["total_latency_ms"] for t in completed_traces]
        costs = [t["metrics"].get("total_cost_usd", 0) for t in completed_traces]
        hallucinations = sum(1 for t in completed_traces if t["metrics"].get("hallucination_detected", False))

        return {
            "total_queries": len(completed_traces),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) > 1 else latencies[0],
            "min_latency_ms": round(min(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "total_cost_usd": round(sum(costs), 4),
            "avg_cost_per_query": round(sum(costs) / len(costs), 6),
            "hallucination_rate": round((hallucinations / len(completed_traces)) * 100, 2),
            "success_rate": round((len(completed_traces) / len(self.traces)) * 100, 2),
            "session_duration_sec": round(time.time() - self.session_start, 1)
        }

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """
        Get a specific trace by ID

        Args:
            trace_id: Trace identifier

        Returns:
            dict: Trace object or None
        """
        return self._get_trace(trace_id)

    def _get_trace(self, trace_id: str) -> Optional[Dict]:
        """Internal method to find trace by ID"""
        for trace in self.traces:
            if trace["trace_id"] == trace_id:
                return trace
        return None

    def export_metrics(self, format: str = "json") -> Dict:
        """
        Export all metrics for external monitoring systems

        Args:
            format: Export format (currently only "json")

        Returns:
            dict: Exportable metrics
        """
        return {
            "session_stats": self.get_session_stats(),
            "traces": [
                {
                    "trace_id": t["trace_id"],
                    "query": t["query"],
                    "status": t["status"],
                    "metrics": t.get("metrics", {})
                }
                for t in self.traces
            ],
            "export_timestamp": datetime.now().isoformat()
        }


# Test the service
if __name__ == "__main__":
    print("Testing Observability Service...")

    obs_service = ObservabilityService()

    # Test 1: Start trace
    trace_id = obs_service.start_trace("Test query")
    print(f"\n1. Trace Started: {trace_id}")

    # Test 2: Record spans
    obs_service.record_span(trace_id, "embedding", {
        "duration_ms": 45,
        "success": True
    })
    obs_service.record_span(trace_id, "search", {
        "duration_ms": 30,
        "results_count": 3,
        "success": True
    })
    obs_service.record_span(trace_id, "generation", {
        "duration_ms": 850,
        "input_tokens": 120,
        "output_tokens": 85,
        "total_tokens": 205,
        "success": True
    })
    print("   Spans recorded: embedding, search, generation")

    # Test 3: Analyze quality
    test_response = "Payment was declined due to insufficient funds."
    test_docs = [
        {"content": "Payments decline for insufficient funds or expired cards.", "relevance_score": 0.85}
    ]
    quality = obs_service.analyze_quality(test_response, test_docs)
    print(f"\n2. Quality Analysis:")
    print(f"   Confidence: {quality['confidence']*100:.0f}%")
    print(f"   Hallucination: {quality['hallucination_detected']}")

    # Test 4: Complete trace
    final_metrics = obs_service.complete_trace(trace_id, quality)
    print(f"\n3. Trace Completed:")
    print(f"   Total latency: {final_metrics['total_latency_ms']:.0f}ms")
    print(f"   Total cost: ${final_metrics['total_cost_usd']:.6f}")

    # Test 5: Session stats
    stats = obs_service.get_session_stats()
    print(f"\n4. Session Stats:")
    print(f"   Total queries: {stats['total_queries']}")
    print(f"   Success rate: {stats['success_rate']}%")

    print("\n✅ Observability Service test complete!")
