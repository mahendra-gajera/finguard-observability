"""
Observability Module - Metrics, Tracing, and Hallucination Detection
"""

import time
from typing import Dict, List
from datetime import datetime
import json


class ObservabilityTracker:
    """Track and aggregate observability metrics"""

    def __init__(self):
        """Initialize tracker"""
        self.traces = []
        self.session_start = time.time()

    def create_trace(self, query: str) -> Dict:
        """Create a new trace for a query"""
        trace = {
            "trace_id": f"trace_{len(self.traces) + 1}_{int(time.time())}",
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "start_time": time.time(),
            "spans": {},
            "metrics": {},
            "status": "in_progress"
        }
        self.traces.append(trace)
        return trace

    def add_span(self, trace: Dict, span_name: str, metrics: Dict):
        """Add a span (step) to the trace"""
        trace["spans"][span_name] = {
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }

    def complete_trace(self, trace: Dict, response: str, quality_check: Dict):
        """Complete the trace with final metrics"""
        total_time = (time.time() - trace["start_time"]) * 1000

        # Aggregate metrics from all spans
        embedding_time = trace["spans"].get("embedding", {}).get("metrics", {}).get("embedding_time_ms", 0)
        search_time = trace["spans"].get("search", {}).get("metrics", {}).get("total_time_ms", 0)
        llm_time = trace["spans"].get("llm", {}).get("metrics", {}).get("llm_time_ms", 0)

        trace["metrics"] = {
            "total_latency_ms": round(total_time, 2),
            "embedding_ms": embedding_time,
            "search_ms": search_time,
            "llm_ms": llm_time,
            "other_ms": round(total_time - embedding_time - search_time - llm_time, 2),
            **trace["spans"].get("llm", {}).get("metrics", {}),
            **quality_check
        }

        trace["response"] = response
        trace["status"] = "completed"
        trace["end_time"] = time.time()

    def get_session_metrics(self) -> Dict:
        """Get aggregated session metrics"""
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


def detect_hallucination(response: str, retrieved_docs: List[Dict]) -> Dict:
    """
    Simple hallucination detection - checks if key facts in response are grounded in sources

    Returns:
        Dict with hallucination check results
    """
    # Combine all retrieved content
    source_content = " ".join([doc['content'].lower() for doc in retrieved_docs])
    response_lower = response.lower()

    # Extract key phrases from response (simple word extraction)
    response_words = set(response_lower.split())

    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
    key_words = response_words - stop_words

    # Check overlap with source content
    grounded_words = sum(1 for word in key_words if word in source_content)
    total_key_words = len(key_words)

    if total_key_words == 0:
        grounding_score = 1.0
    else:
        grounding_score = grounded_words / total_key_words

    # Determine if hallucinated
    # If less than 60% of key words are found in source, flag as potential hallucination
    hallucination_threshold = 0.6
    is_hallucinated = grounding_score < hallucination_threshold

    # Check for "don't have information" type responses
    uncertainty_phrases = [
        "don't have that information",
        "not available in our policies",
        "please contact customer support",
        "i don't have",
        "cannot find",
        "not mentioned"
    ]

    is_uncertain = any(phrase in response_lower for phrase in uncertainty_phrases)

    # If model is uncertain, that's good (not hallucinating) but confidence should be LOW
    if is_uncertain:
        confidence = 0.15  # Low confidence for out-of-scope questions
        hallucination_detected = False
        status = "✅ Properly uncertain - out of scope"
    elif is_hallucinated:
        confidence = grounding_score
        hallucination_detected = True
        status = "⚠️ Potential hallucination detected"
    else:
        confidence = grounding_score
        hallucination_detected = False
        status = "✅ Well-grounded response"

    return {
        "hallucination_detected": hallucination_detected,
        "confidence": round(confidence, 3),
        "grounding_score": round(grounding_score, 3),
        "status": status,
        "key_words_checked": total_key_words,
        "grounded_words": grounded_words
    }


def calculate_relevance_score(query: str, retrieved_docs: List[Dict]) -> Dict:
    """Calculate how relevant the retrieved documents are to the query"""
    if not retrieved_docs:
        return {
            "avg_relevance": 0,
            "top_relevance": 0,
            "relevance_distribution": []
        }

    relevance_scores = [doc['relevance_score'] for doc in retrieved_docs]

    return {
        "avg_relevance": round(sum(relevance_scores) / len(relevance_scores), 3),
        "top_relevance": round(max(relevance_scores), 3),
        "relevance_distribution": [round(s, 3) for s in relevance_scores]
    }


def export_traces_to_json(traces: List[Dict], filename: str = "traces.json"):
    """Export traces for external analysis"""
    with open(filename, 'w') as f:
        json.dump(traces, f, indent=2)
    print(f"✅ Exported {len(traces)} traces to {filename}")


if __name__ == "__main__":
    # Test observability tracker
    tracker = ObservabilityTracker()

    # Simulate a query
    trace = tracker.create_trace("Why was my payment declined?")

    # Simulate adding spans
    tracker.add_span(trace, "embedding", {"embedding_time_ms": 45})
    tracker.add_span(trace, "search", {"total_time_ms": 30, "results_count": 3})
    tracker.add_span(trace, "llm", {"llm_time_ms": 850, "total_tokens": 150, "total_cost_usd": 0.002})

    # Simulate quality check
    test_docs = [
        {"content": "Payments can be declined due to insufficient funds or expired cards."}
    ]
    test_response = "Your payment was declined because of insufficient funds or an expired card."

    quality_check = detect_hallucination(test_response, test_docs)

    # Complete trace
    tracker.complete_trace(trace, test_response, quality_check)

    # Print trace
    print("📊 Trace Metrics:")
    print(json.dumps(trace["metrics"], indent=2))

    # Print session metrics
    print("\n📈 Session Metrics:")
    print(json.dumps(tracker.get_session_metrics(), indent=2))
