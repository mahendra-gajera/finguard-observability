"""
Unit Tests for Observability Module
"""

import unittest
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from observability import (
    detect_hallucination,
    calculate_relevance_score,
    ObservabilityTracker
)


class TestHallucinationDetection(unittest.TestCase):
    """Test hallucination detection logic"""

    def test_well_grounded_response(self):
        """Test response that is well-grounded in sources"""
        response = "Your payment was declined due to insufficient funds or expired card."
        retrieved_docs = [
            {
                "content": "Payment declines happen due to insufficient funds, expired cards, or incorrect details.",
                "relevance_score": 0.85
            }
        ]

        result = detect_hallucination(response, retrieved_docs)

        self.assertFalse(result['hallucination_detected'])
        self.assertGreater(result['confidence'], 0.5)
        self.assertGreater(result['grounding_score'], 0.5)

    def test_out_of_scope_response(self):
        """Test response for out-of-scope question"""
        response = "I don't have that information in our policies."
        retrieved_docs = [
            {
                "content": "Payment policies and refund information.",
                "relevance_score": 0.2
            }
        ]

        result = detect_hallucination(response, retrieved_docs)

        # Should not be flagged as hallucination (correctly uncertain)
        self.assertFalse(result['hallucination_detected'])
        # Should have low confidence (out of scope)
        self.assertLess(result['confidence'], 0.3)
        self.assertIn("out of scope", result['status'].lower())

    def test_potential_hallucination(self):
        """Test response with content not in sources"""
        response = "Our system uses blockchain technology for all transactions."
        retrieved_docs = [
            {
                "content": "Payment processing uses standard banking protocols.",
                "relevance_score": 0.3
            }
        ]

        result = detect_hallucination(response, retrieved_docs)

        # Should flag as potential hallucination
        self.assertTrue(result['hallucination_detected'])
        self.assertLess(result['grounding_score'], 0.6)

    def test_empty_response(self):
        """Test handling of empty response"""
        response = ""
        retrieved_docs = [
            {"content": "Some content", "relevance_score": 0.5}
        ]

        result = detect_hallucination(response, retrieved_docs)

        # Should not crash
        self.assertIsNotNone(result)
        self.assertIn('hallucination_detected', result)

    def test_empty_sources(self):
        """Test handling of no retrieved documents"""
        response = "Some response text"
        retrieved_docs = []

        result = detect_hallucination(response, retrieved_docs)

        # Should flag as potential hallucination (no sources)
        self.assertTrue(result['hallucination_detected'])


class TestRelevanceScoring(unittest.TestCase):
    """Test relevance score calculation"""

    def test_calculate_relevance_with_docs(self):
        """Test relevance calculation with multiple docs"""
        retrieved_docs = [
            {"relevance_score": 0.9},
            {"relevance_score": 0.7},
            {"relevance_score": 0.5}
        ]

        result = calculate_relevance_score("test query", retrieved_docs)

        self.assertAlmostEqual(result['avg_relevance'], 0.7, places=1)
        self.assertEqual(result['top_relevance'], 0.9)

    def test_calculate_relevance_empty(self):
        """Test relevance calculation with no docs"""
        result = calculate_relevance_score("test query", [])

        self.assertEqual(result['avg_relevance'], 0)
        self.assertEqual(result['top_relevance'], 0)


class TestObservabilityTracker(unittest.TestCase):
    """Test observability tracker"""

    def setUp(self):
        """Set up tracker for each test"""
        self.tracker = ObservabilityTracker()

    def test_create_trace(self):
        """Test trace creation"""
        trace = self.tracker.create_trace("test query")

        self.assertIsNotNone(trace)
        self.assertEqual(trace['query'], "test query")
        self.assertEqual(trace['status'], "in_progress")
        self.assertIn('trace_id', trace)
        self.assertIn('start_time', trace)

    def test_add_span(self):
        """Test adding span to trace"""
        trace = self.tracker.create_trace("test query")

        metrics = {
            "latency_ms": 100,
            "tokens": 50
        }

        self.tracker.add_span(trace, "llm", metrics)

        self.assertIn("llm", trace['spans'])
        self.assertEqual(trace['spans']['llm']['metrics']['latency_ms'], 100)

    def test_complete_trace(self):
        """Test completing a trace"""
        trace = self.tracker.create_trace("test query")

        quality_check = {
            "confidence": 0.8,
            "hallucination_detected": False
        }

        self.tracker.complete_trace(trace, "test response", quality_check)

        self.assertEqual(trace['status'], "completed")
        self.assertEqual(trace['response'], "test response")
        self.assertIn('metrics', trace)
        self.assertIn('total_latency_ms', trace['metrics'])

    def test_session_metrics_empty(self):
        """Test session metrics with no traces"""
        metrics = self.tracker.get_session_metrics()

        self.assertEqual(metrics['total_queries'], 0)
        self.assertEqual(metrics['avg_latency_ms'], 0)

    def test_session_metrics_with_traces(self):
        """Test session metrics with completed traces"""
        # Create and complete a trace
        trace = self.tracker.create_trace("test query")

        # Add spans with proper metrics
        self.tracker.add_span(trace, "search", {"total_time_ms": 50, "embedding_time_ms": 20})
        self.tracker.add_span(trace, "llm", {
            "llm_time_ms": 100,
            "input_tokens": 50,
            "output_tokens": 30,
            "total_cost_usd": 0.001
        })

        quality_check = {
            "confidence": 0.8,
            "hallucination_detected": False,
            "grounding_score": 0.75,
            "status": "Well-grounded"
        }

        self.tracker.complete_trace(trace, "response", quality_check)

        metrics = self.tracker.get_session_metrics()

        self.assertEqual(metrics['total_queries'], 1)
        self.assertGreaterEqual(metrics['avg_latency_ms'], 0)  # Can be 0 if test runs too fast
        self.assertEqual(metrics['success_rate'], 100.0)


if __name__ == '__main__':
    unittest.main()
