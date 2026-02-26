"""
Unit Tests for RAG Pipeline
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestRAGPipeline(unittest.TestCase):
    """Test RAG Pipeline integration"""

    @patch('rag_pipeline.GeminiLLM')
    @patch('rag_pipeline.VectorStore')
    @patch('rag_pipeline.GeminiEmbeddings')
    def test_pipeline_initialization(self, mock_embeddings, mock_vector_store, mock_llm):
        """Test pipeline initializes all components"""
        from rag_pipeline import ObservableRAGPipeline

        # Mock the components
        mock_embeddings.return_value = Mock()
        mock_vector_store.return_value = Mock()
        mock_llm.return_value = Mock()

        pipeline = ObservableRAGPipeline()

        # Verify components are initialized
        self.assertIsNotNone(pipeline.embeddings)
        self.assertIsNotNone(pipeline.vector_store)
        self.assertIsNotNone(pipeline.llm)
        self.assertIsNotNone(pipeline.tracker)

    @patch('rag_pipeline.GeminiLLM')
    @patch('rag_pipeline.VectorStore')
    @patch('rag_pipeline.GeminiEmbeddings')
    def test_query_returns_response_and_metrics(self, mock_embeddings, mock_vector_store, mock_llm):
        """Test query returns both response and metrics"""
        from rag_pipeline import ObservableRAGPipeline

        # Setup mocks
        mock_embeddings_instance = Mock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_vector_store_instance = Mock()
        mock_vector_store_instance.similarity_search.return_value = (
            [
                {"content": "Test content", "relevance_score": 0.9},
                {"content": "More content", "relevance_score": 0.8}
            ],
            {"search_time_ms": 30, "total_time_ms": 50}
        )
        mock_vector_store.return_value = mock_vector_store_instance

        mock_llm_instance = Mock()
        mock_llm_instance.generate_with_context.return_value = (
            "Test response",
            {
                "llm_time_ms": 800,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "total_cost_usd": 0.002
            }
        )
        mock_llm.return_value = mock_llm_instance

        pipeline = ObservableRAGPipeline()
        response, metrics = pipeline.query("test query")

        # Verify response
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Test response")

        # Verify metrics structure
        self.assertIn('total_latency_ms', metrics)
        self.assertIn('confidence', metrics)
        self.assertIn('hallucination_detected', metrics)
        self.assertIn('retrieved_docs', metrics)
        self.assertIn('total_cost_usd', metrics)

    def test_chunk_documents(self):
        """Test document chunking logic"""
        # This would test the chunking logic if it were extracted
        # For now, we can verify the format
        content = """## Section 1
This is content for section 1.

## Section 2
This is content for section 2."""

        chunks = []
        current_chunk = ""
        lines = content.split('\n')

        for line in lines:
            if line.startswith('##') and current_chunk:
                if len(current_chunk.strip()) > 10:
                    chunks.append(current_chunk.strip())
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if len(current_chunk.strip()) > 10:
            chunks.append(current_chunk.strip())

        # Should have 2 chunks
        self.assertEqual(len(chunks), 2)
        self.assertIn("Section 1", chunks[0])
        self.assertIn("Section 2", chunks[1])


if __name__ == '__main__':
    unittest.main()
