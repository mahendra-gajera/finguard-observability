"""
Unit Tests for Embeddings Module
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestGeminiEmbeddings(unittest.TestCase):
    """Test Gemini embeddings functionality"""

    @patch('embeddings.genai')
    def test_embed_query_success(self, mock_genai):
        """Test successful query embedding"""
        # Mock the API response
        mock_response = {
            'embedding': [0.1] * 768  # 768-dim vector
        }
        mock_genai.embed_content.return_value = mock_response

        # Import after mocking
        from embeddings import GeminiEmbeddings

        embeddings = GeminiEmbeddings()
        vector, metrics = embeddings.embed_query("test query")

        # Verify embedding
        self.assertEqual(len(vector), 768)
        self.assertIsInstance(vector, list)

        # Verify metrics
        self.assertIn('embedding_time_ms', metrics)
        self.assertIn('model', metrics)
        self.assertTrue(metrics['success'])

    @patch('embeddings.genai')
    def test_embed_documents_batch(self, mock_genai):
        """Test batch document embedding"""
        # Mock the API response
        mock_response = {
            'embedding': [0.1] * 768
        }
        mock_genai.embed_content.return_value = mock_response

        from embeddings import GeminiEmbeddings

        embeddings = GeminiEmbeddings()
        texts = ["doc1", "doc2", "doc3"]
        vectors, metrics = embeddings.embed_documents(texts)

        # Should have 3 embeddings
        self.assertEqual(len(vectors), 3)
        self.assertEqual(len(vectors[0]), 768)

        # Verify metrics
        self.assertEqual(metrics['batch_size'], 3)
        self.assertIn('embedding_time_ms', metrics)

    @patch('embeddings.genai')
    def test_embed_query_error_handling(self, mock_genai):
        """Test error handling in embedding"""
        # Mock API error
        mock_genai.embed_content.side_effect = Exception("API Error")

        from embeddings import GeminiEmbeddings

        embeddings = GeminiEmbeddings()

        with self.assertRaises(Exception):
            embeddings.embed_query("test query")

    def test_embedding_dimension(self):
        """Test embedding dimension is correct"""
        from embeddings import GeminiEmbeddings

        embeddings = GeminiEmbeddings()
        self.assertEqual(embeddings.dimension, 768)


if __name__ == '__main__':
    unittest.main()
