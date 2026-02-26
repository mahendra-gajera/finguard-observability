"""
Unit Tests for LLM Module
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))


class TestGeminiLLM(unittest.TestCase):
    """Test Gemini LLM functionality"""

    @patch('llm.genai')
    def test_generate_success(self, mock_genai):
        """Test successful text generation"""
        # Mock response
        mock_response = Mock()
        mock_response.text = "This is a test response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 50
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 70

        # Mock model
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        from llm import GeminiLLM

        llm = GeminiLLM()
        response, metrics = llm.generate("test prompt")

        # Verify response
        self.assertEqual(response, "This is a test response")

        # Verify metrics
        self.assertEqual(metrics['input_tokens'], 50)
        self.assertEqual(metrics['output_tokens'], 20)
        self.assertEqual(metrics['total_tokens'], 70)
        self.assertIn('llm_time_ms', metrics)
        self.assertIn('total_cost_usd', metrics)

    @patch('llm.genai')
    def test_generate_with_context(self, mock_genai):
        """Test generation with RAG context"""
        # Mock response
        mock_response = Mock()
        mock_response.text = "Answer based on context"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        from llm import GeminiLLM

        llm = GeminiLLM()
        context = "This is the context from retrieved documents"
        query = "What is the answer?"

        response, metrics = llm.generate_with_context(query, context)

        # Verify response
        self.assertIsInstance(response, str)
        self.assertEqual(response, "Answer based on context")

        # Verify metrics
        self.assertGreater(metrics['input_tokens'], 0)
        self.assertIn('total_cost_usd', metrics)

    @patch('llm.genai')
    def test_cost_calculation(self, mock_genai):
        """Test token cost calculation"""
        # Mock response with known token counts
        mock_response = Mock()
        mock_response.text = "Response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 1000  # 1K tokens
        mock_response.usage_metadata.candidates_token_count = 1000  # 1K tokens
        mock_response.usage_metadata.total_token_count = 2000

        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        from llm import GeminiLLM

        llm = GeminiLLM()
        response, metrics = llm.generate("test")

        # Expected cost: (1000/1000 * 0.00001) + (1000/1000 * 0.00003) = 0.00004
        expected_cost = 0.00004
        self.assertAlmostEqual(metrics['total_cost_usd'], expected_cost, places=6)

    @patch('llm.genai')
    def test_error_handling(self, mock_genai):
        """Test LLM error handling"""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model

        from llm import GeminiLLM

        llm = GeminiLLM()

        with self.assertRaises(Exception):
            llm.generate("test prompt")


if __name__ == '__main__':
    unittest.main()
