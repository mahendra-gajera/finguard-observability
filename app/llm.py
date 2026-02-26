"""
LLM Module - Google Gemini with Token Tracking and Cost Calculation
"""

import time
import os
from typing import Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiLLM:
    """Google Gemini LLM with observability"""

    def __init__(self):
        """Initialize Gemini LLM"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        self.model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "500"))

        # Cost per 1K tokens (USD)
        self.input_cost_per_1k = float(os.getenv("GEMINI_INPUT_COST", "0.00001"))
        self.output_cost_per_1k = float(os.getenv("GEMINI_OUTPUT_COST", "0.00003"))

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
            }
        )

    def generate(self, prompt: str) -> tuple:
        """
        Generate response with full observability

        Returns:
            tuple: (response_text, metrics_dict)
        """
        start_time = time.time()

        try:
            # Generate response
            response = self.model.generate_content(prompt)

            duration_ms = (time.time() - start_time) * 1000

            # Extract token usage (if available)
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total_tokens = response.usage_metadata.total_token_count
            else:
                # Estimate tokens if not available (rough estimate: 1 token ≈ 4 characters)
                input_tokens = len(prompt) // 4
                output_tokens = len(response.text) // 4
                total_tokens = input_tokens + output_tokens

            # Calculate cost
            input_cost = (input_tokens / 1000) * self.input_cost_per_1k
            output_cost = (output_tokens / 1000) * self.output_cost_per_1k
            total_cost = input_cost + output_cost

            metrics = {
                "llm_time_ms": round(duration_ms, 2),
                "model": self.model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost_usd": round(input_cost, 6),
                "output_cost_usd": round(output_cost, 6),
                "total_cost_usd": round(total_cost, 6),
                "temperature": self.temperature,
                "success": True
            }

            return response.text, metrics

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "llm_time_ms": round(duration_ms, 2),
                "model": self.model_name,
                "error": str(e),
                "success": False
            }

            raise Exception(f"LLM generation failed: {str(e)}")

    def generate_with_context(self, query: str, context: str) -> tuple:
        """
        Generate response with RAG context

        Returns:
            tuple: (response_text, metrics_dict)
        """
        prompt = f"""You are a helpful FinGuard customer support assistant. Answer the customer's question based ONLY on the provided context.

Context from FinGuard policies:
{context}

Customer Question: {query}

Instructions:
1. Answer based strictly on the provided context
2. Be concise and helpful
3. If the context doesn't contain the answer, say "I don't have that information in our policies. Please contact customer support."
4. Include relevant policy references when applicable
5. Be polite and professional

Answer:"""

        return self.generate(prompt)


def calculate_session_cost(metrics_list: list) -> Dict:
    """Calculate cumulative cost for a session"""
    total_cost = sum(m.get('total_cost_usd', 0) for m in metrics_list)
    total_tokens = sum(m.get('total_tokens', 0) for m in metrics_list)
    total_queries = len(metrics_list)

    return {
        "total_queries": total_queries,
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_query": round(total_cost / total_queries, 6) if total_queries > 0 else 0,
        "avg_tokens_per_query": round(total_tokens / total_queries, 1) if total_queries > 0 else 0
    }


if __name__ == "__main__":
    # Test LLM
    llm = GeminiLLM()

    test_context = """
    Payment declines can occur due to:
    1. Insufficient funds
    2. Expired card
    3. Security restrictions
    4. Transaction limit exceeded
    """

    test_query = "Why was my payment declined?"

    print("Generating response...")
    response, metrics = llm.generate_with_context(test_query, test_context)

    print(f"\n✅ Response generated successfully!")
    print(f"\n📝 Response:\n{response}")
    print(f"\n📊 Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
