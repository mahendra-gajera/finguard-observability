"""
Embeddings Module - Google Gemini Embeddings with Observability
"""

import time
import os
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbeddings:
    """Google Gemini Embeddings with timing and observability"""

    def __init__(self):
        """Initialize Gemini embeddings"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model_name = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
        self.dimension = 768  # Gemini embedding dimension

    def embed_query(self, text: str) -> tuple:
        """
        Generate embedding for a single query

        Returns:
            tuple: (embedding_vector, metrics_dict)
        """
        start_time = time.time()

        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query"
            )

            embedding = result['embedding']

            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "embedding_time_ms": round(duration_ms, 2),
                "model": self.model_name,
                "dimension": self.dimension,
                "input_length": len(text),
                "success": True
            }

            return embedding, metrics

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "embedding_time_ms": round(duration_ms, 2),
                "model": self.model_name,
                "error": str(e),
                "success": False
            }

            raise Exception(f"Embedding generation failed: {str(e)}")

    def embed_documents(self, texts: List[str]) -> tuple:
        """
        Generate embeddings for multiple documents (batch)

        Returns:
            tuple: (list_of_embeddings, metrics_dict)
        """
        start_time = time.time()

        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])

            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "embedding_time_ms": round(duration_ms, 2),
                "batch_size": len(texts),
                "avg_time_per_doc": round(duration_ms / len(texts), 2),
                "model": self.model_name,
                "success": True
            }

            return embeddings, metrics

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "embedding_time_ms": round(duration_ms, 2),
                "batch_size": len(texts),
                "error": str(e),
                "success": False
            }

            raise Exception(f"Batch embedding failed: {str(e)}")


if __name__ == "__main__":
    # Test embeddings
    embeddings = GeminiEmbeddings()

    # Test single query
    test_query = "Why was my payment declined?"
    embedding, metrics = embeddings.embed_query(test_query)

    print(f"✅ Embedding generated successfully!")
    print(f"📊 Metrics: {metrics}")
    print(f"📐 Embedding dimension: {len(embedding)}")
    print(f"🔢 First 5 values: {embedding[:5]}")
