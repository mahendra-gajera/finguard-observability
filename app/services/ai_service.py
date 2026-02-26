"""
AI Service - Pure AI/ML Operations
Handles embeddings, vector search, and LLM generation without observability concerns
"""

import os
import sys
from typing import List, Dict, Tuple
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from embeddings import GeminiEmbeddings
from vector_store import VectorStore
from llm import GeminiLLM

load_dotenv()


class AIService:
    """
    AI Service - Provides pure AI operations

    Responsibilities:
    - Generate text embeddings
    - Search vector database
    - Generate LLM responses
    - Return basic metrics (time, tokens)

    Does NOT handle:
    - Trace management
    - Quality checking
    - Session tracking
    - Hallucination detection
    """

    def __init__(self):
        """Initialize AI components"""
        self.embeddings = GeminiEmbeddings()
        self.vector_store = VectorStore(self.embeddings)
        self.llm = GeminiLLM()
        self.top_k = int(os.getenv("TOP_K_RESULTS", "3"))

    def embed_text(self, text: str) -> Tuple[List[float], Dict]:
        """
        Generate embedding for a single text

        Args:
            text: Input text to embed

        Returns:
            tuple: (embedding_vector, metrics)

        Metrics returned:
            - operation: "embedding"
            - duration_ms: Time taken
            - model: Model name
            - dimension: Vector dimension
            - success: Boolean
        """
        try:
            vector, metrics = self.embeddings.embed_query(text)

            # Return clean, simple metrics
            return vector, {
                "operation": "embedding",
                "duration_ms": metrics.get('embedding_time_ms', 0),
                "model": metrics.get('model', 'unknown'),
                "dimension": metrics.get('dimension', 768),
                "success": True
            }
        except Exception as e:
            return None, {
                "operation": "embedding",
                "duration_ms": 0,
                "error": str(e),
                "success": False
            }

    def search_documents(self, query_vector: List[float], top_k: int = None) -> Tuple[List[Dict], Dict]:
        """
        Search for similar documents using vector

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return (default from config)

        Returns:
            tuple: (list_of_documents, metrics)

        Documents format:
            [{
                "content": "document text",
                "relevance_score": 0.85,
                "metadata": {...}
            }]

        Metrics returned:
            - operation: "search"
            - duration_ms: Time taken
            - results_count: Number of docs found
            - top_relevance: Highest relevance score
            - success: Boolean
        """
        if top_k is None:
            top_k = self.top_k

        try:
            # Use embeddings directly since we already have the vector
            results = self.vector_store.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            # Format results
            documents = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    # Convert distance to relevance score (1 - normalized_distance)
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    relevance = max(0, 1 - distance)

                    documents.append({
                        "content": doc,
                        "relevance_score": round(relevance, 3),
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                    })

            # Return clean metrics
            return documents, {
                "operation": "search",
                "duration_ms": 0,  # Quick operation
                "results_count": len(documents),
                "top_relevance": documents[0]['relevance_score'] if documents else 0,
                "success": True
            }
        except Exception as e:
            return [], {
                "operation": "search",
                "duration_ms": 0,
                "error": str(e),
                "success": False
            }

    def generate_response(self, query: str, context: str) -> Tuple[str, Dict]:
        """
        Generate LLM response with context

        Args:
            query: User's question
            context: Retrieved document context

        Returns:
            tuple: (response_text, metrics)

        Metrics returned:
            - operation: "generation"
            - duration_ms: Time taken
            - input_tokens: Token count
            - output_tokens: Token count
            - total_tokens: Token count
            - model: Model name
            - success: Boolean
        """
        try:
            response, metrics = self.llm.generate_with_context(query, context)

            # Return clean metrics
            return response, {
                "operation": "generation",
                "duration_ms": metrics.get('llm_time_ms', 0),
                "input_tokens": metrics.get('input_tokens', 0),
                "output_tokens": metrics.get('output_tokens', 0),
                "total_tokens": metrics.get('total_tokens', 0),
                "model": metrics.get('model', 'unknown'),
                "success": True
            }
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}", {
                "operation": "generation",
                "duration_ms": 0,
                "error": str(e),
                "success": False
            }

    def index_documents(self, file_path: str) -> Dict:
        """
        Load and index documents from file

        Args:
            file_path: Path to document file

        Returns:
            dict: Indexing results with metrics
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Simple chunking by sections
            chunks = []
            current_chunk = ""
            lines = content.split('\n')

            for line in lines:
                if line.startswith('##') and current_chunk:
                    if len(current_chunk.strip()) > 50:
                        chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    current_chunk += line + '\n'

            if len(current_chunk.strip()) > 50:
                chunks.append(current_chunk.strip())

            # Create metadata
            metadatas = [{"source": file_path, "chunk_id": i} for i in range(len(chunks))]

            # Index documents
            metrics = self.vector_store.add_documents(chunks, metadatas)

            return {
                "success": True,
                "chunks_created": len(chunks),
                "total_time_ms": metrics.get('total_time_ms', 0)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Test the service
if __name__ == "__main__":
    print("Testing AI Service...")

    ai_service = AIService()

    # Test embedding
    test_text = "What are the payment policies?"
    vector, embed_metrics = ai_service.embed_text(test_text)
    print(f"\n1. Embedding Test:")
    print(f"   Vector dimension: {len(vector) if vector else 0}")
    print(f"   Metrics: {embed_metrics}")

    # Test search (if vector exists)
    if vector:
        docs, search_metrics = ai_service.search_documents(vector, top_k=2)
        print(f"\n2. Search Test:")
        print(f"   Documents found: {len(docs)}")
        print(f"   Metrics: {search_metrics}")

    # Test generation
    test_query = "Why was my payment declined?"
    test_context = "Payments can be declined due to insufficient funds or expired cards."
    response, gen_metrics = ai_service.generate_response(test_query, test_context)
    print(f"\n3. Generation Test:")
    print(f"   Response: {response[:100]}...")
    print(f"   Metrics: {gen_metrics}")

    print("\n✅ AI Service test complete!")
