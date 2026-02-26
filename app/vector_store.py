"""
Vector Store Module - ChromaDB with Observability
"""

import time
import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

# Safe print function for Windows emoji issues
def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'ignore').decode('ascii'))

class VectorStore:
    """ChromaDB vector store with search timing"""

    def __init__(self, embedding_function):
        """Initialize ChromaDB"""
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = os.getenv("COLLECTION_NAME", "fintech_policies")

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embedding_function = embedding_function

        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            safe_print(f"✅ Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "FinGuard fintech policies and FAQs"}
            )
            safe_print(f"✅ Created new collection: {self.collection_name}")

    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> Dict:
        """
        Add documents to vector store

        Returns:
            Dict with metrics
        """
        start_time = time.time()

        try:
            # Generate embeddings
            embeddings, embed_metrics = self.embedding_function.embed_documents(texts)

            # Generate IDs
            ids = [f"doc_{i}" for i in range(len(texts))]

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas if metadatas else [{}] * len(texts),
                ids=ids
            )

            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "operation": "add_documents",
                "num_documents": len(texts),
                "total_time_ms": round(duration_ms, 2),
                "embedding_time_ms": embed_metrics['embedding_time_ms'],
                "indexing_time_ms": round(duration_ms - embed_metrics['embedding_time_ms'], 2),
                "success": True
            }

            return metrics

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "operation": "add_documents",
                "error": str(e),
                "total_time_ms": round(duration_ms, 2),
                "success": False
            }

            raise Exception(f"Failed to add documents: {str(e)}")

    def similarity_search(self, query: str, k: int = 3) -> tuple:
        """
        Search for similar documents

        Returns:
            tuple: (results_list, metrics_dict)
        """
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding, embed_metrics = self.embedding_function.embed_query(query)

            # Search
            search_start = time.time()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=['documents', 'metadatas', 'distances']
            )
            search_time_ms = (time.time() - search_start) * 1000

            # Format results
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'relevance_score': round(1 - results['distances'][0][i], 3) if results['distances'] else None
                    })

            total_duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "operation": "similarity_search",
                "query_length": len(query),
                "embedding_time_ms": embed_metrics['embedding_time_ms'],
                "search_time_ms": round(search_time_ms, 2),
                "total_time_ms": round(total_duration_ms, 2),
                "results_count": len(formatted_results),
                "top_score": formatted_results[0]['relevance_score'] if formatted_results else 0,
                "avg_score": round(sum(r['relevance_score'] for r in formatted_results) / len(formatted_results), 3) if formatted_results else 0,
                "success": True
            }

            return formatted_results, metrics

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            metrics = {
                "operation": "similarity_search",
                "error": str(e),
                "total_time_ms": round(duration_ms, 2),
                "success": False
            }

            raise Exception(f"Search failed: {str(e)}")

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "active"
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "status": "error"
            }


if __name__ == "__main__":
    # Test vector store
    from embeddings import GeminiEmbeddings

    embeddings = GeminiEmbeddings()
    vector_store = VectorStore(embeddings)

    # Test adding documents
    test_docs = [
        "Payments may be declined due to insufficient funds.",
        "Refunds take 5-7 business days to process."
    ]
    test_metadata = [
        {"source": "payment_policy", "section": "declines"},
        {"source": "refund_policy", "section": "timing"}
    ]

    print("Adding test documents...")
    add_metrics = vector_store.add_documents(test_docs, test_metadata)
    print(f"📊 Add Metrics: {add_metrics}")

    # Test search
    print("\nTesting search...")
    results, search_metrics = vector_store.similarity_search("Why was payment declined?", k=2)
    print(f"📊 Search Metrics: {search_metrics}")
    print(f"📄 Results: {len(results)} documents found")
    for i, result in enumerate(results, 1):
        safe_print(f"\n{i}. Score: {result['relevance_score']}")
        safe_print(f"   Content: {result['content'][:100]}...")
