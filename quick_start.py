"""
FinGuard AI - Quick Start Script
Run this to test the RAG pipeline before launching the UI
"""

import sys
import os
import io

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from rag_pipeline import ObservableRAGPipeline
import json

def print_banner():
    print("=" * 70)
    print("  FinGuard AI - Observable RAG Quick Start")
    print("=" * 70)
    print()

def print_metrics(metrics):
    print(f"\n{'='*70}")
    print("📊 Query Metrics")
    print('='*70)
    print(f"⏱️  Total Latency:    {metrics['total_latency_ms']:.0f}ms")
    print(f"   ├─ Embedding:     {metrics['embedding_ms']:.0f}ms")
    print(f"   ├─ Search:        {metrics['search_ms']:.0f}ms")
    print(f"   └─ LLM:           {metrics['llm_ms']:.0f}ms")
    print()
    print(f"💰 Cost:             ${metrics['total_cost_usd']:.6f}")
    print(f"   ├─ Input tokens:  {metrics['input_tokens']}")
    print(f"   ├─ Output tokens: {metrics['output_tokens']}")
    print(f"   └─ Total tokens:  {metrics['total_tokens']}")
    print()
    print(f"🎯 Quality:")
    print(f"   ├─ Confidence:    {metrics['confidence']*100:.0f}%")
    print(f"   ├─ Grounding:     {metrics['grounding_score']*100:.0f}%")
    print(f"   ├─ Hallucination: {'❌ YES' if metrics['hallucination_detected'] else '✅ NO'}")
    print(f"   └─ Status:        {metrics['status']}")
    print()
    print(f"📄 Sources:          {len(metrics['retrieved_docs'])} documents")
    for i, doc in enumerate(metrics['retrieved_docs'][:3], 1):
        print(f"   {i}. Relevance: {doc['relevance_score']:.2f}")
        print(f"      Content: {doc['content'][:80]}...")

def main():
    print_banner()

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables")
        print("\n📝 Please follow these steps:")
        print("   1. Get your free API key from: https://makersuite.google.com/app/apikey")
        print("   2. Copy .env.example to .env")
        print("   3. Add your API key to .env")
        print()
        return

    print("🔧 Initializing FinGuard AI RAG Pipeline...")
    try:
        rag = ObservableRAGPipeline()
        print("✅ RAG Pipeline initialized successfully!\n")
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return

    # Index documents
    print("📚 Loading and indexing fintech policies...")
    data_path = os.path.join(os.path.dirname(__file__), "data/fintech_policies.txt")

    if not os.path.exists(data_path):
        print(f"❌ Policy file not found: {data_path}")
        return

    try:
        result = rag.index_documents(data_path)
        print(f"✅ Indexed {result['chunks_created']} chunks in {result['total_time_ms']:.0f}ms\n")
    except Exception as e:
        print(f"❌ Indexing failed: {str(e)}")
        return

    # Test queries
    test_queries = [
        "Why was my payment declined?",
        "How long do refunds take?",
        "What are forex charges?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Test Query {i}/{len(test_queries)}: {query}")
        print('='*70)

        try:
            response, metrics = rag.query(query)

            print(f"\n💬 Response:")
            print(f"{response}\n")

            print_metrics(metrics)

        except Exception as e:
            print(f"❌ Query failed: {str(e)}")

    # Session statistics
    print(f"\n{'='*70}")
    print("📈 Session Statistics")
    print('='*70)
    stats = rag.get_session_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'cost' in key:
                print(f"   {key:25s}: ${value:.4f}")
            else:
                print(f"   {key:25s}: {value:.2f}")
        else:
            print(f"   {key:25s}: {value}")

    print(f"\n{'='*70}")
    print("✅ Quick Start Complete!")
    print('='*70)
    print("\n🚀 Ready to launch? Run: streamlit run app/main.py")
    print()

if __name__ == "__main__":
    main()
