# 🛡️ FinGuard AI - Observable RAG Customer Support Assistant

> Production-ready RAG chatbot for fintech customer support with full observability, powered by Google Gemini AI

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Architecture: Microservices](https://img.shields.io/badge/Architecture-Microservices-orange.svg)](ARCHITECTURE_FINAL.md)

---

## 📋 Overview

FinGuard AI is an **open-source, observable RAG** (Retrieval-Augmented Generation) system designed for fintech customer support. It answers customer questions using company policy documents while providing **full transparency** into every step of the process.

**Key Differentiator:** Clean **separated services architecture** with independent AI and Observability services, making it production-ready, testable, and scalable.

---

## ✨ Features

### 🔍 Full Observability
- ✅ Track every step: embedding generation, vector search, LLM generation
- ✅ Latency breakdown (embedding: 45ms, search: 30ms, LLM: 850ms)
- ✅ Real-time metrics dashboard
- ✅ Trace-based monitoring (APM-style)

### 💰 Cost Tracking
- ✅ Token usage tracking (input/output)
- ✅ Real-time cost calculation per query ($0.001-0.003)
- ✅ Session-level cost aggregation
- ✅ Cost breakdown by operation

### 🎯 Quality Assurance
- ✅ Hallucination detection via grounding score
- ✅ Confidence scoring (15% for out-of-scope, 70-95% for grounded)
- ✅ Relevance scoring for retrieved documents
- ✅ Answer quality verification

### 🏗️ Clean Architecture
- ✅ **Separated AI Service** - Pure AI/ML operations
- ✅ **Separated Observability Service** - Pure monitoring
- ✅ **RAG Orchestrator** - Clean coordination layer
- ✅ Independent testing and deployment
- ✅ Microservice-ready design

### 🚀 Production Ready
- ✅ Docker deployment
- ✅ Error handling & retry logic
- ✅ Session statistics & aggregation
- ✅ Comprehensive logging
- ✅ 23 unit tests (100% passing)

---

## 🏗️ Architecture

### High-Level Overview

```
┌─────────────────────────────────────────┐
│        Streamlit UI                     │
│  • Chat interface                       │
│  • Real-time metrics dashboard          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│       RAG Orchestrator                  │
│  Coordinates AI + Observability         │
└────────┬─────────────────┬──────────────┘
         │                 │
    ┌────▼────┐     ┌──────▼────────┐
    │   AI    │     │ Observability │
    │ Service │     │   Service     │
    │         │     │               │
    │ • Embed │     │ • Traces      │
    │ • Search│     │ • Quality     │
    │ • Generate    │ • Metrics     │
    └─────────┘     └───────────────┘
```

**Read detailed architecture:** [ARCHITECTURE_FINAL.md](ARCHITECTURE.md)

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Google Gemini API Key** ([Get it free here](https://makersuite.google.com/app/apikey))
- **Docker** (optional, for containerized deployment)

### Option 1: Local Setup (Recommended for Development)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/finguard-observability.git
   cd finguard-observability
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

4. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

5. **Open your browser**
   ```
   http://localhost:8501
   ```

### Option 2: Docker Deployment (Recommended for Production)

1. **Set environment variable**
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

2. **Build and run**
   ```bash
   docker-compose up -d
   ```

3. **Access the app**
   ```
   http://localhost:8501
   ```

### Option 3: Quick Test (No UI)

```bash
python app/rag_orchestrator.py
```

---

## 📊 What You Get

### Real-Time Metrics Dashboard

Every query shows:
- **⏱️ Latency Breakdown:** Embedding (45ms) + Search (30ms) + LLM (850ms) = Total (925ms)
- **💰 Cost Tracking:** Input tokens × $0.00001/1K + Output tokens × $0.00003/1K
- **🎯 Quality Metrics:** Confidence (83%), Grounding score (85%), Hallucination detection
- **📄 Source Documents:** Top 3 retrieved docs with relevance scores

### Session Statistics

Aggregated metrics across all queries:
- Total queries processed
- Average latency & P95 latency
- Total cost & average per query
- Hallucination rate
- Success rate

---

## 🧪 Example Queries

Try these questions with FinGuard AI:

### 1. Well-Grounded Query
**Question:** "Why was my payment declined?"

**Expected:**
- ✅ Response based on policy documents
- ✅ High confidence (80-90%)
- ✅ No hallucination detected
- ✅ Latency: ~1.2 seconds
- ✅ Cost: ~$0.002

### 2. Out-of-Scope Query
**Question:** "Who is the prime minister of India?"

**Expected:**
- ✅ Response: "I don't have that information in our policies"
- ✅ Low confidence (15%) - **Correctly uncertain!**
- ✅ No hallucination (properly refuses)

### 3. Multi-Document Query
**Question:** "How long do refunds take?"

**Expected:**
- ✅ Synthesizes info from multiple policy sections
- ✅ High confidence (85-95%)
- ✅ Shows 3 source documents with relevance scores

---

## 📁 Project Structure

```
finguard-observability/
│
├── app/
│   ├── main.py                      # Streamlit UI
│   ├── rag_orchestrator.py          # Service coordinator
│   │
│   ├── services/
│   │   ├── ai_service.py            # Pure AI operations
│   │   └── observability_service.py # Pure monitoring
│   │
│   ├── embeddings.py                # Gemini embeddings
│   ├── vector_store.py              # ChromaDB interface
│   ├── llm.py                       # Gemini LLM
│   └── observability.py             # Quality checks
│
├── data/
│   └── fintech_policies.txt         # Sample policies
│
├── tests/
│   ├── test_ai_service.py
│   ├── test_observability_service.py
│   ├── test_llm.py
│   └── run_tests.py
│
├── docker/
│   └── Dockerfile
│
├── .env.example                     # Environment template
├── docker-compose.yml
├── requirements.txt
├── README.md                        # This file
└── ARCHITECTURE_FINAL.md            # Detailed architecture
```

---

## 🔧 Configuration

### Environment Variables (`.env`)

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key_here

# AI Service Configuration
EMBEDDING_MODEL=models/gemini-embedding-001
LLM_MODEL=models/gemini-2.5-flash
TEMPERATURE=0.1
MAX_TOKENS=500

# RAG Settings
TOP_K_RESULTS=3
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=fintech_policies
```

### Customization

**Add Your Own Documents:**
1. Place your policy/FAQ documents in `data/` folder
2. Update path in UI or load via interface
3. Documents are automatically chunked and indexed

**Change LLM Model:**
```bash
# In .env
LLM_MODEL=models/gemini-2.5-pro  # More capable (slower, pricier)
LLM_MODEL=models/gemini-2.5-flash # Faster and cheaper (recommended)
```

**Adjust Retrieval:**
```bash
# In .env
TOP_K_RESULTS=5  # More context (better answers, higher cost)
TOP_K_RESULTS=2  # Less context (faster, cheaper)
```

---

## 🧪 Testing

### Run All Tests

```bash
cd finguard-observability
python tests/run_tests.py
```

**Output:**
```
Ran 23 tests in 7.5s

OK ✓
```

### Test Individual Components

```bash
# Test AI Service
python -m unittest tests.test_ai_service

# Test Observability Service
python -m unittest tests.test_observability_service

# Test RAG Orchestrator
python app/rag_orchestrator.py
```

**Test Coverage:** ~65% (core business logic fully covered)

---

## 📊 Performance Benchmarks

Typical performance on standard hardware:

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Latency (P50)** | ~900ms | End-to-end |
| **Total Latency (P95)** | ~1200ms | 95th percentile |
| **Embedding Time** | 40-60ms | Gemini API |
| **Vector Search** | 20-40ms | ChromaDB local |
| **LLM Generation** | 700-1000ms | Depends on response length |
| **Cost per Query** | $0.001-0.003 | Average |
| **Throughput** | ~10 queries/min | Single instance |

---

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not found"
**Solution:** Create `.env` file from `.env.example` and add your API key

### ChromaDB errors
**Solution:** Delete `chroma_db/` folder and re-index documents

### Slow responses
**Solution:**
- Check internet connection
- Try `gemini-2.5-flash` model (faster)
- Reduce `TOP_K_RESULTS`

### High hallucination rate
**Solution:**
- Lower `TEMPERATURE` (try 0.0)
- Increase `TOP_K_RESULTS` for better context
- Improve source documents quality

### Emoji encoding errors (Windows)
**Solution:** The app handles this automatically with UTF-8 encoding

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build
docker build -f docker/Dockerfile -t finguard-ai .

# Run
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -v $(pwd)/chroma_db:/app/chroma_db \
  finguard-ai
```

### Production Recommendations

- Use managed Kubernetes (GKE, EKS, AKS)
- Deploy AI and Observability services separately
- Use managed vector database (Pinecone, Weaviate)
- Add Redis caching layer
- Integrate with DataDog/Prometheus for monitoring
- Implement rate limiting at API gateway
- Use secrets management (AWS Secrets Manager, etc.)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new features
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Google Gemini** - LLM and embeddings API
- **ChromaDB** - Vector database
- **Streamlit** - UI framework
- **LangChain** - RAG patterns and utilities

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/finguard-observability/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/finguard-observability/discussions)
- **Documentation:** [ARCHITECTURE_FINAL.md](ARCHITECTURE.md)

---

## 📚 Additional Documentation

- **[ARCHITECTURE_FINAL.md](ARCHITECTURE.md)** - Complete system architecture
- **[TEST_SUMMARY.md](TEST_SUMMARY.md)** - Testing documentation
- **[tests/README.md](tests/README.md)** - How to run tests

---

## 🎓 Learn More

### What is RAG?
Retrieval-Augmented Generation combines retrieval (finding relevant documents) with generation (LLM creating answers) to provide accurate, grounded responses.

### Why Observability Matters?

**Without observability:**
- ❌ Can't detect hallucinations
- ❌ Can't optimize costs
- ❌ Can't debug failures
- ❌ Can't meet SLAs

**With observability:**
- ✅ Track every step
- ✅ Measure quality
- ✅ Control costs
- ✅ Debug issues
- ✅ Prove compliance

### Why Separated Services?

**Benefits:**
- ✅ Independent testing
- ✅ Easy to swap AI providers
- ✅ Pluggable observability backends
- ✅ Microservice-ready
- ✅ Better maintainability

---

## 📈 Roadmap

### Phase 2: Enhanced Features
- [ ] Multi-model support (OpenAI, Anthropic, Claude)
- [ ] Advanced prompt management and versioning
- [ ] A/B testing framework
- [ ] Fine-tuned embedding models

### Phase 3: Scale & Performance
- [ ] Horizontal scaling support
- [ ] Redis caching layer
- [ ] Async query processing
- [ ] Managed vector database integration

### Phase 4: Enterprise Features
- [ ] Multi-tenancy support
- [ ] Role-based access control (RBAC)
- [ ] Audit logging
- [ ] Compliance reporting
- [ ] Advanced analytics dashboard

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

