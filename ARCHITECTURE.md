# FinGuard AI - System Architecture (v2.0)

## Overview

FinGuard AI is a production-ready, observable RAG (Retrieval-Augmented Generation) system for fintech customer support. Built with separated AI and Observability services for clean architecture, independent testing, and easy scalability.

**Version:** 2.0 (Separated Services Architecture)
**Last Updated:** February 2026

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit UI Layer                         │
│                         (main.py)                               │
│                                                                 │
│  • Chat interface                                              │
│  • Real-time metrics dashboard                                │
│  • Session statistics                                          │
│  • Query history                                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Orchestrator                             │
│                   (rag_orchestrator.py)                         │
│                                                                 │
│  Coordinates query flow between AI and Observability           │
│                                                                 │
│  Flow:                                                          │
│  1. Start trace (Obs)                                          │
│  2. Generate embedding (AI) → Record span (Obs)                │
│  3. Search documents (AI) → Record span (Obs)                  │
│  4. Generate response (AI) → Record span (Obs)                 │
│  5. Analyze quality (Obs)                                      │
│  6. Complete trace (Obs)                                       │
│  7. Return response + metrics                                  │
└──────────┬─────────────────────────┬────────────────────────────┘
           │                         │
           │                         │
    ┌──────▼────────┐       ┌────────▼─────────────────┐
    │  AI Service   │       │ Observability Service    │
    │               │       │                          │
    │ PURE AI ONLY  │       │ PURE MONITORING ONLY     │
    └───────────────┘       └──────────────────────────┘
```

---

## Architecture Principles

### 1. Separation of Concerns
- **AI Service** handles only AI/ML operations
- **Observability Service** handles only monitoring/tracking
- **RAG Orchestrator** coordinates without business logic
- **UI Layer** handles only presentation

### 2. Loose Coupling
- Services communicate through clean interfaces
- No direct dependencies between AI and Observability
- Easy to swap implementations

### 3. Single Responsibility
- Each component has one well-defined purpose
- Easy to test, maintain, and extend

### 4. Microservice Ready
- Services can be deployed independently
- Horizontal scaling possible
- Container-ready architecture

---

## Component Details

### 1. AI Service (`services/ai_service.py`)

**Purpose:** Pure AI/ML operations

**Responsibilities:**
- Generate text embeddings (768-dim vectors)
- Search vector database for relevant documents
- Generate LLM responses with context
- Index and chunk documents

**Key Methods:**
```python
embed_text(text: str) → (vector, metrics)
search_documents(vector: list, top_k: int) → (docs, metrics)
generate_response(query: str, context: str) → (response, metrics)
index_documents(file_path: str) → (result)
```

**Returns Simple Metrics:**
```python
{
    "operation": "embedding",
    "duration_ms": 45,
    "model": "gemini-embedding-001",
    "dimension": 768,
    "success": True
}
```

**Does NOT Handle:**
- ❌ Trace management
- ❌ Quality checking
- ❌ Session statistics
- ❌ Hallucination detection

**Technology Stack:**
- Google Gemini Embeddings API (768-dim)
- Google Gemini 2.5 Flash LLM
- ChromaDB for vector storage
- Persistent local storage

---

### 2. Observability Service (`services/observability_service.py`)

**Purpose:** Pure monitoring and quality analysis

**Responsibilities:**
- Create and manage execution traces
- Track operation spans (timing, metrics)
- Detect hallucinations and calculate confidence
- Aggregate session-level statistics
- Export metrics for external systems

**Key Methods:**
```python
start_trace(query: str) → trace_id
record_span(trace_id: str, operation: str, metrics: dict) → void
analyze_quality(response: str, docs: list) → quality_metrics
complete_trace(trace_id: str, quality: dict) → full_metrics
get_session_stats() → aggregated_metrics
```

**Returns Rich Metrics:**
```python
{
    "trace_id": "trace_123",
    "total_latency_ms": 3821,
    "embedding_ms": 45,
    "search_ms": 30,
    "llm_ms": 3746,
    "confidence": 0.81,
    "grounding_score": 0.83,
    "hallucination_detected": False,
    "status": "Well-grounded response",
    "total_cost_usd": 0.000008,
    "retrieved_docs_count": 3,
    "top_relevance": 0.89
}
```

**Does NOT Handle:**
- ❌ AI API calls
- ❌ Embedding generation
- ❌ Vector search
- ❌ Response generation

**Quality Analysis:**
- Hallucination detection via grounding score
- Confidence calculation (15% for out-of-scope, 70-95% for grounded)
- Relevance scoring for retrieved documents
- Uncertainty phrase detection

---

### 3. RAG Orchestrator (`rag_orchestrator.py`)

**Purpose:** Coordinate AI and Observability services

**Responsibilities:**
- Manage query lifecycle
- Delegate operations to appropriate service
- Build context from retrieved documents
- Maintain backward compatibility

**Query Flow:**
```
User Query
    ↓
1. obs_service.start_trace(query) → trace_id
    ↓
2. ai_service.embed_text(query) → (vector, metrics)
   obs_service.record_span(trace_id, "embedding", metrics)
    ↓
3. ai_service.search_documents(vector) → (docs, metrics)
   obs_service.record_span(trace_id, "search", metrics)
    ↓
4. Build context from retrieved documents
    ↓
5. ai_service.generate_response(query, context) → (response, metrics)
   obs_service.record_span(trace_id, "generation", metrics)
    ↓
6. obs_service.analyze_quality(response, docs) → quality_metrics
    ↓
7. obs_service.complete_trace(trace_id, quality) → final_metrics
    ↓
Return (response, final_metrics)
```

**Interface:**
```python
query(user_query: str) → (response, metrics)
index_documents(file_path: str) → result
get_session_stats() → stats
get_collection_info() → collection_stats
```

---

### 4. Streamlit UI (`main.py`)

**Purpose:** User interface and visualization

**Features:**
- Chat interface for user queries
- Real-time metrics dashboard
- Latency breakdown visualization
- Cost tracking per query
- Session statistics display
- Query history
- Document indexing controls

**Metrics Displayed:**
- Total latency (with breakdown)
- Token usage (input/output)
- Cost per query and session total
- Confidence score
- Hallucination status
- Retrieved document count and relevance

---

## Data Flow

### Complete Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ User Query: "Why was my payment declined?"                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ START TRACE         │
                  │ trace_id = "trace_1"│
                  └─────────┬───────────┘
                            │
                   ┌────────▼────────┐
                   │ Generate        │
                   │ Embedding       │
                   │                 │
                   │ Input:  query   │
                   │ Output: [768]   │
                   │ Time:   45ms    │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Vector Search   │
                   │                 │
                   │ Input:  vector  │
                   │ Output: 3 docs  │
                   │ Scores: 0.89,   │
                   │         0.82,   │
                   │         0.76    │
                   │ Time:   30ms    │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Build Context   │
                   │                 │
                   │ Format docs     │
                   │ into prompt     │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ LLM Generation  │
                   │                 │
                   │ Input:  120 tok │
                   │ Output: 85 tok  │
                   │ Time:   850ms   │
                   │ Cost:   $0.0027 │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Quality Check   │
                   │                 │
                   │ Grounding: 83%  │
                   │ Confidence: 83% │
                   │ Hallucination:  │
                   │   NO ✓          │
                   │ Status: Well-   │
                   │   grounded      │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ COMPLETE TRACE  │
                   │                 │
                   │ Total: 950ms    │
                   │ Cost: $0.0027   │
                   │ Sources: 3      │
                   └────────┬────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ Response: "Your payment may have been declined for several      │
│ reasons, according to our policies..."                          │
│                                                                  │
│ Metrics:                                                         │
│ • Total latency: 950ms                                          │
│ • Confidence: 83%                                               │
│ • Cost: $0.0027                                                 │
│ • Hallucination: NO                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Trace Structure

### Observability Trace Object

```python
{
    "trace_id": "trace_1_1709123456",
    "query": "Why was my payment declined?",
    "timestamp": "2026-02-26T10:30:00Z",
    "start_time": 1709123456.789,
    "status": "completed",

    "spans": {
        "embedding": {
            "metrics": {
                "operation": "embedding",
                "duration_ms": 45,
                "model": "gemini-embedding-001",
                "dimension": 768,
                "success": true
            },
            "timestamp": "2026-02-26T10:30:00.045Z"
        },
        "search": {
            "metrics": {
                "operation": "search",
                "duration_ms": 30,
                "results_count": 3,
                "top_relevance": 0.89,
                "success": true
            },
            "timestamp": "2026-02-26T10:30:00.075Z"
        },
        "generation": {
            "metrics": {
                "operation": "generation",
                "duration_ms": 850,
                "input_tokens": 120,
                "output_tokens": 85,
                "total_tokens": 205,
                "model": "gemini-2.5-flash",
                "success": true
            },
            "timestamp": "2026-02-26T10:30:00.925Z"
        }
    },

    "metrics": {
        "trace_id": "trace_1_1709123456",
        "query": "Why was my payment declined?",
        "total_latency_ms": 950,
        "embedding_ms": 45,
        "search_ms": 30,
        "llm_ms": 850,
        "other_ms": 25,
        "input_tokens": 120,
        "output_tokens": 85,
        "total_tokens": 205,
        "total_cost_usd": 0.0027,
        "top_relevance": 0.89,
        "results_count": 3,
        "confidence": 0.83,
        "grounding_score": 0.83,
        "hallucination_detected": false,
        "status": "Well-grounded response",
        "retrieved_docs_count": 3,
        "avg_relevance": 0.82,
        "retrieved_docs": [...]
    },

    "end_time": 1709123457.739
}
```

---

## Configuration

### Environment Variables (`.env`)

```bash
# API Keys (Required)
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

# Observability Service
GEMINI_INPUT_COST=0.00001
GEMINI_OUTPUT_COST=0.00003
```

---

## Performance Characteristics

### Typical Query Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Embedding Generation** | 40-60ms | Google Gemini API |
| **Vector Search** | 20-40ms | ChromaDB local |
| **LLM Generation** | 700-1000ms | Depends on response length |
| **Quality Analysis** | < 10ms | Local computation |
| **Total Latency (P50)** | ~900ms | End-to-end |
| **Total Latency (P95)** | ~1200ms | 95th percentile |

### Cost Model

| Metric | Value | Notes |
|--------|-------|-------|
| **Per Query** | $0.001-0.003 | Depends on tokens |
| **Per 100 Queries** | ~$0.20 | Average |
| **Per 1000 Queries** | ~$2.00 | At scale |
| **Input Token Cost** | $0.00001/1K | Gemini pricing |
| **Output Token Cost** | $0.00003/1K | Gemini pricing |

### Throughput

| Metric | Value | Notes |
|--------|-------|-------|
| **Single Instance** | ~10 queries/min | Limited by API latency |
| **Concurrent Queries** | Limited by API rate | Can parallelize |

---

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# Run application
streamlit run app/main.py

# Access at http://localhost:8501
```

### Docker Deployment

```bash
# Set API key
export GOOGLE_API_KEY=your_key

# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

### Production Deployment

**Recommended Setup:**
- Use managed Kubernetes (GKE, EKS, AKS)
- Separate services into different pods
- Use managed vector database (Pinecone, Weaviate)
- Integrate with production observability (DataDog, Prometheus)
- Add API gateway for rate limiting
- Implement caching layer (Redis)

---

## Observability & Monitoring

### Built-in Metrics

**Per-Query Metrics:**
- Total latency with breakdown
- Token usage (input/output/total)
- Cost calculation
- Confidence score (0-1)
- Grounding score (0-1)
- Hallucination detection (boolean)
- Retrieved document count and relevance

**Session Metrics:**
- Total queries processed
- Average latency
- P95 latency
- Min/max latency
- Total cost
- Average cost per query
- Hallucination rate (%)
- Success rate (%)
- Session duration

### Quality Assurance

**Hallucination Detection Algorithm:**
1. Extract key words from response
2. Compare with retrieved document content
3. Calculate grounding score (% overlap)
4. Detect uncertainty phrases ("I don't have...")
5. Classify:
   - Out-of-scope: 15% confidence
   - Low grounding (< 60%): Flag as hallucination
   - Well-grounded (≥ 60%): High confidence (70-95%)

---

## Testing

### Unit Tests

**23 tests covering:**
- AI Service operations (4 tests)
- Observability Service tracking (13 tests)
- RAG Pipeline orchestration (3 tests)
- LLM operations (4 tests)

**Run tests:**
```bash
python tests/run_tests.py
```

### Integration Testing

**Test orchestrator:**
```bash
python app/rag_orchestrator.py
```

**Test UI:**
```bash
streamlit run app/main.py
```

---

## Security Considerations

### API Key Management
- Store in `.env` file (not committed to Git)
- Use environment variables in production
- Rotate keys regularly

### Data Privacy
- All processing local except API calls
- No data sent to third parties (except Gemini API)
- ChromaDB stored locally
- No PII logging

### Rate Limiting
- Managed by Google Gemini API quotas
- Implement client-side rate limiting for production

---

## Scalability & Future Enhancements

### Current Limitations
- Single-threaded query processing
- Local ChromaDB (single node)
- API rate limits

### Phase 2: Horizontal Scaling
- Add Redis caching for repeated queries
- Implement async processing
- Use managed vector database
- Deploy services independently
- Add load balancer

### Phase 3: Advanced Features
- Multi-model support (OpenAI, Anthropic)
- Pluggable observability backends (Prometheus, DataDog)
- A/B testing framework
- Advanced prompt management
- Fine-tuned embedding models

### Phase 4: Production Hardening
- Circuit breakers
- Retry logic with exponential backoff
- Request queuing
- Health checks
- Distributed tracing (OpenTelemetry)

---

## Technology Stack

### Core Technologies
- **Python:** 3.11+
- **Streamlit:** UI framework
- **Google Gemini:** LLM & embeddings
- **ChromaDB:** Vector database
- **LangChain:** RAG utilities

### Dependencies
```
streamlit>=1.31.0
google-generativeai>=0.8.6
chromadb>=1.5.1
langchain>=1.2.8
pandas>=2.2.0
python-dotenv>=1.0.0
```

---

## Architecture Decision Records (ADRs)

### ADR-001: Separated Services Architecture
**Decision:** Separate AI and Observability into independent services
**Date:** 2026-02-26
**Status:** Implemented

**Context:** Original monolithic architecture had tight coupling

**Decision:** Create AI Service, Observability Service, and RAG Orchestrator

**Consequences:**
- ✅ Clean separation of concerns
- ✅ Independent testing
- ✅ Swappable components
- ⚠️ Slightly more complex structure

### ADR-002: Google Gemini as Primary LLM
**Decision:** Use Google Gemini 2.5 Flash
**Status:** Implemented

**Rationale:**
- Free tier available
- Fast inference (700-1000ms)
- Low cost ($0.001-0.003 per query)
- Good quality responses

**Alternatives Considered:**
- OpenAI GPT-4: Higher cost
- Anthropic Claude: No free tier
- Local models: Slower, quality varies

### ADR-003: ChromaDB for Vector Storage
**Decision:** Use ChromaDB for local vector storage
**Status:** Implemented

**Rationale:**
- Simple setup (no external service)
- Persistent storage
- Good performance for < 100K docs
- Python-native

**Future:** Consider Pinecone/Weaviate for scale

---

## Support & Contribution

### Documentation
- **README.md** - Quick start guide
- **ARCHITECTURE_FINAL.md** - This document
- **REFACTOR_SUMMARY.md** - Refactoring details
- **TEST_SUMMARY.md** - Testing documentation

### Issues & Support
- Report bugs via GitHub Issues
- Feature requests welcome
- Pull requests accepted

---

## Version History

### v2.0 (2026-02-26) - Separated Services Architecture
- ✅ Created AI Service (pure AI operations)
- ✅ Created Observability Service (pure monitoring)
- ✅ Created RAG Orchestrator (coordination)
- ✅ Updated UI to use new architecture
- ✅ All tests passing
- ✅ Zero breaking changes

### v1.0 (2026-02-25) - Initial Release
- Observable RAG pipeline
- Google Gemini integration
- ChromaDB vector store
- Streamlit UI
- Basic testing

---

## License

MIT License - See LICENSE file for details

---

**Architecture designed for:** Clean separation, independent testing, easy scaling, production readiness

**Maintained by:** FinGuard AI Team
**Last Review:** February 2026
