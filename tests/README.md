# FinGuard AI - Unit Tests

## Overview
Basic unit tests for the Observable RAG system components.

## Test Coverage

### 1. `test_observability.py`
Tests for observability module:
- **Hallucination detection**
  - Well-grounded responses (should pass)
  - Out-of-scope questions (should be uncertain)
  - Potential hallucinations (should be flagged)
  - Edge cases (empty responses, no sources)
- **Relevance scoring**
  - Calculate average relevance
  - Handle empty document lists
- **Observability tracker**
  - Trace creation and management
  - Span tracking
  - Session metrics aggregation

### 2. `test_embeddings.py`
Tests for embeddings module:
- Query embedding generation
- Batch document embedding
- Error handling
- Metrics tracking (timing, dimensions)

### 3. `test_llm.py`
Tests for LLM module:
- Text generation
- Context-based generation (RAG)
- Token counting
- Cost calculation accuracy
- Error handling

### 4. `test_rag_pipeline.py`
Tests for RAG pipeline:
- Pipeline initialization
- Query processing flow
- Response and metrics structure
- Document chunking logic

## Running Tests

### Run all tests
```bash
cd C:\Users\mgajera\poc\finguard-observability
python tests/run_tests.py
```

### Run specific test file
```bash
python -m unittest tests/test_observability.py
python -m unittest tests/test_embeddings.py
python -m unittest tests/test_llm.py
python -m unittest tests/test_rag_pipeline.py
```

### Run specific test class
```bash
python -m unittest tests.test_observability.TestHallucinationDetection
```

### Run specific test method
```bash
python -m unittest tests.test_observability.TestHallucinationDetection.test_well_grounded_response
```

## Test Results Format

```
test_well_grounded_response (test_observability.TestHallucinationDetection) ... ok
test_out_of_scope_response (test_observability.TestHallucinationDetection) ... ok
test_potential_hallucination (test_observability.TestHallucinationDetection) ... ok

----------------------------------------------------------------------
Ran 15 tests in 0.045s

OK
```

## Key Test Scenarios

### Hallucination Detection Tests

#### ✅ Well-Grounded Response
- **Input:** Response matches retrieved documents
- **Expected:** No hallucination, high confidence (>50%)
- **Example:** Payment decline reasons from policy docs

#### ✅ Out-of-Scope Question
- **Input:** Question not covered by documents
- **Expected:** No hallucination (properly uncertain), LOW confidence (<30%)
- **Example:** "Who is prime minister of India?"

#### ⚠️ Potential Hallucination
- **Input:** Response contains info not in sources
- **Expected:** Hallucination flagged, low grounding score (<60%)
- **Example:** "Uses blockchain" when docs don't mention it

### Cost Calculation Test
- **Input:** 1000 input tokens, 1000 output tokens
- **Expected Cost:** $0.00004
  - Input: 1000/1000 × $0.00001 = $0.00001
  - Output: 1000/1000 × $0.00003 = $0.00003
  - Total: $0.00004

## Mocking Strategy

Tests use Python's `unittest.mock` to avoid real API calls:

```python
@patch('embeddings.genai')
def test_embed_query_success(self, mock_genai):
    # Mock API response
    mock_response = {'embedding': [0.1] * 768}
    mock_genai.embed_content.return_value = mock_response

    # Test code here
```

## Test Dependencies

- `unittest` (built-in)
- `unittest.mock` (built-in)
- No external dependencies required

## Adding New Tests

1. Create new test file: `test_<module>.py`
2. Import module to test
3. Create test class inheriting from `unittest.TestCase`
4. Add test methods (must start with `test_`)
5. Use assertions to verify behavior

Example:
```python
import unittest

class TestMyFeature(unittest.TestCase):
    def test_feature_works(self):
        result = my_function()
        self.assertEqual(result, expected_value)
```

## Test Best Practices

1. **One assertion focus per test** - Test one thing at a time
2. **Clear test names** - `test_what_when_expected`
3. **Arrange-Act-Assert** - Setup, execute, verify
4. **Mock external dependencies** - Don't call real APIs
5. **Test edge cases** - Empty inputs, errors, boundaries

## Known Limitations

- Tests use mocks, not real API calls
- No integration tests with actual Google Gemini API
- No performance/load testing
- No end-to-end Streamlit UI testing

## Future Test Additions

- Integration tests with real APIs (separate test environment)
- Performance benchmarks
- Load testing for concurrent queries
- UI component testing
- End-to-end workflow tests

## Troubleshooting

### Import errors
If you see "ModuleNotFoundError", ensure you're running from the project root:
```bash
cd C:\Users\mgajera\poc\finguard-observability
python tests/run_tests.py
```

### API key warnings
Tests use mocks, so you don't need a real API key. If you see warnings, they can be ignored during testing.

### Emoji encoding errors
Tests avoid emojis in assertions to prevent Windows encoding issues.

---

**Test Coverage:** ~60% (core logic covered, UI and integrations not covered)
**Test Type:** Unit tests with mocks
**Run Time:** < 1 second for all tests
