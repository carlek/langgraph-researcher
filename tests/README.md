# Test Suite

This directory contains unit tests for the langgraph-researcher pipeline. All LLM calls are mocked using `unittest.mock` to enable fast, deterministic testing without requiring API keys.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[test]"
```

### Run all tests

```bash
pytest
```

### Run with coverage

```bash
pytest --cov=research_pipeline --cov-report=html
```

### Run specific test file

```bash
pytest tests/test_chains.py
pytest tests/test_nodes.py
pytest tests/test_schemas.py
```

### Run specific test class

```bash
pytest tests/test_chains.py::TestGenerateAnalystPersonas
```

### Run specific test

```bash
pytest tests/test_chains.py::TestGenerateAnalystPersonas::test_generate_analysts
```

## Test Structure

### `conftest.py`
Shared pytest fixtures and mock setup:
- `mock_llm`: Mock ChatAnthropic instance
- `mock_llm_for`: Patches the `llm_for()` function globally
- `mock_analyst`: Sample Analyst schema instance
- `mock_perspectives`: Sample Perspectives response
- `mock_search_query`: Sample SearchQuery response

### `test_schemas.py`
Tests for Pydantic models:
- `TestAnalyst`: Validation, freezing, whitespace stripping
- `TestSearchQuery`: Required fields validation
- `TestPerspectives`: Creation with multiple analysts

### `test_chains.py`
Tests for LLM chain functions:
- `TestEnsureFinalMessage`: Message formatting helper
- `TestGenerateAnalystPersonas`: Analyst persona generation
- `TestAskAnalystQuestion`: Question generation
- `TestPlanSearchQuery`: Search query planning
- `TestAnswerAsExpert`: Expert answer generation
- `TestWriteInterviewSection`: Section writing
- `TestWriteReportBody`: Report body synthesis
- `TestWriteBookend`: Introduction/conclusion generation

### `test_nodes.py`
Tests for graph node handlers:
- `TestCreateAnalysts`: Analyst generation node
- `TestHumanFeedback`: Human feedback (no-op) node
- `TestAskQuestion`: Question generation node
- `TestSearchWeb`: Web search node
- `TestSearchWikipedia`: Wikipedia search node
- `TestAnswerQuestion`: Expert answer node
- `TestSaveInterview`: Interview saving node
- `TestWriteSection`: Section writing node
- `TestWriteReport`: Report writing node
- `TestWriteIntroduction`: Introduction writing node
- `TestWriteConclusion`: Conclusion writing node

## Mocking Strategy

The test suite uses `unittest.mock` to mock all LLM interactions:

```python
@pytest.fixture
def mock_llm_for(mock_llm):
    """Mock the llm_for function."""
    with patch("research_pipeline.llm.llm_for", return_value=mock_llm) as mock_fn:
        yield mock_fn
```

This allows tests to:
1. Run without API keys
2. Execute deterministically with predictable responses
3. Verify correct function arguments via `assert_called_with()`
4. Run quickly without network latency

## Adding New Tests

When adding tests for new functions:

1. Import the function/class to test
2. Use the `mock_llm_for` fixture to mock LLM calls
3. Set up mock return values appropriate to the function
4. Call the function with test data
5. Assert on the return value and mock call arguments

Example:

```python
def test_my_new_function(mock_llm_for, mock_analyst):
    """Test description."""
    # Setup mock
    mock_llm_for.return_value.invoke.return_value = AIMessage(content="Response")
    
    # Call function
    result = my_new_function(mock_analyst, ["context"])
    
    # Assert
    assert result is not None
    mock_llm_for.assert_called_with("writer")
```

## CI/CD Integration

To integrate these tests into CI/CD, add to your workflow:

```yaml
- name: Run tests
  run: pytest --cov=research_pipeline --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```
