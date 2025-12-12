# Testing Guide

This document provides comprehensive guidance on testing the Coordination Gap Detector application.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Milestone 1E Tests](#milestone-1e-tests)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)

## Quick Start

### Prerequisites

Make sure you have the development environment set up:

```bash
# Install dependencies
uv pip install -e ".[dev]"

# Start required services
docker compose up -d
```

### Run All Tests

```bash
# Run all tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=src --cov-report=html --cov-report=term-missing
```

## Test Structure

The test suite is organized by component:

```
tests/
├── conftest.py                     # Shared fixtures and configuration
├── test_embeddings.py              # Embedding generation tests
├── test_text_processing.py         # Text utility tests
├── test_vector_store.py            # Vector store integration tests
├── test_mock_data.py               # Mock data generation tests
├── test_detection/                 # Gap detection tests
├── test_ranking/                   # Ranking algorithm tests
├── test_search/                    # Search functionality tests
├── test_ingestion/                 # Data ingestion tests
└── test_integration/               # End-to-end integration tests
```

## Running Tests

### By Module

```bash
# Vector store operations
uv run pytest tests/test_vector_store.py -v

# Embedding generation
uv run pytest tests/test_embeddings.py -v

# Text processing utilities
uv run pytest tests/test_text_processing.py -v

# Mock data generation
uv run pytest tests/test_mock_data.py -v
```

### By Test Class

```bash
# Run specific test class
uv run pytest tests/test_embeddings.py::TestEmbeddingGenerator -v

# Run specific test method
uv run pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_generate_embedding_single_text -v
```

### By Pattern Matching

```bash
# Run all tests with "search" in the name
uv run pytest -k "search" -v

# Run all tests except integration tests
uv run pytest -k "not integration" -v

# Run tests matching multiple patterns
uv run pytest -k "embedding or vector" -v
```

### With Different Output Formats

```bash
# Minimal output
uv run pytest -q

# Verbose output with test names
uv run pytest -v

# Very verbose with full diff output
uv run pytest -vv

# Show test output even for passing tests
uv run pytest -s

# Show test durations
uv run pytest --durations=10
```

### Debugging Failed Tests

```bash
# Stop on first failure
uv run pytest -x

# Drop into debugger on failure
uv run pytest --pdb

# Run only failed tests from last run
uv run pytest --lf

# Run failed tests first, then others
uv run pytest --ff
```

### Performance Options

```bash
# Run tests in parallel (requires pytest-xdist)
uv run pytest -n auto

# Run with specific number of workers
uv run pytest -n 4
```

## Milestone 1E Tests

Milestone 1E focuses on vector store and embedding functionality. Here's how to test these components:

### Vector Store Tests (test_vector_store.py)

**28 test cases covering:**
- Initialization and connection
- Single document insertion
- Batch document insertion
- Semantic similarity search
- Search with threshold filtering
- Search with metadata filtering
- Document deletion (single and batch)
- Collection clearing
- Document retrieval by ID
- Edge cases (empty queries, special characters, multilingual content)

**Run all vector store tests:**
```bash
uv run pytest tests/test_vector_store.py -v
```

**Run specific test groups:**
```bash
# Test insertion operations
uv run pytest tests/test_vector_store.py -k "insert" -v

# Test search operations
uv run pytest tests/test_vector_store.py -k "search" -v

# Test deletion operations
uv run pytest tests/test_vector_store.py -k "delete" -v
```

**Example test output:**
```
tests/test_vector_store.py::TestVectorStore::test_initialization PASSED
tests/test_vector_store.py::TestVectorStore::test_insert_single_document PASSED
tests/test_vector_store.py::TestVectorStore::test_search_semantic_similarity PASSED
...
========================= 28 passed in 15.23s =========================
```

### Embedding Generation Tests (test_embeddings.py)

**18 test cases covering:**
- Model initialization
- Single text embedding generation
- Batch embedding generation
- Empty text handling
- Deterministic output verification
- Semantic similarity validation
- Unicode text support
- Large batch processing

**Run all embedding tests:**
```bash
uv run pytest tests/test_embeddings.py -v
```

**Key test scenarios:**
```bash
# Test basic embedding generation
uv run pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_generate_embedding_single_text -v

# Test batch processing
uv run pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_generate_embeddings_batch -v

# Test semantic similarity
uv run pytest tests/test_embeddings.py::TestEmbeddingGenerator::test_semantic_similarity -v
```

### Text Processing Tests (test_text_processing.py)

**30+ test cases covering:**
- Text cleaning and normalization
- URL removal
- Mention removal
- Whitespace normalization
- Document chunking
- Text truncation
- Keyword extraction
- Text field combination

**Run all text processing tests:**
```bash
uv run pytest tests/test_text_processing.py -v
```

**Run specific utility tests:**
```bash
# Test text cleaning
uv run pytest tests/test_text_processing.py::TestCleanText -v

# Test chunking
uv run pytest tests/test_text_processing.py::TestChunkText -v

# Test keyword extraction
uv run pytest tests/test_text_processing.py::TestExtractKeywords -v
```

### Run All Milestone 1E Tests

```bash
# Run all three test modules
uv run pytest tests/test_vector_store.py tests/test_embeddings.py tests/test_text_processing.py -v

# With coverage report
uv run pytest tests/test_vector_store.py tests/test_embeddings.py tests/test_text_processing.py --cov=src.db.vector_store --cov=src.models.embeddings --cov=src.utils.text_processing --cov-report=term-missing
```

## Docker Testing

### Run Tests in Container

```bash
# Start services
docker compose up -d

# Run all tests
docker compose exec api pytest -v

# Run Milestone 1E tests
docker compose exec api pytest tests/test_vector_store.py tests/test_embeddings.py tests/test_text_processing.py -v

# Run with coverage
docker compose exec api pytest --cov=src --cov-report=html

# Copy coverage report from container
docker compose cp api:/app/htmlcov ./htmlcov
```

### Test Data Setup

```bash
# Generate mock data for testing
docker compose exec api python scripts/generate_mock_data.py --scenarios all

# Verify ChromaDB is accessible
docker compose exec api python -c "from src.db.vector_store import get_vector_store; print(get_vector_store().check_connection())"

# Clear test data
docker compose exec postgres psql -U coordination_user -d coordination -c "TRUNCATE TABLE messages, sources CASCADE;"
```

## Coverage Reports

### Generate Coverage Reports

```bash
# HTML report (most detailed)
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Terminal report
uv run pytest --cov=src --cov-report=term-missing

# XML report (for CI tools)
uv run pytest --cov=src --cov-report=xml

# Multiple formats
uv run pytest --cov=src --cov-report=html --cov-report=term-missing --cov-report=xml
```

### Coverage Thresholds

The project targets the following coverage levels:
- **Overall**: 85%+
- **Vector store**: 95%+
- **Embeddings**: 90%+
- **Text processing**: 85%+

```bash
# Fail if coverage below threshold
uv run pytest --cov=src --cov-fail-under=85
```

## Writing Tests

### Test Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
@pytest.fixture
def vector_store():
    """Create a clean VectorStore instance."""
    store = VectorStore()
    store.clear_collection()
    yield store
    store.clear_collection()

@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"id": 1, "content": "Test message", "metadata": {}},
        # ...
    ]
```

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

```python
class TestVectorStore:
    """Test suite for VectorStore class."""

    def test_insert_single_document(self, vector_store):
        """Test inserting a single document."""
        # Test implementation
        pass
```

### Assertion Best Practices

```python
# Good: Specific assertions
assert len(results) == 5
assert all(score >= 0.7 for _, _, score, _ in results)

# Bad: Vague assertions
assert results
assert results is not None
```

### Test Organization

```python
def test_search_semantic_similarity(self, vector_store, sample_messages):
    """Test semantic similarity search."""
    # Arrange: Set up test data
    message_ids = [msg["id"] for msg in sample_messages]
    contents = [msg["content"] for msg in sample_messages]
    vector_store.insert_batch(message_ids, contents)

    # Act: Perform the operation
    results = vector_store.search("OAuth authentication", limit=3)

    # Assert: Verify the results
    assert len(results) > 0
    assert len(results) <= 3
    assert "oauth" in results[0][1].lower()
```

## Continuous Integration

### GitHub Actions

The project uses GitHub Actions for CI. Tests run automatically on:
- Push to main branch
- Pull requests
- Scheduled nightly builds

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    uv run pytest --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

Set up pre-commit hooks to run tests before commits:

```bash
# .git/hooks/pre-commit
#!/bin/bash
uv run pytest tests/test_embeddings.py tests/test_text_processing.py -x
```

## Troubleshooting

### Common Issues

#### Import Errors

```bash
# Problem: Module not found
ModuleNotFoundError: No module named 'src'

# Solution: Install package in editable mode
uv pip install -e .
```

#### ChromaDB Connection Issues

```bash
# Problem: ChromaDB connection fails
# Solution: Check if ChromaDB service is running
docker compose ps chromadb

# Restart ChromaDB
docker compose restart chromadb
```

#### Slow Tests

```bash
# Problem: Tests are running slowly
# Solution: Use pytest-xdist for parallel execution
uv pip install pytest-xdist
uv run pytest -n auto
```

#### Failed Tests in CI but Pass Locally

```bash
# Problem: Tests fail in CI but pass locally
# Solution: Run tests with same environment
docker compose exec api pytest -v
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)

---

For questions or issues with tests, please open an issue on GitHub or consult the main [README.md](./README.md).
