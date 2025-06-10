# Tests Directory

This directory contains all tests for the Qdrant RAG MCP Server, organized by category.

## Structure

- **unit/**: Unit tests for individual components
  - AST chunking, BM25 index, context tracking, enhanced ranking, etc.
  
- **integration/**: Integration tests that test multiple components
  - Documentation indexing, specialized embeddings, progressive context, etc.
  
- **performance/**: Performance and memory tests
  - Model memory usage, search performance, Apple Silicon optimization
  
- **debug/**: Debug and troubleshooting scripts (not pytest tests)
  - Model loading diagnostics, dimension mismatch debugging, thread safety tests
  
- **examples/**: Example test data and validation
  - `test_projects/` - Sample projects for integration testing
  - `validation_guides/` - Manual testing guides
  - Simple example scripts

## Running Tests

### Using the test runner script (recommended):
```bash
# Run all tests
./run_tests.sh all

# Run specific category
./run_tests.sh unit
./run_tests.sh integration
./run_tests.sh performance

# Run with coverage
./run_tests.sh coverage

# Run quick tests only
./run_tests.sh quick
```

### Using pytest directly:
```bash
# Run all tests
uv run pytest

# Run specific category
uv run pytest tests/unit/
uv run pytest tests/integration/

# Run specific test
uv run pytest tests/unit/test_ast_chunking.py

# Run with coverage
uv run pytest --cov=src tests/

# Run tests with specific markers
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "requires_qdrant"  # Only tests that need Qdrant
```

### Running debug scripts:
```bash
# Debug scripts are standalone Python scripts
uv run python tests/debug/debug_model_loading.py
uv run python tests/debug/test_dimension_mismatch.py
```

## Test Categories

### Unit Tests
Fast, isolated tests for individual components:
- `test_ast_chunking.py` - AST-based code chunking
- `test_bm25_index.py` - BM25 keyword search index
- `test_context_tracking.py` - Context window tracking
- `test_enhanced_ranking.py` - Search result ranking
- `test_logging.py` - Logging functionality
- `test_scoring_pipeline.py` - Scoring pipeline

### Integration Tests
Tests that involve multiple components:
- `test_specialized_embeddings*.py` - Specialized embedding models
- `test_documentation_*.py` - Documentation indexing
- `test_smart_reindex*.py` - Smart reindexing functionality
- `test_progressive_*.py` - Progressive context features

### Performance Tests
Tests focused on performance and resource usage:
- `test_model_memory*.py` - Model memory usage
- `test_search_memory.py` - Search memory usage
- `test_apple_silicon_http.py` - Apple Silicon optimizations

### Debug Scripts
Standalone scripts for debugging issues:
- `debug_model_loading.py` - Debug model loading issues
- `test_dimension_mismatch.py` - Debug dimension mismatches
- `test_thread_safety_*.py` - Thread safety debugging

## Recent Fixes

### v0.3.3.post4 - Dimension Mismatch Fix
Fixed issue where Python files would sometimes get 384D embeddings instead of 768D during batch reindexing. The fix includes:
1. Dimension-compatible fallback logic
2. Thread safety improvements in SpecializedEmbeddingManager

See `debug/` directory for the debugging scripts used to identify and fix this issue:
- `test_dimension_mismatch.py` - Initial reproduction
- `test_concurrent_dimension_issue.py` - Identified thread safety issue
- `test_thread_safety_fixed.py` - Verified the fix

## Writing Tests

New tests should:
1. Be placed in the appropriate category directory
2. Follow the naming convention `test_*.py`
3. Use pytest fixtures from `conftest.py`
4. Add appropriate markers (@pytest.mark.unit, @pytest.mark.slow, etc.)
5. Include docstrings explaining what is being tested

Example:
```python
import pytest

@pytest.mark.unit
def test_example_feature(sample_python_file):
    '''Test that example feature works correctly'''
    # Test implementation
    assert True
```