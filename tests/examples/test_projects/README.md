# Test Projects

This directory contains sample projects used for testing various features of Qdrant RAG MCP.

## Projects

### reindex-test/
A simple Python project with configuration files used for testing:
- Reindexing functionality
- File modification detection
- Smart incremental reindexing

Structure:
```
reindex-test/
├── config.json      # Sample configuration file
├── docs/
│   └── README.md    # Sample documentation
└── src/
    ├── hello.py     # Sample Python module
    └── goodbye.py   # Another Python module
```

## Usage

These test projects are used by integration tests in `tests/integration/`:
- `test_incremental_reindex.py`
- `test_smart_reindex.py`
- `test_smart_reindex_direct.py`

To add a new test project:
1. Create a directory with sample files
2. Include various file types (code, config, docs)
3. Document its purpose in this README
4. Use it in your integration tests

Example in tests:
```python
def test_reindex_functionality():
    test_project = Path(__file__).parent.parent / "examples/test_projects/reindex-test"
    # Use test_project for testing...
```