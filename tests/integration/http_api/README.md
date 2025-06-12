# HTTP API Integration Tests

This directory contains integration tests for the HTTP API endpoints.

## Test Files

- `test_http_api.sh` - General HTTP API endpoint tests including:
  - Health check endpoint
  - Index operations (index_code, index_config, index_documentation, index_directory)
  - Search operations (search, search_code, search_config, search_docs)
  - Context operations (get_context)
  - Reindex operations

## Running Tests

From the project root:

```bash
# Run all HTTP API tests
./tests/integration/http_api/test_http_api.sh

# Make sure the HTTP server is running first
python src/http_server.py
```

## Prerequisites

- HTTP server running on port 8081
- Qdrant running on port 6333
- jq installed for JSON parsing