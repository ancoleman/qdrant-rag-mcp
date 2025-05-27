# Hybrid Search Implementation

This document describes the hybrid search feature implemented in v0.1.4 of the Qdrant RAG MCP Server.

## Overview

Hybrid search combines traditional keyword-based search (BM25) with semantic vector search to provide improved retrieval precision. According to our roadmap, this basic implementation delivers:

- **+30% precision improvement** over pure vector search
- Better handling of exact keyword matches
- More robust retrieval when semantic similarity alone isn't sufficient

## Architecture

### Components

1. **BM25Manager** (`src/utils/hybrid_search.py`)
   - Manages BM25 indices for each collection
   - Uses Langchain's BM25Retriever with rank_bm25 backend
   - Provides collection-specific keyword search

2. **HybridSearcher** (`src/utils/hybrid_search.py`)
   - Implements Reciprocal Rank Fusion (RRF) 
   - Supports multiple fusion strategies
   - Singleton pattern for efficient resource usage

3. **Search Integration** (`src/qdrant_mcp_context_aware.py`)
   - Extended search functions with `search_mode` parameter
   - Three modes: "vector", "keyword", "hybrid" (default)
   - Automatic BM25 index updates during document indexing

## Implementation Details

### BM25 Indexing

When documents are indexed:
1. Documents are stored in Qdrant with vector embeddings (existing behavior)
2. BM25 index is updated with all documents in the collection
3. Each document has a unique ID: `{file_path}_{chunk_index}`

### Search Modes

1. **Vector Mode**: Pure semantic search using embeddings (existing behavior)
2. **Keyword Mode**: Pure BM25 keyword search
3. **Hybrid Mode**: Combines both using Reciprocal Rank Fusion

### Reciprocal Rank Fusion (RRF)

The RRF algorithm combines rankings from different search methods:

```python
RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
```

Where:
- `d` is a document
- `weight_i` is the weight for search method i
- `rank_i(d)` is the rank of document d in method i
- `k` is a constant (default: 60)

Default weights:
- Vector search: 0.7
- BM25 search: 0.3

## API Changes

### Search Function

```python
@mcp.tool()
def search(
    query: str, 
    n_results: int = 5, 
    cross_project: bool = False,
    search_mode: str = "hybrid"  # NEW parameter
) -> Dict[str, Any]:
```

### Response Format

Hybrid search results include additional scoring information:

```json
{
  "results": [
    {
      "score": 0.85,           // Combined score
      "vector_score": 0.92,    // Semantic similarity score
      "bm25_score": 0.78,      // Keyword relevance score
      "search_mode": "hybrid", // Mode used
      // ... other fields
    }
  ],
  "search_mode": "hybrid"      // Search mode used
}
```

## Dependencies

Added in v0.1.4:
- `langchain-community>=0.3.24` - For BM25Retriever
- `rank-bm25>=0.2.2` - BM25 algorithm implementation

## Performance Considerations

1. **Index Updates**: Currently rebuilds entire BM25 index on updates (O(n))
   - Future optimization: Incremental updates

2. **Memory Usage**: BM25 indices are kept in memory
   - Scales linearly with document count
   - Future optimization: Persistent storage

3. **Search Latency**: Minimal overhead
   - Parallel execution of vector and BM25 search
   - RRF fusion is O(n log n) for n results

## Usage Examples

### Default Hybrid Search
```python
# In Claude Code
results = search("authentication middleware")
```

### Vector-Only Search
```python
# For semantic similarity only
results = search("authentication middleware", search_mode="vector")
```

### Keyword-Only Search
```python
# For exact matches
results = search("def authenticate_user", search_mode="keyword")
```

## Future Enhancements

As per the roadmap, next improvements could include:
1. **Advanced Hybrid Search** (Phase 2.1)
   - Dependency graph integration
   - Query-adaptive weighting
   - Multi-signal search

2. **Query Enhancement** (Phase 3.1)
   - Query reformulation
   - Synonym expansion
   - Code vocabulary mapping

## Testing

The hybrid search functionality can be tested using:
1. The MCP tools in Claude Code
2. The HTTP API (if running http_server.py)
3. Direct comparison of search modes for the same query

## Limitations

1. BM25 indices are not persisted (rebuilt on server restart)
2. No incremental index updates (full rebuild required)
3. Fixed fusion weights (not query-adaptive)
4. Language-specific tokenization not implemented

These limitations are acceptable for the v0.1.4 basic implementation and can be addressed in future releases as per the roadmap.