# Development Workflow Guide for Qdrant RAG MCP Server

This guide provides efficient search strategies and workflows for developers implementing new features or debugging the Qdrant RAG MCP Server.

## 🎯 Using RAG Search to Navigate the Codebase

### Quick Tips for Efficient Development

1. **Use specific search queries** instead of browsing files manually
2. **Search for function implementations** with patterns like `"function_name implementation"`
3. **Find related code** by searching for multiple keywords: `"file_hash metadata modified_at"`
4. **Use the Task tool** to break down complex implementations into searchable components

## 📋 Implementation Task Search Strategies

### Smart Reindex (v0.2.4) - File Hash Tracking

**Goal**: Implement incremental reindexing that only processes changed files.

**Key Search Queries**:
```bash
# Find existing reindex implementation
"reindex_directory function implementation"
"delete_file_chunks calculate_file_hash"
"file_hash metadata modified_at"

# Find where to add incremental logic
"detect_changes implementation comparison"
"scroll through all points file metadata"

# Understand collection clearing
"clear_project_collections delete collection"
```

**Files to Focus On**:
- `src/qdrant_mcp_context_aware.py` - Main reindex logic
- `src/utils/file_hash.py` - Hash calculation utilities

### Adaptive Search Intelligence (v0.2.5)

**Goal**: Implement query intent classification and dynamic search optimization.

**Key Search Queries**:
```bash
# Current search implementation
"search_code search_docs hybrid search"
"search_mode vector keyword hybrid"
"query processing search intent"

# Ranking and scoring
"enhanced_ranker ranking_signals"
"vector_score bm25_score combined"

# Configuration handling
"server_config.json search parameters"
```

**Files to Focus On**:
- `src/qdrant_mcp_context_aware.py` - Search methods
- `src/utils/enhanced_ranker.py` - Ranking logic
- `src/utils/hybrid_search.py` - Search mode handling

### Progressive Context Management

**Goal**: Implement multi-level context API for efficient token usage.

**Key Search Queries**:
```bash
# Context expansion features
"include_context context_chunks expand"
"get_file_chunks surrounding chunks"
"_expand_search_context implementation"

# Token optimization
"truncate_content max_length token"
"chunk_size chunk_overlap configuration"
```

**Files to Focus On**:
- Context expansion logic in search methods
- Chunk retrieval and assembly

### Query Enhancement

**Goal**: Implement query reformulation and synonym expansion.

**Key Search Queries**:
```bash
# Current query handling
"query reformulation synonym"
"embedding_model encode query"
"search preprocessing normalization"

# Language processing
"ast_chunker language detection"
"code vocabulary mapping"
```

## 🔍 Common Development Patterns

### Adding New File Type Support

1. **Search for file type detection**:
   ```
   "suffix in supported_extensions"
   "file_type determination extension"
   ```

2. **Find indexer patterns**:
   ```
   "CodeIndexer ConfigIndexer DocumentationIndexer"
   "index_file chunk metadata"
   ```

3. **Update collection routing**:
   ```
   "get_collection_name file_type"
   "ensure_collection create"
   ```

### Modifying Search Behavior

1. **Understand current search flow**:
   ```
   "search tool implementation mcp"
   "filter_conditions must match"
   ```

2. **Find ranking logic**:
   ```
   "ranking_signals score calculation"
   "combine_search_results deduplication"
   ```

### Adding New Metadata Fields

1. **Find metadata storage patterns**:
   ```
   "payload metadata chunk storage"
   "PointStruct vector payload"
   ```

2. **Update all indexers**:
   ```
   "index_code payload fields"
   "index_config metadata extraction"
   "index_documentation chunk metadata"
   ```

## 🚀 Testing Your Changes

### Quick Test Commands

After implementing changes, test with:

```bash
# Test health check
"Check system health"

# Test reindexing
"Detect changes in current directory"
"Smart reindex this directory"

# Test search
"Search for [your test query]"
"Search code for [function name]"
```

### Debugging Search Issues

1. **Check indexed content**:
   ```
   "Get file chunks for [file path]"
   ```

2. **Verify collections**:
   ```
   "Get current project context"
   ```

3. **Review logs**:
   ```bash
   ./scripts/qdrant-logs -f --operation [operation_name]
   ```

## 📝 Code Organization Tips

### Key Files Reference

- **Main Server**: `src/qdrant_mcp_context_aware.py`
  - MCP tool definitions
  - Core indexing/search logic
  - Collection management

- **Indexers**: `src/indexers/`
  - `code_indexer.py` - AST-based code chunking
  - `config_indexer.py` - Configuration file parsing
  - `documentation_indexer.py` - Markdown/docs parsing

- **Utilities**: `src/utils/`
  - `ast_chunker.py` - Language-specific AST parsing
  - `enhanced_ranker.py` - Multi-signal ranking
  - `hybrid_search.py` - BM25 + vector search
  - `file_hash.py` - File change detection

### Common Gotchas

1. **MCP Server Restart Required**: Changes to `qdrant_mcp_context_aware.py` require restarting Claude Code
2. **Collection Names**: Always use `get_collection_name()` for project isolation
3. **Error Handling**: Use structured error returns with `error_code` for better debugging
4. **Logging**: Use `get_logger()` for project-aware logging

## 🔗 Related Documentation

- [Advanced RAG Implementation Roadmap](technical/advanced-rag-implementation-roadmap.md)
- [RAG Usage Examples](rag-usage-examples.md)
- [Technical Guides](technical/)

## 💡 Pro Tips

1. **Use the Agent tool** for complex searches across multiple files
2. **Create todos** for multi-step implementations to track progress
3. **Search for test files** to understand expected behavior
4. **Check git history** for similar past implementations

Remember: The RAG server is your assistant for understanding its own codebase. Use it liberally!