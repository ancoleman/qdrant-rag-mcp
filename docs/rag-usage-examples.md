# Qdrant RAG Server - Practical Usage Examples with Claude Code

This guide provides real-world examples of how to effectively use the Qdrant RAG server when working with Claude Code to enhance your development workflow.

## Table of Contents
1. [Setting Correct Working Directory](#0-setting-correct-working-directory-v022)
2. [Finding Implementation Details](#1-finding-implementation-details)
3. [Understanding Configuration](#2-understanding-configuration)
4. [Debugging Issues](#3-debugging-issues)
5. [Finding Examples](#4-finding-examples)
6. [Architecture Questions](#5-architecture-questions)
7. [Finding Related Code](#6-finding-related-code)
8. [Configuration Updates](#7-configuration-updates)
9. [Cross-Project Search](#8-cross-project-search-global)
10. [Understanding Scripts](#9-understanding-scripts)
11. [Using Hybrid Search Effectively](#10-using-hybrid-search-effectively-v014)
12. [Finding Test Examples](#11-finding-test-examples)
13. [Checking for File Changes](#12-checking-for-file-changes-v024)
14. [Indexing Custom File Types](#13-indexing-custom-file-types)
15. [Pro Tips](#pro-tips)

## 0. Setting Correct Working Directory (v0.2.2+)

**Scenario**: You're working in `/project-a` but health check shows the wrong project context.

### Natural Language Approach (No Configuration Required!)

```
You: "Get the current working directory with pwd, export MCP_CLIENT_CWD to that value, then run a health check"

Claude: I'll set the correct working directory for the MCP server and verify the project context.
[Runs: pwd]
[Output: /Users/you/projects/project-a]
[Runs: export MCP_CLIENT_CWD=/Users/you/projects/project-a]
[Uses: qdrant-rag:health_check]

✓ Health check shows:
- Project: project_a
- Path: /Users/you/projects/project-a
- The MCP server now correctly knows your working directory!
```

### For Indexing with Correct Context

```
You: "First run pwd and export MCP_CLIENT_CWD=$(pwd), then reindex this directory"

Claude: I'll ensure the MCP server knows the correct directory before indexing.
[Runs: export MCP_CLIENT_CWD=$(pwd)]
[Uses: qdrant-rag:reindex_directory with correct project context]
```

**Why this works**: The MCP server runs in its own process and doesn't automatically know where you're working. By setting `MCP_CLIENT_CWD`, you tell it your actual location.

**Pro tip**: Start every Claude Code session with the working directory setup to ensure correct project detection throughout your session.

## 1. Finding Implementation Details

**Scenario**: You need to understand how a specific feature is implemented.

```
You: "How does the embedding model get initialized in this project?"

Claude: Let me search for embedding model initialization...
[Uses: qdrant-rag:search "embedding model initialization"]
[Finds: EmbeddingsManager class in src/utils/embeddings.py]

The embedding model is initialized in the EmbeddingsManager class at 
src/utils/embeddings.py:21. It uses lazy initialization with auto-device 
detection (CPU/CUDA/MPS) and defaults to 'all-MiniLM-L6-v2' model...
```

**What's happening**: Claude uses hybrid search (v0.1.4+) combining semantic understanding with keyword matching. This finds relevant code even if you don't know exact names, while also prioritizing exact matches like "EmbeddingsManager".

## 2. Understanding Configuration

**Scenario**: You need to know what configuration options are available.

```
You: "What environment variables does this project use?"

Claude: Let me search for environment variable usage...
[Uses: qdrant-rag:search "environment variables env getenv"]
[Finds: config.py, .env.example, various script files]

The project uses these environment variables:
- EMBEDDING_MODEL: Choose embedding model (default: all-MiniLM-L12-v2)
- QDRANT_HOST/QDRANT_PORT: Qdrant connection
- SENTENCE_TRANSFORMERS_HOME: Model cache directory
- HTTP_SERVER_PORT: HTTP test server port (default: 8081)
[Shows actual values from config.py]
```

**What's happening**: The RAG server indexes configuration files, making it easy to discover all configurable options.

## 3. Debugging Issues

**Scenario**: You encounter an error and need to trace its origin.

```
You: "I'm getting a 'chunk_id' attribute error. Where is this used?"

Claude: Let me search for chunk_id usage...
[Uses: qdrant-rag:search_code "chunk_id attribute"]
[Finds: qdrant_mcp_context_aware.py where chunk_id is generated]

I found the issue. The chunk_id is generated in src/qdrant_mcp_context_aware.py:239 
using hashlib. The error suggests you might be running an older cached version 
of the MCP server. Try restarting the MCP connection...
```

**What's happening**: Claude can quickly locate error-related code and provide context about potential causes.

## 4. Finding Examples

**Scenario**: You need to understand how to use a specific component.

```
You: "Show me how to use the code indexer"

Claude: Let me find code indexer usage examples...
[Uses: qdrant-rag:search_code "code_indexer index_file"]
[Finds: actual usage in qdrant_mcp_context_aware.py]

Here's how the code indexer is used:

```python
from indexers.code_indexer import CodeIndexer

# Initialize
code_indexer = get_code_indexer()

# Index a file
chunks = code_indexer.index_file(str(file_path))

# Process chunks
for chunk in chunks:
    # chunk.content - the code text
    # chunk.line_start/line_end - line numbers
    # chunk.chunk_index - position in file
```
```

**What's happening**: The search finds actual usage patterns in your codebase, providing real examples.

## 5. Architecture Questions

**Scenario**: You need to understand how a system component works.

```
You: "How does the context-aware project detection work?"

Claude: [Uses: qdrant-rag:search "project detection get_current_project"]
[Finds: implementation details and marker files]

The context-aware detection works by:
1. Starting from the current working directory
2. Looking for project markers in parent directories:
   - .git directory
   - package.json, pyproject.toml, Cargo.toml
   - go.mod, pom.xml, etc.
3. Creating project-specific collections like 'project_myapp_code'
4. Scoping searches to the current project by default

Implementation is in get_current_project() function.
```

**What's happening**: Claude can explain architectural decisions by finding and analyzing relevant code.

## 6. Finding Related Code

**Scenario**: You need to find all files related to a specific feature.

```
You: "What files handle MCP protocol communication?"

Claude: [Uses: qdrant-rag:search "mcp protocol @mcp.tool FastMCP"]
[Finds: all MCP-related files]

The MCP protocol is handled by:
- src/qdrant_mcp_context_aware.py - Main MCP server implementation
- config/mcp_manifest.json - API method definitions
- run_mcp.sh - Server startup script
- src/qdrant_mcp_debug.py - Debug version

Key decorators:
- @mcp.tool() - Defines MCP-callable functions
- Functions like index_code(), search(), get_context()
```

**What's happening**: The semantic search understands related concepts and finds all relevant files.

## 7. Configuration Updates

**Scenario**: You need to add a new configuration option.

```
You: "I need to add a new config option for chunk size"

Claude: Let me find where chunk configuration is defined...
[Uses: qdrant-rag:search_config "chunk_size chunk_overlap"]
[Finds: config files and usage]

Chunk size is configured in multiple places:
1. config/server_config.json - Default values:
   ```json
   "chunking": {
     "chunk_size": 1500,
     "chunk_overlap": 200
   }
   ```
2. src/config.py:45 - Where it's loaded
3. src/indexers/code_indexer.py:64 - Where it's used

To add a new option, update all three locations.
```

**What's happening**: The config search specifically targets configuration files and their usage.

## 8. Cross-Project Search (Global)

**Scenario**: You need to find implementations across multiple projects.

```
You: "Search all my projects for OAuth implementation"

Claude: [Uses: qdrant-rag:search "OAuth authentication" cross_project=true]
[Finds: OAuth implementations across different indexed projects]

Found OAuth implementations across your projects:
- project_webapp: src/auth/oauth_handler.py
  - Implements OAuth2 flow with Google/GitHub
- project_api: lib/oauth2/provider.js
  - JWT-based OAuth provider
- project_mobile: auth/OAuthManager.swift
  - iOS OAuth integration

Would you like me to examine any specific implementation?
```

**What's happening**: With cross_project=true, the search spans all indexed projects, not just the current one.

## 9. Understanding Scripts

**Scenario**: You need to understand what a script does before running it.

```
You: "What does the setup.sh script do?"

Claude: [Uses: qdrant-rag:search "setup.sh script"]
[Finds: the script and related documentation]

The setup.sh script (scripts/setup.sh) performs initial setup:
1. Creates Python virtual environment
2. Installs dependencies via uv
3. Downloads embedding models
4. Creates necessary directories (data/, logs/)
5. Copies .env.example to .env if needed
6. Verifies Qdrant connection

Safe to run multiple times - it checks for existing setup.
```

**What's happening**: Claude reads and summarizes script contents, explaining what will happen.

## 10. Using Hybrid Search Effectively (v0.1.4+)

**Scenario**: You want to leverage the new hybrid search for better results.

```
You: "Find the BM25Manager class"

Claude: Let me search for that class definition...
[Uses: qdrant-rag:search "BM25Manager class"]
[Hybrid search combines keyword match for "BM25Manager" with semantic understanding]

Found BM25Manager class at src/utils/hybrid_search.py:47. This class manages
BM25 keyword indices for collections. The hybrid search gives this high ranking
because:
- Keyword match: Exact match on "BM25Manager" 
- Semantic match: Understanding you want a class definition
```

**Comparing Search Modes**:

```
# Default (Hybrid) - Best for most searches
You: "Find calculate_sum function"
[Finds both exact matches AND semantically similar calculation functions]

# Keyword mode - When you need exact matches
You: "Search for 'def calculate_sum(' using keyword mode"
[Only finds exact text matches]

# Vector mode - When you need conceptual matches
You: "Find all calculation-related functions using vector mode"  
[Finds semantically similar code regardless of naming]
```

**What's happening**: Hybrid search automatically balances exact keyword matching (great for function/class names) with semantic understanding (great for concepts), giving you the best of both approaches.

## 11. Finding Test Examples

**Scenario**: You need to know how to test specific functionality.

```
You: "How do I test the HTTP API?"

Claude: [Uses: qdrant-rag:search "test http api curl"]
[Finds: test scripts and documentation]

You can test the HTTP API using tests/integration/http_api/test_http_api.sh which includes:

```bash
# Test search endpoint
curl -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your search term", "n_results": 5}'

# Test indexing
curl -X POST http://localhost:8081/index_directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project"}'

# Test health check
curl http://localhost:8081/health
```

Full test suite available in the script.
```

**What's happening**: The search finds testing documentation and examples.

## 12. Checking for File Changes (v0.2.4+)

**Scenario**: You want to see what files have changed since the last indexing without running a full reindex.

```
You: "Check what files have changed in this project"

Claude: I'll check for changes since the last index...
[Uses: qdrant-rag:detect_changes]

Here's what changed:
- Added: 2 files
  - src/new_feature.py
  - tests/test_new_feature.py
- Modified: 3 files
  - src/main.py (updated 2 hours ago)
  - config/settings.json (updated 1 hour ago)
  - README.md (updated 30 minutes ago)
- Deleted: 1 file
  - src/deprecated_module.py
- Unchanged: 156 files

Would you like me to reindex these changes?
```

**Use Cases for detect_changes**:
- **Pre-reindex check**: See what would be updated before running reindex
- **CI/CD integration**: Automate index updates based on changes
- **Monitoring**: Track codebase evolution over time
- **Debugging**: Verify the indexer is tracking files correctly

## 13. Indexing Custom File Types

**Scenario**: You need to index file types not included in the default patterns (like .sql, .graphql, .proto files).

```
You: "Index this directory including SQL and GraphQL files"

Claude: I'll index the directory with custom patterns for SQL and GraphQL files...
[Uses: qdrant-rag:index_directory with patterns ["*.sql", "*.graphql"]]

Successfully indexed:
- 156 Python files (.py)
- 23 SQL files (.sql) 
- 8 GraphQL schema files (.graphql)
- 45 JSON config files (.json)
Total: 232 files indexed
```

**Common Custom Pattern Examples**:
```
# Database files
"Index with patterns *.sql *.ddl *.dml"

# API definitions
"Index patterns *.proto *.graphql *.openapi.yaml"

# Infrastructure as Code
"Index patterns *.tf *.tfvars for Terraform"

# Specific project types
"Index only Rust: *.rs *.toml Cargo.lock"
"Index only Go: *.go go.mod go.sum"
"Index Flutter: *.dart pubspec.yaml"
```

**Note**: When you specify custom patterns, they're added to the default patterns, not replacing them. To index ONLY specific types, be explicit about it.

## Pro Tips

### 1. **Be Specific with Queries**
- ❌ Poor: "error"
- ✅ Good: "error handling in indexers"
- ✅ Better: "error handling CodeIndexer try except"

### 2. **Use the Right Search Type**
- `search` - General semantic search across all content
- `search_code` - When you need actual implementation code
- `search_config` - When looking for configuration values

### 3. **Leverage Search Modes (v0.1.4+)**
The server now supports three search modes for better precision:

**Hybrid Mode (Default)** - Best of both worlds:
```
# Automatically combines keyword and semantic search
"Find the authenticate_user function"
"Search for database connection handling"
```

**Keyword Mode** - For exact matches:
```
# Use when you know exact function/variable names
"Search for 'def calculate_sum' using keyword mode"
"Find BM25Manager class with keyword search"
```

**Vector Mode** - For concepts:
```
# Use when searching for patterns or ideas
"Find error handling patterns using vector mode"
"Search for authentication flows with semantic search"
```

### 4. **Include Multiple Keywords**
- ❌ Poor: "model"
- ✅ Good: "embedding model initialization device"
- ✅ Better: "EmbeddingsManager model initialization MPS CUDA"

### 5. **Use After Major Changes**
```bash
# After significant code changes
qdrant-rag:index_directory .
```

### 6. **Leverage Project Context**
```python
# Searches current project by default
qdrant-rag:search "database connection"

# Search across all projects
qdrant-rag:search "database connection" cross_project=true
```

### 7. **Common Search Patterns**

**Finding where something is defined:**
```
"class MyClass definition"
"function process_data defined"
"where is CONFIG_FILE set"
```

**Finding usage examples:**
```
"how to use DataProcessor"
"examples calling api_endpoint"
"usage of decorator @cache"
```

**Finding configuration:**
```
"port configuration"
"timeout settings"
"environment variables for production"
```

**Debugging:**
```
"AttributeError chunk_id"
"ImportError module not found"
"where is logging configured"
```

### 8. **Understanding Code Flow**
```
You: "Trace the flow from HTTP request to database query"

Claude: [Uses multiple searches to trace the path]
[Builds a complete picture of the request flow]
```

## Benefits of Using RAG with Claude Code

1. **Instant Context**: Claude understands your entire codebase structure
2. **Hybrid Search (v0.1.4+)**: Combines keyword matching with semantic understanding for +30% better precision
3. **Cross-File Intelligence**: Understands relationships between files
4. **Up-to-Date Information**: Always searches current code, not outdated documentation
5. **Project Isolation**: Searches are scoped to your current project by default
6. **Smart Ranking**: Balances exact matches (function names) with conceptual similarity

## When RAG is Most Valuable

- **Large Codebases**: When manual searching becomes impractical
- **Unfamiliar Projects**: When onboarding to a new codebase
- **Complex Architectures**: When tracing through multiple layers
- **Debugging**: When you need to find all usages of something
- **Refactoring**: When you need to understand impact of changes
- **Documentation**: When written docs are outdated or missing

The RAG server essentially gives Claude Code a semantic understanding of your entire codebase, making it much more effective at finding relevant code and understanding your project's architecture.