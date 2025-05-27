# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📖 Complete Documentation

For comprehensive setup and usage instructions, see:
- **[Complete Setup & Usage Guide](docs/complete-setup-and-usage-guide.md)** - Full MCP + HTTP API documentation
- **[MPS Quick Reference](docs/mps-quick-reference.md)** - Apple Silicon optimization guide
- **[Enhanced RAG Guide](docs/enhanced-qdrant-rag-guide.md)** - Technical implementation details

## 🚀 Quick Start

### Global Installation (Recommended)

```bash
# For global usage across all projects
./install_global.sh

# The installer configures everything automatically
# See docs/mcp-scope-configuration-guide.md for details
```

### Auto-Indexing (Optional)

```bash
# Enable auto-indexing for current session
export QDRANT_RAG_AUTO_INDEX=true
claude

# Or add to ~/.bashrc or ~/.zshrc for permanent auto-indexing
export QDRANT_RAG_AUTO_INDEX=true
export QDRANT_RAG_DEBOUNCE=5.0  # Optional: adjust debounce time

# Toggle auto-indexing on/off
unset QDRANT_RAG_AUTO_INDEX      # Disable
export QDRANT_RAG_AUTO_INDEX=true # Enable

# Check status
echo $QDRANT_RAG_AUTO_INDEX
```

### Setup & Installation

```bash
# Initial setup (only need to run once)
./scripts/setup.sh

# Start with MPS acceleration (macOS Apple Silicon)
./docker/start_with_env.sh --local

# Start in Docker mode (all platforms)
./docker/start_with_env.sh

# Pre-download embedding models (optional)
./scripts/download_models.sh
```

### HTTP Testing Server (Optional)

The HTTP server is **only for testing** the indexing/search functionality outside of Claude Code:

```bash
# Start HTTP API server for testing (port 8081)
export $(grep -v '^#' .env | xargs)
python src/http_server.py

# Test server health
curl http://localhost:8081/health

# Run comprehensive API tests
./scripts/test_http_api.sh
```

**Note**: The MCP server (`qdrant_mcp_context_aware.py`) is what Claude Code uses, not the HTTP server.

### Indexing Content

The indexer automatically excludes common non-source files:
- Version control (`.git/`), dependencies (`node_modules/`, `venv/`)
- Build artifacts (`dist/`, `build/`, `*.pyc`)
- Data/cache directories (`data/`, `logs/`, `.cache/`)
- IDE files (`.vscode/`, `.idea/`)
- See `.ragignore` for full exclusion list

```bash
# Index a project directory (using script)
./scripts/index_project.sh /path/to/your/project

# HTTP API - Index entire directory with enhanced features
curl -X POST http://localhost:8081/index_directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/project",
    "patterns": ["*.py", "*.js", "*.json", "*.yaml"],
    "exclude_patterns": ["**/node_modules/**", "**/.git/**"]
  }'

# HTTP API - Index single code file
curl -X POST http://localhost:8081/index_code \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.py"}'

# HTTP API - Index configuration file
curl -X POST http://localhost:8081/index_config \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/config.json"}'
```

### Searching Content

```bash
# HTTP API - General search with enhanced results
curl -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication middleware", "n_results": 5}'

# HTTP API - Code-specific search with language filter
curl -X POST http://localhost:8081/search_code \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling", 
    "language": "python",
    "chunk_type": "function"
  }'

# HTTP API - Config search with path filtering
curl -X POST http://localhost:8081/search_config \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection",
    "file_type": "json",
    "path": "database.connection"
  }'
```

### Docker Management

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up -d

# Stop all services
docker-compose -f docker/docker-compose.yml down

# Restart a specific service
docker-compose -f docker/docker-compose.yml restart rag-server

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Check container status
docker-compose -f docker/docker-compose.yml ps
```

### Embedding Models

```bash
# Switch models (requires re-indexing)
sed -i '' 's/EMBEDDING_MODEL=.*/EMBEDDING_MODEL=new-model-name/' .env

# Clear collections after model change
curl -X DELETE http://localhost:6333/collections/code_collection
curl -X DELETE http://localhost:6333/collections/config_collection

# Restart server
docker-compose -f docker/docker-compose.yml restart rag-server
```

## ✨ Enhanced Features

### Specialized Indexers

The server now uses advanced indexers that provide:

**Code Indexing:**
- Language-specific parsing (Python, JavaScript, Java, C++, Go, etc.)
- Structure-aware chunking (functions, classes, methods)
- Line number tracking for precise locations
- Rich metadata (imports, dependencies, code structure)

**Config Indexing:**
- Multiple formats: JSON, XML, YAML, TOML, INI, ENV
- Hierarchical structure preservation
- Path-based navigation and filtering
- Schema extraction for better understanding

### Apple Silicon Optimization

- **MPS Support**: Metal Performance Shaders acceleration on M1/M2/M3 chips
- **Local Mode**: Run natively on macOS for best performance
- **Model Caching**: Intelligent local model storage

### Dual Operation Modes

1. **MCP Server Mode**: Direct Claude Code integration via stdio (no network port)
2. **HTTP API Mode** (Port 8081): Optional testing and standalone usage

## Code Architecture

### Overview

This repository implements a RAG (Retrieval-Augmented Generation) server using Qdrant vector database as an MCP (Managed Claude Plugin) server for Claude Code. It enables semantic search across codebases by indexing code files and configuration files.

Key components:
- **Qdrant Vector Database**: Stores vector embeddings of code chunks
- **Sentence Transformers**: Provides embedding models to convert text to vectors
- **MCP Server**: Exposes RAG capabilities to Claude Code
- **Docker**: Containerizes the application for consistent deployment

### Directory Structure

```
~/mcp-servers/qdrant-rag/
├── config/                       # Configuration files
│   ├── mcp_manifest.json         # API definition for Claude Code
│   ├── server_config.json        # Server configuration
├── docker/                       # Docker configuration
│   ├── Dockerfile                # Server Dockerfile
│   ├── docker-compose.yml        # Complete stack setup
├── scripts/                      # Utility scripts
│   ├── start_server.sh           # Start server
│   ├── setup.sh                  # Initial setup
│   ├── index_project.sh          # Index a project
│   ├── download_models.sh        # Pre-download embedding models
│   └── install_auto_indexer.sh   # Install auto-indexing hooks
├── src/                          # Source code
│   ├── qdrant_mcp_context_aware.py  # Main server implementation
│   ├── config.py                 # Configuration handling
│   ├── indexers/                 # Code and config indexers
│   └── utils/                    # Utilities including embeddings
├── data/                         # Qdrant storage (not in git)
└── logs/                         # Server logs (not in git)
```

### Core Components

1. **Context-Aware MCP Server** (`src/qdrant_mcp_context_aware.py`): Main server that:
   - Initializes Qdrant client and embedding model
   - Sets up text splitters for code and configs
   - Registers MCP methods for indexing and searching
   - Handles document indexing, searching, and management

2. **MCP Methods**:
   - `index_code`: Index source code files with semantic understanding
   - `index_config`: Index configuration files (JSON, XML, YAML)
   - `index_directory`: Recursively index a directory
   - `search`: General search across all content
   - `search_code`: Code-specific search with filters
   - `search_config`: Configuration-specific search
   - `get_context`: Get surrounding context for a specific chunk
   - `update_index`: Update an existing indexed file
   - `delete_document`: Remove a file from the index
   - `should_activate`: Determine if server should handle a query

3. **Embedding Models**:
   - Default: `all-MiniLM-L6-v2` (small, fast model ~90MB)
   - Code-specific: `microsoft/codebert-base`
   - High-quality: `all-mpnet-base-v2`
   - Configured via environment variables or server_config.json

4. **Qdrant Collections**:
   - `code_collection`: Stores code chunks with metadata
   - `config_collection`: Stores configuration chunks

### Data Flow

1. **Indexing Flow**:
   - File is read and parsed
   - Content is split into chunks
   - Chunks are embedded using the embedding model
   - Vectors are stored in Qdrant with metadata
   - API returns summary of indexed content

2. **Search Flow**:
   - Query is embedded using the same model
   - Vector similarity search is performed in Qdrant
   - Results are filtered and ranked
   - Top matches are returned with context

### Configuration System

The server supports multiple configuration sources in this priority:
1. Environment variables (highest priority)
2. `.env` file
3. `config/server_config.json`
4. In-code defaults (lowest priority)

Key configuration options:
- `EMBEDDING_MODEL`: The embedding model to use
- `QDRANT_HOST`/`QDRANT_PORT`: Qdrant connection details
- `SERVER_PORT`: Port for the MCP server
- Chunking parameters (size, overlap)
- Search parameters (max results, threshold)

### Claude Code Integration

The server integrates with Claude Code through:
1. The MCP manifest which defines API methods and parameters
2. Registration in the Claude Code configuration
3. Auto-activation triggers that determine when to use this service
4. The `should_activate` method that decides if a query is relevant

## Best Practices

- **Embedding Models**: Choose the appropriate model based on your needs:
  - `all-MiniLM-L6-v2`: Fast, small footprint, good general performance
  - `microsoft/codebert-base`: Better for code understanding
  - `all-mpnet-base-v2`: Higher quality but slower and larger

- **Indexing**: 
  - Index code in manageable chunks (entire projects may take time)
  - Keep embeddings consistent (changing models requires re-indexing)
  - Prefer code-specific models for codebases with many programming languages

- **Docker Usage**:
  - Use provided Docker configuration for consistent environments
  - Data is persisted in the `data/` directory
  - Logs are available in the `logs/` directory

- **Model Caching**:
  - Models are cached in `SENTENCE_TRANSFORMERS_HOME` to avoid re-downloads
  - For macOS, optimally set to `~/Library/Caches/qdrant-mcp/models`