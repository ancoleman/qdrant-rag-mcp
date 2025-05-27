# Scripts Directory

This directory contains utility scripts for the Qdrant RAG MCP Server. Each script has a specific purpose - use the right tool for the job!

## 🚀 Essential Scripts

### `setup.sh`
**Purpose**: Initial project setup
- Creates required directories (data/, logs/)
- Sets up .env file from template
- Installs Python dependencies
- **When to use**: First time setting up the project
```bash
./scripts/setup.sh
```

### `index_project.sh`
**Purpose**: Quick project indexing
- Simple wrapper to index a directory
- Uses the HTTP API (requires server running on port 8081)
- **When to use**: When you need to manually index a project directory
```bash
# Index current directory
./scripts/index_project.sh .

# Index specific project
./scripts/index_project.sh /path/to/project
```

### `download_models.sh`
**Purpose**: Pre-download embedding models
- Downloads models before first use
- Useful for offline setups or production
- Prevents slow first-run experience
- **When to use**: Before deploying or when setting up offline environments
```bash
./scripts/download_models.sh
```

### `manage_models.sh`
**Purpose**: Unified model management
- **When to use**: Managing embedding models, troubleshooting model issues
```bash
# List downloaded models
./scripts/manage_models.sh list

# Debug cache issues
./scripts/manage_models.sh debug

# Set default model in .env
./scripts/manage_models.sh set

# Test model loading
./scripts/manage_models.sh test
```

## 🔧 Testing & Debugging

### `qdrant-logs`
**Purpose**: View and search project-aware logs
- Search and filter JSON structured logs
- Follow logs in real-time
- Export logs for analysis
- **When to use**: Debugging issues, monitoring operations, performance analysis
```bash
# View logs for current project
./scripts/qdrant-logs

# Follow logs in real-time
./scripts/qdrant-logs -f

# Filter by operation and level
./scripts/qdrant-logs --operation index_code --level ERROR

# Search with regex
./scripts/qdrant-logs --search "failed.*authentication"

# List all projects with logs
./scripts/qdrant-logs --list-projects

# Export last 100 logs as JSON
./scripts/qdrant-logs --tail 100 --export json > debug.json
```

### `test_http_api.sh`
**Purpose**: Test HTTP API endpoints
- Comprehensive API testing suite
- Tests indexing and search functionality
- **When to use**: Debugging HTTP API issues, verifying server functionality
```bash
./scripts/test_http_api.sh
```

### `fix_qdrant.sh`
**Purpose**: Fix common Qdrant issues
- Repairs permission problems
- Cleans up corrupted data
- **When to use**: When Qdrant fails to start or has data corruption
```bash
./scripts/fix_qdrant.sh
```

### `create-release.sh`
**Purpose**: Create a new release with detailed notes
- Extracts release notes from CHANGELOG.md
- Creates annotated git tag with full details
- **When to use**: When creating a new version release
```bash
# First update CHANGELOG.md with release notes
# Then run:
./scripts/create-release.sh v0.1.4
```

## 📦 Optional/Advanced

### `auto_indexer.py`
**Purpose**: [OPTIONAL] Standalone file watcher
- Alternative to built-in MCP server file watching
- Runs as separate process
- **When to use**: Advanced setups where you want file watching separate from MCP server
- **Note**: Most users should use `export QDRANT_RAG_AUTO_INDEX=true` instead
```bash
# Watch current directory
python scripts/auto_indexer.py

# Watch specific directory with options
python scripts/auto_indexer.py /path/to/project --debounce 5 --initial-index
```

## Environment Variables for Auto-Indexing

Instead of using scripts, enable auto-indexing via environment:

```bash
# Enable auto-indexing for current session
export QDRANT_RAG_AUTO_INDEX=true

# Toggle auto-indexing
unset QDRANT_RAG_AUTO_INDEX  # Disable
export QDRANT_RAG_AUTO_INDEX=true  # Enable

# Custom debounce time (default: 3 seconds)
export QDRANT_RAG_DEBOUNCE=5.0
```

## Deprecated Scripts

The following scripts have been removed as they're obsolete:
- `watch_and_index.sh` - Use built-in auto-indexing instead
- `check_mcp.py` - We use FastMCP now
- `start_server.sh` - Incorrect architecture
- `start_rag_server.sh` - References non-existent files
- `install_auto_indexer.sh` - Overly complex approach
- `install_git_hooks.sh` - Git hooks approach deprecated
- `toggle_auto_index.sh` - Just use environment variables
- `debug_models.sh` - Merged into manage_models.sh
- `list_models.sh` - Merged into manage_models.sh