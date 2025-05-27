# Claude Code + Qdrant RAG Setup & Usage Guide

This guide covers how to set up and use the Qdrant RAG server with Claude Code, plus HTTP testing methods.

## 🎯 Overview

The Qdrant RAG server operates in two modes:
1. **MCP Server Mode**: Integrates directly with Claude Code for seamless AI-assisted development
2. **HTTP API Mode**: Provides REST endpoints for testing and standalone usage

## 🚀 Quick Start

### Option 1: MCP Mode (Recommended for Claude Code)

```bash
# Start the MCP server (Docker mode)
./docker/start_with_env.sh

# Start with MPS acceleration (Local mode - macOS only)
./docker/start_with_env.sh --local
```

### Option 2: HTTP API Mode (For Testing)

```bash
# Start the HTTP wrapper (runs on port 8081)
cd /Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag
export $(grep -v '^#' .env | xargs)
python src/http_server.py
```

## 📋 Claude Code Integration

### Prerequisites

1. **Claude Code Installation**: Ensure you have Claude Code installed
2. **MCP Configuration**: The server should auto-configure during setup

### Manual Claude Code Configuration

If auto-configuration didn't work, manually configure Claude Code:

1. **Create MCP Configuration File**: `~/.claude-code/mcp-servers.json`

```json
{
  "servers": [
    {
      "name": "qdrant-rag",
      "type": "http",
      "config": {
        "url": "http://localhost:8080",
        "headers": {
          "Content-Type": "application/json"
        }
      },
      "start_command": "cd ~/mcp-servers/qdrant-rag && ./docker/start_with_env.sh",
      "health_check": {
        "endpoint": "/health",
        "interval": 30
      },
      "auto_start": true,
      "auto_use": {
        "enabled": true,
        "triggers": {
          "code_patterns": ["implement", "create", "refactor", "similar to", "find function", "search code"],
          "config_patterns": ["configuration", "settings", "parameter", "database config"],
          "search_patterns": ["find", "search", "look for", "where is", "how does"]
        }
      }
    }
  ]
}
```

2. **Or Use Global Installation** (Recommended):

```bash
# Run the global installation script
./install_global.sh

# This automatically:
# - Creates ~/.mcp-servers/qdrant-rag symlink
# - Sets up context-aware runner script
# - Configures Claude Code with user scope (-s user)
```

**Important**: The global installation uses the `-s user` flag to make the MCP server available across ALL your projects. This is the recommended approach for the context-aware server.

### Verifying Global Installation

After running `install_global.sh`, verify it works globally:

```bash
# Test in different directories
cd ~/projects/my-app
claude mcp list  # Should show qdrant-rag

cd ~
claude mcp list  # Should STILL show qdrant-rag
```

### Using with Claude Code

Once configured, Claude Code will automatically use the RAG server when you ask questions like:

- *"Find functions similar to user authentication"*
- *"Where is the database configuration?"*
- *"Show me how error handling works in this codebase"*
- *"Find all the logging functions"*

### Claude Code Workflow

1. **Index Your Project**:
   ```bash
   # Claude Code will typically handle this automatically
   # Or manually trigger via: ./scripts/index_project.sh /path/to/your/project
   ```

2. **Ask Natural Language Questions**:
   - Claude Code detects when to use RAG based on trigger patterns
   - Provides contextually relevant code snippets with line numbers
   - Maintains conversation context across queries

## 🧪 HTTP API Testing

### Starting the HTTP Server

```bash
# Method 1: Using environment variables
cd /Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag
export $(grep -v '^#' .env | xargs)
python src/http_server.py

# Method 2: Direct execution (uses .env automatically)
python src/http_server.py
```

### Available HTTP Endpoints

#### Health & Status

```bash
# Check server health
curl http://localhost:8081/health

# Check available collections
curl http://localhost:8081/collections
```

#### Indexing Operations

```bash
# Index a single code file
curl -X POST http://localhost:8081/index_code \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.py"}'

# Index a configuration file
curl -X POST http://localhost:8081/index_config \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/config.json"}'

# Index an entire directory
curl -X POST http://localhost:8081/index_directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/project",
    "patterns": ["*.py", "*.js", "*.json", "*.yaml"],
    "exclude_patterns": ["**/node_modules/**", "**/.git/**", "**/venv/**"]
  }'
```

#### Search Operations

```bash
# General search across all collections
curl -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication middleware", "n_results": 5}'

# Code-specific search with filters
curl -X POST http://localhost:8081/search_code \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling",
    "language": "python",
    "chunk_type": "function",
    "n_results": 3
  }'

# Configuration search with path filtering
curl -X POST http://localhost:8081/search_config \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection",
    "file_type": "json",
    "path": "database.connection"
  }'
```

### Response Format Examples

#### Successful Code Search Response

```json
{
  "results": [
    {
      "score": 0.85,
      "file_path": "/path/to/auth.py",
      "language": "python",
      "chunk_type": "function",
      "content": "Lines 45-67: def authenticate_user(username, password):\n    ...",
      "line_range": {"start": 45, "end": 67},
      "imports": ["bcrypt", "jwt"],
      "classes": ["UserAuth"],
      "functions": ["authenticate_user", "validate_token"]
    }
  ],
  "query": "authentication",
  "count": 1,
  "filters_applied": {
    "language": "python",
    "chunk_type": "function"
  }
}
```

#### Successful Config Search Response

```json
{
  "results": [
    {
      "score": 0.92,
      "file_path": "/path/to/config.json",
      "file_type": "json",
      "path": "database.connection.host",
      "value": "localhost",
      "content": "database.connection.host: localhost",
      "depth": 3
    }
  ],
  "query": "database host",
  "count": 1
}
```

## 🔧 Configuration Options

### Environment Variables

Key configuration in your `.env` file:

```bash
# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L12-v2  # or all-MiniLM-L6-v2 for faster performance
TOKENIZERS_PARALLELISM=false
MPS_DEVICE_ENABLE=1  # Enable MPS on Apple Silicon

# Server Configuration
SERVER_PORT=8080  # Port for HTTP test server (not used by MCP)
QDRANT_HOST=localhost  # or 'qdrant' in Docker mode
QDRANT_PORT=6333

# Model Cache
SENTENCE_TRANSFORMERS_HOME=~/mcp-servers/qdrant-rag/data/models
```

### Docker vs Local Mode

| Feature | Docker Mode | Local Mode |
|---------|-------------|------------|
| **Command** | `./docker/start_with_env.sh` | `./docker/start_with_env.sh --local` |
| **MPS Support** | ❌ No (Linux container) | ✅ Yes (native macOS) |
| **Compatibility** | All platforms | macOS only |
| **Performance** | Good | Better on Apple Silicon |
| **Debugging** | Container logs | Direct terminal output |
| **Use Case** | Production/CI | Development on Mac |

## 🐛 Troubleshooting

### Common Issues

#### MCP Server Not Starting

```bash
# Check Docker containers
docker ps | grep -E "(qdrant|rag)"

# Check container logs
docker logs rag_mcp_server
docker logs qdrant_mcp

# Restart services
./docker/start_with_env.sh
```

#### Claude Code Not Detecting RAG Server

1. **Check MCP Configuration**:
   ```bash
   cat ~/.claude-code/mcp-servers.json
   ```

2. **Verify Server is Running**:
   ```bash
   curl http://localhost:8080/health
   ```

3. **Reconfigure Claude Code**:
   ```bash
   ./install_global.sh
   ```

#### MCP Server Not Available Globally

**Symptom**: `claude mcp list` shows no servers in other directories

**Fix**:
```bash
# You probably added it without -s user flag
claude mcp remove qdrant-rag
claude mcp add qdrant-rag -s user ~/.mcp-servers/qdrant-rag-global.sh
```

#### Context Not Detecting Project

**Symptom**: `get_context()` shows no project

**Fix**:
1. Ensure you're in a project directory with markers (.git, package.json, etc.)
2. Add a `.project` file to mark project root
3. Check working directory is preserved in runner script

#### HTTP Server Connection Issues

```bash
# Check if HTTP server is running
curl http://localhost:8081/health

# Check process
ps aux | grep http_server.py

# Restart HTTP server
pkill -f "python src/http_server.py"
python src/http_server.py
```

#### Search Returns No Results

1. **Check if content is indexed**:
   ```bash
   curl http://localhost:6333/collections/code_collection
   curl http://localhost:6333/collections/config_collection
   ```

2. **Lower search threshold**: Temporarily disable score filtering in search

3. **Verify embedding model**: Check logs for successful model loading

#### MPS Not Working

- **In Docker Mode**: Expected - MPS only works in local mode
- **In Local Mode**: 
  - Verify: `MPS_DEVICE_ENABLE=1` in `.env`
  - Check logs for: `"Using Apple Metal Performance Shaders (MPS) backend"`
  - Ensure: macOS 12.3+ and Apple Silicon

### Debugging Commands

```bash
# Check server status
curl http://localhost:8081/health
curl http://localhost:6333/health

# View Qdrant dashboard
open http://localhost:6333/dashboard

# Check logs
tail -f /tmp/http_server.log  # HTTP server logs
docker logs -f rag_mcp_server  # MCP server logs
docker logs -f qdrant_mcp     # Qdrant logs

# Check collections
curl http://localhost:6333/collections

# Test indexing
curl -X POST http://localhost:8081/index_code \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/qdrant_mcp_context_aware.py"}'
```

### Performance Optimization

#### For Development (macOS)

```bash
# Use local mode with MPS
./docker/start_with_env.sh --local

# Use faster model for quick testing
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### For Production

```bash
# Use Docker mode for consistency
./docker/start_with_env.sh

# Use higher quality model
EMBEDDING_MODEL=all-mpnet-base-v2
```

## 🔄 Auto-Indexing (Optional)

The MCP server includes built-in file watching for automatic reindexing as your code changes.

### Enabling Auto-Indexing

```bash
# Option 1: Per-session
export QDRANT_RAG_AUTO_INDEX=true
claude

# Option 2: Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export QDRANT_RAG_AUTO_INDEX=true' >> ~/.bashrc
echo 'export QDRANT_RAG_DEBOUNCE=5.0' >> ~/.bashrc  # Optional: adjust debounce time
```

### How It Works

- **Automatic Detection**: Watches for changes in relevant files (.py, .js, .json, etc.)
- **Smart Debouncing**: Waits for changes to settle before reindexing (default: 3 seconds)
- **Efficient Updates**: Only reindexes changed files, not the entire project
- **Background Processing**: Runs in a separate thread without blocking MCP operations

### Configuration Options

```bash
# Enable/disable auto-indexing
export QDRANT_RAG_AUTO_INDEX=true|false

# Set debounce time (seconds)
export QDRANT_RAG_DEBOUNCE=5.0

# Watch specific directory (default: current directory)
export QDRANT_RAG_WATCH_DIR=/path/to/watch
```

## 📖 Best Practices

### 1. Indexing Strategy

- **Start Small**: Index a few key files first
- **Use Exclusions**: Exclude `node_modules`, `.git`, `venv` directories
- **Re-index Changes**: Update index when code changes significantly
- **Monitor Size**: Large codebases may need chunking adjustments

### 2. Search Optimization

- **Use Specific Queries**: "user authentication function" vs "auth"
- **Leverage Filters**: Filter by language, file type, or chunk type
- **Adjust Thresholds**: Lower score threshold for broader results
- **Combine Searches**: Use both code and config searches

### 3. Claude Code Integration

- **Natural Questions**: Ask questions as you would to a colleague
- **Context Matters**: Reference specific files or functions for better results
- **Iterative Queries**: Build on previous searches for deeper understanding
- **File Organization**: Well-organized code yields better RAG results

## 🎯 Usage Examples

### Typical Claude Code Interactions

```
You: "Find all the authentication functions in our codebase"
Claude: *Uses RAG to search and provides relevant functions with context*

You: "How is database connection configured?"
Claude: *Searches config files and shows connection settings*

You: "Show me error handling patterns similar to the user service"
Claude: *Finds and compares error handling across similar services*
```

### HTTP API Testing Workflow

```bash
# 1. Start servers
./docker/start_with_env.sh --local
python src/http_server.py

# 2. Index your project
curl -X POST http://localhost:8081/index_directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/your/project"}'

# 3. Test searches
curl -X POST http://localhost:8081/search_code \
  -H "Content-Type: application/json" \
  -d '{"query": "your search term", "language": "python"}'

# 4. Iterate and refine
# Adjust queries, filters, and thresholds based on results
```

## 🔗 Integration Patterns

### MCP Server Architecture

```
Claude Code ←→ MCP Protocol ←→ Qdrant RAG Server ←→ Qdrant Vector DB
                                        ↓
                               Specialized Indexers
                               (Code + Config)
```

### HTTP API Architecture

```
HTTP Client ←→ FastAPI Server ←→ RAG Server ←→ Qdrant Vector DB
                    ↓
            REST API Endpoints
            (Testing & Integration)
```

### Data Flow

1. **Indexing**: Files → Specialized Indexers → Chunks → Embeddings → Qdrant
2. **Searching**: Query → Embeddings → Vector Search → Ranked Results → Formatted Response
3. **Claude Integration**: Natural Language → MCP Protocol → RAG Search → Contextual Response

This setup provides a powerful semantic search capability that enhances your development workflow with Claude Code while also offering flexible HTTP API access for testing and integration.