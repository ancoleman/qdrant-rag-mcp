# Qdrant RAG with MPS Quick Reference

This guide provides quick reference for running the Qdrant RAG Server with Metal Performance Shaders (MPS) acceleration on Apple Silicon and using the enhanced indexers.

## 🚀 Quick Start

```bash
# Start with Docker mode (compatible everywhere, but NO MPS)
./docker/start_with_env.sh

# Start with Local mode (Apple Silicon with MPS)
./docker/start_with_env.sh --local
```

## 💻 Modes Comparison

| Feature | Docker Mode | Local Mode |
|---------|------------|------------|
| Runs On | Any OS | macOS only |
| MPS Support | ❌ No | ✅ Yes |
| Qdrant | In Docker | In Docker |
| RAG Server | In Docker | Native macOS |
| Setup | Simpler | More advanced |
| Performance | Good | Better on M-series |
| Model Cache | Container volume | Local directory |

## 🔧 Environment Setup (.env)

```bash
# For Apple Silicon MPS Performance (Local Mode)
EMBEDDING_MODEL=all-MiniLM-L12-v2
TOKENIZERS_PARALLELISM=false
MPS_DEVICE_ENABLE=1
SENTENCE_TRANSFORMERS_HOME=~/mcp-servers/qdrant-rag/data/models

# For Docker Mode
# SENTENCE_TRANSFORMERS_HOME=/app/data/models
```

## 📋 Common Commands

```bash
# View Docker container logs
docker logs rag_mcp_server

# Check if MPS is being used (look for "Using device: mps")
docker logs rag_mcp_server | grep "device:"

# Restart Docker containers
docker compose -f docker/docker-compose.yml restart

# Check Qdrant is running
curl http://localhost:6333/health

# Test RAG server
curl http://localhost:8080/health
```

## 📚 Enhanced Indexing Features

The server now uses specialized indexers for better context and understanding:

### Code Indexing

```bash
# Index code files with enhanced features
curl -X POST http://localhost:8080/index_code \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/file.py"
  }'

# Search code with language filtering
curl -X POST http://localhost:8080/search_code \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication middleware",
    "language": "python",
    "chunk_type": "function"
  }'
```

Supported languages:
- Python (.py)
- JavaScript (.js, .jsx)
- TypeScript (.ts, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h)
- C# (.cs)
- Go (.go)
- Ruby (.rb)
- Rust (.rs)
- PHP (.php)
- Swift (.swift)
- Kotlin (.kt)
- And more!

### Config Indexing

```bash
# Index config files with enhanced features
curl -X POST http://localhost:8080/index_config \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/config.json"
  }'

# Search config with path filtering
curl -X POST http://localhost:8080/search_config \
  -H "Content-Type: application/json" \
  -d '{
    "query": "database connection",
    "file_type": "json",
    "path": "database.connection"
  }'
```

Supported config formats:
- JSON (.json)
- XML (.xml)
- YAML (.yaml, .yml)
- TOML (.toml)
- INI (.ini)
- Environment (.env)
- Properties (.properties)
- Config (.config, .conf, .cfg)

### Directory Indexing

```bash
# Index entire project with enhanced patterns
curl -X POST http://localhost:8080/index_directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/project",
    "patterns": ["*.py", "*.js", "*.json", "*.yaml"],
    "exclude_patterns": ["**/node_modules/**", "**/.git/**"]
  }'
```

## 🔄 Workflow Tips

1. **Development on Mac**:
   - Use local mode with MPS for best performance
   - `./docker/start_with_env.sh --local`
   - Get logs directly in terminal

2. **Deployment/Production**:
   - Use Docker mode for consistency
   - `./docker/start_with_env.sh`
   - Check logs with `docker logs`

3. **Switching Models**:
   - Update .env file: `EMBEDDING_MODEL=new-model-name`
   - Restart using appropriate mode script

4. **Optimal Indexing**:
   - Use `index_directory` with appropriate `exclude_patterns`
   - Re-index after code changes
   - Use language and chunk_type filters for precise code search

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Using device: cpu" in local mode | Check MPS_DEVICE_ENABLE=1 is set |
| No MPS in Docker | Expected - use local mode instead |
| Connection error | Check Qdrant is running - `curl http://localhost:6333/health` |
| Model not found | Check internet access and model name spelling |
| Indexing fails | Try fallback indexer, verify file exists and access permissions |

## 🔍 Verifying MPS Usage

When MPS is correctly configured and available:

```
Loading embedding model: all-MiniLM-L12-v2 from cache: ~/data/models
Using Apple Metal Performance Shaders (MPS) backend
Using device: mps
Successfully loaded embedding model: all-MiniLM-L12-v2
```

When MPS is unavailable (e.g., in Docker):

```
Loading embedding model: all-MiniLM-L12-v2 from cache: /app/data/models
MPS_DEVICE_ENABLE=1 but running on Linux aarch64
Note: MPS is only available on macOS with Apple Silicon (M1/M2/M3)
Using device: cpu
```