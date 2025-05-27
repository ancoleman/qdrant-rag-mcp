# Qdrant MCP RAG Server - Quick Reference

## 🚀 Quick Start

```bash
# Initial setup (one time)
cd ~/mcp-servers/qdrant-rag
./scripts/setup.sh

# Daily startup (handles model downloads)
./scripts/start_server.sh

# Pre-download models (optional)
./scripts/download_models.sh

# Index a project
./scripts/index_project.sh /path/to/your/project

# View logs
docker-compose -f docker/docker-compose.yml logs -f rag-server

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## 🧠 Embedding Models

### Model Management
```bash
# Check current model
grep EMBEDDING_MODEL .env

# Pre-download models
./scripts/download_models.sh

# List downloaded models
ls -la ~/Library/Caches/qdrant-mcp/models/

# Check model sizes
du -sh ~/Library/Caches/qdrant-mcp/models/*
```

### Quick Model Selection
```bash
# Fast & Small (recommended start)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Better quality, still fast
EMBEDDING_MODEL=all-MiniLM-L12-v2

# Best general purpose
EMBEDDING_MODEL=all-mpnet-base-v2

# For code projects
EMBEDDING_MODEL=microsoft/codebert-base
```

### macOS Optimized Setup
```bash
# Apple Silicon (M1/M2/M3)
cat >> .env <<EOF
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOKENIZERS_PARALLELISM=false
MPS_DEVICE_ENABLE=1
EOF

# Intel Mac
cat >> .env <<EOF
EMBEDDING_MODEL=all-MiniLM-L12-v2
TOKENIZERS_PARALLELISM=false
EOF
```

### Switching Models
```bash
# 1. Update .env
sed -i '' 's/EMBEDDING_MODEL=.*/EMBEDDING_MODEL=all-mpnet-base-v2/' .env

# 2. Clear collections
curl -X DELETE http://localhost:6333/collections/code_collection
curl -X DELETE http://localhost:6333/collections/config_collection

# 3. Restart and re-index
docker-compose -f docker/docker-compose.yml restart
./scripts/index_project.sh /your/project
```

## 📁 Key File Locations

```
~/mcp-servers/qdrant-rag/
├── config/server_config.json     # Server configuration
├── config/mcp_manifest.json      # MCP service definition
├── docker/docker-compose.yml     # Docker setup
├── .env                          # Environment variables
├── src/                          # Source code
│   ├── qdrant_mcp_context_aware.py # Main server
│   ├── config.py                # Config handler
│   ├── indexers/                # Code/config indexers
│   └── utils/embeddings.py      # Embedding manager
└── logs/                         # Server logs

~/.claude-code/
├── mcp-servers.json             # Claude Code MCP config
└── workspace.json               # Project workspace config
```

## 🔧 Common Commands

### Module Testing
```bash
# Test config loading
python -c "from src.config import get_config; print(get_config().get('server.port'))"

# Test embeddings
python -c "from src.utils import get_embeddings_manager; em = get_embeddings_manager(); print(em.get_model_info())"

# Test indexers
python -c "from src.indexers import CodeIndexer; ci = CodeIndexer(); print('CodeIndexer ready')"

# List all MCP methods
cat config/mcp_manifest.json | jq '.methods[] | {name, category}'
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

# Check service status
docker-compose -f docker/docker-compose.yml ps
```

### Indexing Operations
```bash
# Index entire directory
curl -X POST http://localhost:8080/index_directory \
  -H "Content-Type: application/json" \
  -d '{"directory": "/path/to/project", "patterns": ["*.py", "*.js"]}'

# Index single file
curl -X POST http://localhost:8080/index_code \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.py"}'

# Update existing file
curl -X POST http://localhost:8080/update_index \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.py"}'

# Delete from index
curl -X POST http://localhost:8080/delete_document \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/file.py"}'
```

### Search Operations
```bash
# General search
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication middleware", "n_results": 5}'

# Code-specific search
curl -X POST http://localhost:8080/search_code \
  -H "Content-Type: application/json" \
  -d '{"query": "error handling", "filters": {"file_type": ".py"}}'

# Config search
curl -X POST http://localhost:8080/search_config \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection", "file_type": ".json"}'
```

### Health & Monitoring
```bash
# Health check
curl http://localhost:8080/health

# Server metrics
curl http://localhost:8080/metrics

# Qdrant health
curl http://localhost:6333/health

# Qdrant collections info
curl http://localhost:6333/collections

# Check MCP manifest
cat config/mcp_manifest.json | jq '.methods[].name'

# Validate manifest
python -m json.tool config/mcp_manifest.json

# List available methods
curl http://localhost:8080/health | jq '.methods'
```

## 🐛 Troubleshooting

### Service Won't Start
```bash
# Check if ports are in use
lsof -i :8080
lsof -i :6333

# Check Docker logs
docker-compose -f docker/docker-compose.yml logs

# Restart Docker
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up -d

# Validate configuration files
python -m json.tool config/server_config.json
python -m json.tool config/mcp_manifest.json
```

### MCP Manifest Issues
```bash
# Check manifest syntax
cat config/mcp_manifest.json | jq .

# List all method names
cat config/mcp_manifest.json | jq '.methods[].name'

# Check method parameters
cat config/mcp_manifest.json | jq '.methods[] | select(.name=="search")'

# Validate against Claude Code
# In Claude Code, check if methods appear correctly
```

### Indexing Issues
```bash
# Check file permissions
ls -la /path/to/file

# Verify Qdrant is running
curl http://localhost:6333/health

# Check collection exists
curl http://localhost:6333/collections/code_collection
```

### Search Not Working
```bash
# Verify indexed documents
curl http://localhost:6333/collections/code_collection

# Check server logs
tail -f logs/rag_server.log

# Test embedding model
curl -X POST http://localhost:8080/test_embeddings \
  -H "Content-Type: application/json" \
  -d '{"text": "test query"}'
```

## 📋 Environment Variables

```bash
# Required
QDRANT_HOST=localhost
QDRANT_PORT=6333
SERVER_PORT=8080

# Optional
QDRANT_API_KEY=your-api-key
LOG_LEVEL=INFO
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 🔄 Daily Workflow

1. **Morning Startup**:
   ```bash
   cd ~/mcp-servers/qdrant-rag
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. **Index New Files**:
   ```bash
   # After adding new files to your project
   ./scripts/index_project.sh /your/project
   ```

3. **Use with Claude Code**:
   ```bash
   # Natural language - MCP activates automatically
   claude-code "Create a service similar to our user service"
   claude-code "What's the database configuration?"
   ```

4. **End of Day**:
   ```bash
   # Optional - can leave running
   docker-compose -f docker/docker-compose.yml down
   ```

## 📦 Backup & Restore

### Backup Qdrant Data
```bash
# Backup
docker run --rm -v qdrant-rag_qdrant_storage:/data \
  -v $(pwd)/backups:/backup alpine \
  tar czf /backup/qdrant_backup_$(date +%Y%m%d).tar.gz -C /data .

# Restore
docker run --rm -v qdrant-rag_qdrant_storage:/data \
  -v $(pwd)/backups:/backup alpine \
  tar xzf /backup/qdrant_backup_20240115.tar.gz -C /data
```

## 🔐 Security Best Practices

1. **Never commit**:
   - `.env` files
   - API keys
   - Passwords
   - Private certificates

2. **Always use**:
   - Environment variables for secrets
   - Docker secrets in production
   - HTTPS in production
   - API authentication

3. **Regular updates**:
   ```bash
   # Update Docker images
   docker-compose -f docker/docker-compose.yml pull
   
   # Update Python packages
   pip install -r requirements.txt --upgrade
   ```

## 📊 Performance Tuning

```json
// For large codebases, adjust in server_config.json
{
  "indexing": {
    "chunk_size": 2000,
    "batch_size": 200
  },
  "search": {
    "max_results": 10,
    "score_threshold": 0.6
  }
}
```

## 🤝 Contributing

```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
./scripts/test.sh

# Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature

# Create pull request
```

## 📞 Support

- Check logs: `logs/rag_server.log`
- GitHub Issues: `github.com/yourname/qdrant-rag/issues`
- Documentation: `README.md`
