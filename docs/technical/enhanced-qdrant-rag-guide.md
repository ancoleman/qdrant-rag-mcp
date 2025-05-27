# Enhanced Qdrant RAG Server Guide

This guide includes the latest updates and improvements to the Qdrant RAG MCP Server, with special focus on Apple Silicon support and local model management.

## Recent Improvements

1. **Improved Environment Variable Handling**
   - Added proper .env file loading with python-dotenv
   - Enhanced environment variable inheritance in Docker
   - Fixed environment variable conflicts between Docker and local execution

2. **Enhanced Embedding Model Handling**
   - Better model fallback mechanisms
   - Improved model cache directory configuration
   - Added support for local models with proper cache paths

3. **Apple Silicon Optimization**
   - Added Metal Performance Shaders (MPS) support for M-series chips
   - Device detection for optimal performance
   - Clear error messaging when MPS isn't available

4. **Dual-Mode Execution**
   - New Docker mode (compatibility with all platforms)
   - New Local mode (uses macOS native features like MPS)
   - Flexible switching between modes

## Setup Quick Start

Here's how to get started with the enhanced server:

```bash
# Clone the repository (if not already done)
cd ~/mcp-servers
git clone <repository-url> qdrant-rag
cd qdrant-rag

# Make scripts executable
chmod +x scripts/*.sh
chmod +x docker/start_with_env.sh

# Configure environment (review and edit as needed)
cp .env.example .env
```

## Running the Server

### Docker Mode (Best for compatibility)

This mode runs everything in Docker containers:

```bash
# Start using the enhanced script
./docker/start_with_env.sh

# Or use docker compose directly
cd docker
docker compose up -d
```

### Local Mode (Best for Apple Silicon)

This mode runs Qdrant in Docker but the RAG server directly on your Mac:

```bash
# Run with local flag to use MPS acceleration
./docker/start_with_env.sh --local
```

## Configuring Embedding Models

### Model Setup for Apple Silicon

To get maximum performance on M-series Macs, use these settings in your `.env` file:

```bash
# For macOS with Apple Silicon (M1/M2/M3):
EMBEDDING_MODEL=all-MiniLM-L12-v2
TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings
MPS_DEVICE_ENABLE=1  # Enable Metal Performance Shaders

# Model Cache Directory
SENTENCE_TRANSFORMERS_HOME=~/mcp-servers/qdrant-rag/data/models  # Local path
```

### Understanding Docker vs Local Mode

1. **Docker Mode** 
   - Runs everything in containers
   - Environment variables from `.env` pass through Docker
   - MPS acceleration is NOT available (containers run Linux)
   - Model cache is stored in container volumes

2. **Local Mode**
   - Runs only Qdrant in Docker
   - RAG server runs natively on your Mac
   - MPS acceleration IS available (when enabled)
   - Model cache is stored in your local filesystem
   - Uses your local Python installation

## Metal Performance Shaders (MPS) Support

MPS enables your RAG server to use the GPU on Apple Silicon devices:

1. **Requirements**
   - Apple Silicon Mac (M1/M2/M3)
   - macOS 12.3+
   - PyTorch with MPS support

2. **Setup**
   ```bash
   # Install PyTorch with MPS support
   pip install torch torchvision
   
   # Set environment variables
   MPS_DEVICE_ENABLE=1
   ```

3. **Verification**
   - When running in local mode with debug logging
   - You should see "Using Apple Metal Performance Shaders (MPS) backend" in logs
   - And "Using device: mps" when the model is loaded

4. **Troubleshooting**
   - If you see "Using device: cpu" even with MPS_DEVICE_ENABLE=1:
     - Check that you're running in local mode
     - Verify PyTorch is installed correctly
     - Ensure you're on macOS 12.3+

## Environment Variables in Detail

Here are the most important environment variables:

```bash
# Qdrant Configuration
QDRANT_HOST=localhost  # Use "qdrant" in Docker mode
QDRANT_PORT=6333
QDRANT_API_KEY=   # Optional

# Server Configuration
SERVER_PORT=8080
LOG_LEVEL=INFO

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L12-v2  # Or other model name
TOKENIZERS_PARALLELISM=false  # Recommended setting
MPS_DEVICE_ENABLE=1  # For Apple Silicon GPUs

# Model Cache Directory
SENTENCE_TRANSFORMERS_HOME=~/mcp-servers/qdrant-rag/data/models  # Local path
# SENTENCE_TRANSFORMERS_HOME=/app/data/models  # Docker path
```

## Example Workflows

### Indexing a Project

```bash
# If using Docker mode
docker exec rag_mcp_server python -c "from src.indexers import index_directory; index_directory('/claude-code/my-project')"

# If using Local mode
cd /Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag
python -c "from src.indexers import index_directory; index_directory('/path/to/my-project')"
```

### Searching with the RAG Server

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{"query": "implement authentication middleware", "n_results": 5}'
```

### Switching Models

```bash
# 1. Update .env
sed -i '' 's/EMBEDDING_MODEL=.*/EMBEDDING_MODEL=all-mpnet-base-v2/' .env

# 2. Restart the server
./docker/start_with_env.sh
# Or for local mode:
./docker/start_with_env.sh --local
```

## Troubleshooting

### Common Issues

1. **"MPS_DEVICE_ENABLE=1 but running on Linux aarch64"**
   - This message appears when running Docker mode. This is expected as Docker runs in Linux which doesn't support MPS.
   - Solution: Use local mode if you need MPS acceleration.

2. **"Error loading model X: Y"**
   - Model couldn't be loaded from cache or downloaded.
   - Check internet connection, disk space, and model name spelling.

3. **Container keeps restarting**
   - Health checks might be failing.
   - Check logs with `docker logs rag_mcp_server`.

### Docker Mode Logs

```bash
# View service logs
docker logs rag_mcp_server
```

### Local Mode Logs

Logs appear directly in the terminal when running in local mode.

## Advanced Configuration

### Custom Model Download Location

```bash
# In .env
SENTENCE_TRANSFORMERS_HOME=/custom/path/to/models

# Then run
mkdir -p /custom/path/to/models
chmod 755 /custom/path/to/models
```

### Pre-downloading Models

```python
# Create a script: download_model.py
from sentence_transformers import SentenceTransformer
import os

model_name = "all-MiniLM-L12-v2"
cache_dir = os.path.expanduser("~/mcp-servers/qdrant-rag/data/models")

print(f"Downloading {model_name}...")
model = SentenceTransformer(model_name_or_path=model_name, cache_folder=cache_dir)
print(f"Successfully downloaded model with dimension: {model.get_sentence_embedding_dimension()}")
```

Then run:
```bash
python download_model.py
```

## Best Practices

1. **Use Local Mode for Development on Mac**
   - Local mode gives you the benefit of MPS acceleration
   - Better performance on Apple Silicon
   - More direct debugging capability

2. **Use Docker Mode for Production/Deployment**
   - More consistent environment
   - Better isolation
   - Easier to deploy to non-Mac systems

3. **Balance Model Size and Performance**
   - Smaller models (all-MiniLM-L6-v2) are faster but less accurate
   - Larger models (all-mpnet-base-v2) are more accurate but slower
   - Test different models to find the right balance

## Complete Updates List

1. **In qdrant_mcp_context_aware.py**
   - Added python-dotenv loading
   - Improved environment variable handling
   - Enhanced embedding model initialization
   - Added MPS support for Apple Silicon
   - Better error handling for model loading
   - Improved fallback mechanisms
   - Integrated specialized indexers for code and config files
   - Enhanced search methods with better filtering
   - Improved directory indexing with exclude patterns

2. **In docker-compose.yml**
   - Added environment variable pass-through
   - Added model cache volume mapping
   - Improved Docker networking config
   - Fixed health check issues

3. **In Dockerfile**
   - Added python-dotenv installation
   - Added .env file handling
   - Improved container security
   - Enhanced error handling

4. **Added start_with_env.sh script**
   - Dual-mode execution (Docker and Local)
   - Simplified startup process
   - Better environment variable handling
   - Clearer error messages

## Enhanced Indexers Integration

With the specialized indexers now integrated, the RAG server provides:

1. **Improved Code Indexing**
   - Language-specific parsing (Python, JavaScript, Java, etc.)
   - Structure-aware chunking (functions, classes, methods)
   - Line number tracking for precise locations
   - Rich metadata about imports, dependencies, and code structure

2. **Advanced Config Handling**
   - Support for multiple formats (JSON, XML, YAML, TOML, INI, ENV)
   - Hierarchical structure preservation
   - Path-based navigation and filtering
   - Schema extraction for better understanding

3. **Superior Search Experience**
   - Filter by language, chunk type, file type
   - Search within specific config paths
   - Get more meaningful results with structural context
   - Better result formatting with line numbers and previews

These enhancements significantly improve the quality of RAG responses by providing more precise and contextually relevant results.