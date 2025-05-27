#!/bin/bash
# This script starts the RAG server with the correct environment variables

# Navigate to project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

# Check for command line arguments
MODE="docker"
if [[ "$1" == "--local" ]]; then
    MODE="local"
fi

# Check if .env file exists
if [ -f "$ROOT_DIR/.env" ]; then
    echo "Loading environment variables from $ROOT_DIR/.env"
    
    # Export all variables from .env file
    export $(grep -v '^#' $ROOT_DIR/.env | xargs)
    
    echo "Environment variables loaded:"
    echo "EMBEDDING_MODEL: $EMBEDDING_MODEL"
    echo "MPS_DEVICE_ENABLE: $MPS_DEVICE_ENABLE"
    echo "TOKENIZERS_PARALLELISM: $TOKENIZERS_PARALLELISM"
else
    echo "Warning: .env file not found at $ROOT_DIR/.env"
fi

if [[ "$MODE" == "docker" ]]; then
    echo "Starting in Docker mode..."
    
    # Start Docker Compose
    cd "$ROOT_DIR/docker"
    docker compose down
    docker compose up -d
    
    # Wait a few seconds and show logs
    echo "Starting RAG server in Docker..."
    sleep 5
    docker logs rag_mcp_server
    
    echo ""
    echo "Note: MPS (Metal Performance Shaders) will NOT work in Docker mode,"
    echo "      as Docker containers run in Linux and MPS is macOS-only."
    echo "To use MPS with Apple Silicon, run with --local flag:"
    echo "  ./start_with_env.sh --local"
    
else
    echo "Starting in local mode (supports MPS on Apple Silicon)..."
    
    # Update path for local execution
    export PYTHONPATH=$ROOT_DIR
    
    # Modify SENTENCE_TRANSFORMERS_HOME to use local path
    if [[ "$SENTENCE_TRANSFORMERS_HOME" == *"~"* ]]; then
        # Expand ~ in path
        SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME/#\~/$HOME}"
        export SENTENCE_TRANSFORMERS_HOME=$SENTENCE_TRANSFORMERS_HOME
        echo "Expanded SENTENCE_TRANSFORMERS_HOME: $SENTENCE_TRANSFORMERS_HOME"
    fi
    
    # Start Qdrant in Docker if needed
    cd "$ROOT_DIR/docker"
    if ! docker ps | grep -q qdrant_mcp; then
        echo "Starting Qdrant in Docker..."
        docker compose up -d qdrant
    else
        echo "Qdrant already running in Docker"
    fi
    
    # Set the Qdrant host to localhost for local mode
    export QDRANT_HOST=localhost
    
    # Run the server locally
    echo "Starting RAG server locally with Python..."
    cd "$ROOT_DIR"
    python src/qdrant_mcp_server.py
fi