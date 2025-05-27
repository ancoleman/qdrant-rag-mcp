#!/bin/bash

# scripts/manage_models.sh - Unified model management tool
# Combines functionality from list_models.sh and debug_models.sh

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

CACHE_DIR="${SENTENCE_TRANSFORMERS_HOME:-~/Library/Caches/qdrant-mcp/models}"
CACHE_DIR="${CACHE_DIR/#\~/$HOME}"

# Map of directory names to model names
declare -A model_map=(
    ["models--sentence-transformers--all-MiniLM-L6-v2"]="all-MiniLM-L6-v2"
    ["models--sentence-transformers--all-MiniLM-L12-v2"]="all-MiniLM-L12-v2"
    ["models--sentence-transformers--all-mpnet-base-v2"]="all-mpnet-base-v2"
    ["models--sentence-transformers--all-distilroberta-v1"]="all-distilroberta-v1"
    ["models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1"]="multi-qa-MiniLM-L6-cos-v1"
    ["models--microsoft--codebert-base"]="microsoft/codebert-base"
    ["models--microsoft--unixcoder-base"]="microsoft/unixcoder-base"
    ["models--Salesforce--codet5-small"]="Salesforce/codet5-small"
    ["models--intfloat--e5-large-v2"]="intfloat/e5-large-v2"
    ["models--BAAI--bge-large-en-v1.5"]="BAAI/bge-large-en-v1.5"
)

usage() {
    echo -e "${BLUE}Usage:${NC}"
    echo "  $0 list       - List downloaded models"
    echo "  $0 debug      - Debug model cache issues"
    echo "  $0 set        - Set default model in .env"
    echo "  $0 test       - Test model loading"
    echo ""
}

list_models() {
    echo -e "${BLUE}=== Downloaded Models ===${NC}"
    echo -e "${YELLOW}Cache directory: ${CACHE_DIR}${NC}"
    echo ""
    
    if [ ! -d "$CACHE_DIR" ]; then
        echo -e "${RED}Cache directory does not exist!${NC}"
        echo "Run './scripts/download_models.sh' to download models"
        exit 1
    fi
    
    echo "Models found:"
    i=1
    declare -a available_models=()
    
    for dir in "$CACHE_DIR"/models--*; do
        if [ -d "$dir/snapshots" ]; then
            dir_name=$(basename "$dir")
            
            if [ -n "${model_map[$dir_name]}" ]; then
                model_name="${model_map[$dir_name]}"
                size=$(du -sh "$dir" | cut -f1)
                echo -e "  ${GREEN}$i.${NC} $model_name ($size)"
                available_models+=("$model_name")
                ((i++))
            fi
        fi
    done
    
    if [ ${#available_models[@]} -eq 0 ]; then
        echo -e "${YELLOW}No models found in cache${NC}"
        echo "Run './scripts/download_models.sh' to download models"
    else
        echo ""
        total_size=$(du -sh "$CACHE_DIR" | cut -f1)
        echo -e "${BLUE}Total disk usage: ${total_size}${NC}"
    fi
}

debug_cache() {
    echo -e "${BLUE}=== Model Cache Debug ===${NC}"
    echo "Cache directory: $CACHE_DIR"
    echo ""
    
    # Check if directory exists
    if [ ! -d "$CACHE_DIR" ]; then
        echo -e "${RED}ERROR: Cache directory does not exist!${NC}"
        echo "Expected at: $CACHE_DIR"
        echo ""
        echo "To fix:"
        echo "  mkdir -p \"$CACHE_DIR\""
        echo "  ./scripts/download_models.sh"
        exit 1
    fi
    
    # Directory structure
    echo "Directory structure:"
    find "$CACHE_DIR" -type d -not -path "$CACHE_DIR" | head -20
    echo ""
    
    # Config files
    echo "Config files found:"
    find "$CACHE_DIR" -name "config.json" -type f | head -10
    echo ""
    
    # Check permissions
    echo "Permissions check:"
    ls -ld "$CACHE_DIR"
    echo ""
}

set_default_model() {
    list_models
    
    if [ ${#available_models[@]} -eq 0 ]; then
        exit 1
    fi
    
    echo ""
    read -p "Enter model number to set as default: " model_choice
    
    if [ "$model_choice" -ge 1 ] && [ "$model_choice" -le "${#available_models[@]}" ]; then
        selected_model="${available_models[$((model_choice-1))]}"
        
        # Update .env
        if [ -f .env ]; then
            if grep -q "EMBEDDING_MODEL=" .env; then
                sed -i.bak "s|EMBEDDING_MODEL=.*|EMBEDDING_MODEL=$selected_model|" .env
                echo -e "${GREEN}Updated EMBEDDING_MODEL in .env to: $selected_model${NC}"
            else
                echo "EMBEDDING_MODEL=$selected_model" >> .env
                echo -e "${GREEN}Added EMBEDDING_MODEL to .env: $selected_model${NC}"
            fi
        else
            echo "EMBEDDING_MODEL=$selected_model" > .env
            echo -e "${GREEN}Created .env with EMBEDDING_MODEL: $selected_model${NC}"
        fi
    else
        echo -e "${RED}Invalid selection${NC}"
        exit 1
    fi
}

test_loading() {
    echo -e "${BLUE}=== Testing Model Loading ===${NC}"
    
    # Get current model from .env
    if [ -f .env ]; then
        current_model=$(grep "EMBEDDING_MODEL=" .env | cut -d'=' -f2)
        echo "Current model in .env: ${current_model:-not set}"
    fi
    echo ""
    
    python3 << 'EOF'
import os
from sentence_transformers import SentenceTransformer

cache_dir = os.path.expanduser("~/Library/Caches/qdrant-mcp/models")

# Get model from env
model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
print(f"Testing model: {model_name}")

try:
    print("Loading model...")
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    print("✓ Model loaded successfully!")
    
    # Test encoding
    test_text = "This is a test sentence."
    embedding = model.encode(test_text)
    print(f"✓ Encoding works! Dimension: {len(embedding)}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if model is downloaded: ./scripts/manage_models.sh list")
    print("2. Download models: ./scripts/download_models.sh")
    print("3. Check cache directory: ./scripts/manage_models.sh debug")
EOF
}

# Main script logic
case "$1" in
    list)
        list_models
        ;;
    debug)
        debug_cache
        ;;
    set)
        set_default_model
        ;;
    test)
        test_loading
        ;;
    "")
        list_models
        ;;
    *)
        usage
        exit 1
        ;;
esac