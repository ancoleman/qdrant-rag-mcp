#!/bin/bash

# scripts/download_models_simple.sh - Simplified version that works

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load environment variables
if [ -f .env ]; then
    set -a
    source <(grep -v '^#' .env | grep -v '^$')
    set +a
fi

# Set cache directory - handle potential comments in env vars
TEMP_HOME="${SENTENCE_TRANSFORMERS_HOME:-~/Library/Caches/qdrant-mcp/models}"
# Remove any comments from the path
CACHE_DIR=$(echo "$TEMP_HOME" | sed 's/#.*//' | xargs)
CACHE_DIR="${CACHE_DIR/#\~/$HOME}"

echo -e "${BLUE}=== Embedding Model Downloader ===${NC}"
echo -e "${YELLOW}Cache directory: ${CACHE_DIR}${NC}"
echo ""

# Show current specialized embeddings configuration
echo -e "${GREEN}Current Specialized Embeddings Configuration:${NC}"
echo -e "  Code: ${QDRANT_CODE_EMBEDDING_MODEL:-nomic-ai/CodeRankEmbed}"
echo -e "  Config: ${QDRANT_CONFIG_EMBEDDING_MODEL:-jinaai/jina-embeddings-v3}"
echo -e "  Documentation: ${QDRANT_DOC_EMBEDDING_MODEL:-hkunlp/instructor-large}"
echo -e "  General: ${QDRANT_GENERAL_EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
echo -e "  Enabled: ${QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED:-true}"
echo ""

# Specialized Embedding Models (Phase 2 - Recommended)
declare -a SPECIALIZED_MODELS=(
    "nomic-ai/CodeRankEmbed:2.0GB:🚀 Optimized for code understanding (PRIMARY for code)"
    "jinaai/jina-embeddings-v3:2.0GB:⚙️ Specialized for configuration files (PRIMARY for config)"
    "hkunlp/instructor-large:1.5GB:📚 Optimized for documentation with instruction support (PRIMARY for docs)"
    "sentence-transformers/all-MiniLM-L6-v2:90MB:🔗 General purpose and backward compatibility"
    "microsoft/codebert-base:440MB:💻 Fallback for code (if CodeRankEmbed fails)"
    "jinaai/jina-embeddings-v2-base-en:1.0GB:⚙️ Fallback for config (if v3 fails)"
    "sentence-transformers/all-mpnet-base-v2:420MB:📚 Fallback for documentation"
)

# General Purpose Models (Legacy - Single Model Mode)
declare -a GENERAL_MODELS=(
    "all-MiniLM-L6-v2:90MB:Fast, good for general use"
    "all-MiniLM-L12-v2:130MB:Better accuracy, still fast"
    "all-mpnet-base-v2:420MB:Excellent quality, general purpose"
    "all-distilroberta-v1:290MB:High quality, robust"
    "multi-qa-MiniLM-L6-cos-v1:90MB:Optimized for Q&A"
    "microsoft/codebert-base:440MB:Good for code understanding"
    "microsoft/unixcoder-base:550MB:Great for multiple programming languages"
    "Salesforce/codet5-small:220MB:Good for code generation"
    "intfloat/e5-large-v2:1.3GB:State-of-the-art quality"
    "BAAI/bge-large-en-v1.5:1.3GB:Excellent for retrieval"
)

# Function to check if model exists
check_model_exists() {
    local model=$1
    # Convert model name to directory format (org/model -> models--org--model)
    local model_dir="models--${model//\/--}"
    model_dir="${model_dir//\/$/}"  # Remove trailing slash if any
    
    # For sentence-transformers models, check with prefix
    if [[ "$model" == "sentence-transformers/"* ]]; then
        if [ -d "$CACHE_DIR/models--${model//\/--}" ]; then
            return 0
        fi
    fi
    
    # Check general pattern
    if [ -d "$CACHE_DIR/models--${model//\/--}" ]; then
        return 0
    fi
    
    return 1
}

# Function to get model size  
get_model_size() {
    local model=$1
    local model_dir="models--${model//\/--}"
    
    if [ -d "$CACHE_DIR/$model_dir" ]; then
        size=$(du -sh "$CACHE_DIR/$model_dir" | cut -f1)
        echo "$size"
        return
    fi
    
    echo "Not found"
}

# Display available models
echo -e "${GREEN}🚀 SPECIALIZED EMBEDDINGS (Phase 2 - Recommended):${NC}"
echo -e "${BLUE}Content-type specific models for optimal performance${NC}"
echo ""

total_models=0
declare -a ALL_MODELS=()

for i in "${!SPECIALIZED_MODELS[@]}"; do
    IFS=':' read -r model_name expected_size description <<< "${SPECIALIZED_MODELS[$i]}"
    ALL_MODELS+=("${SPECIALIZED_MODELS[$i]}")
    
    # Check if already downloaded
    if check_model_exists "$model_name"; then
        actual_size=$(get_model_size "$model_name")
        status="${GREEN}✓ Downloaded ($actual_size)${NC}"
    else
        status="${YELLOW}Not downloaded${NC}"
    fi
    
    printf "  %2d. %-35s %-8s %b\n" $((i+1)) "$model_name" "[$expected_size]" "$status"
    echo "      $description"
    echo ""
done

specialized_count=${#SPECIALIZED_MODELS[@]}
total_models=$specialized_count

echo -e "${GREEN}📚 GENERAL PURPOSE MODELS (Legacy/Single Model Mode):${NC}"
echo ""

for i in "${!GENERAL_MODELS[@]}"; do
    IFS=':' read -r model_name expected_size description <<< "${GENERAL_MODELS[$i]}"
    ALL_MODELS+=("${GENERAL_MODELS[$i]}")
    
    # Check if already downloaded
    if check_model_exists "$model_name"; then
        actual_size=$(get_model_size "$model_name")
        status="${GREEN}✓ Downloaded ($actual_size)${NC}"
    else
        status="${YELLOW}Not downloaded${NC}"
    fi
    
    printf "  %2d. %-35s %-8s %b\n" $((i+specialized_count+1)) "$model_name" "[$expected_size]" "$status"
    echo "      $description"
    echo ""
done

total_models=${#ALL_MODELS[@]}

echo -e "${BLUE}=== QUICK OPTIONS ===${NC}"
echo "   0. Download all specialized models (recommended)"
echo "  88. Download all models (specialized + general)"
echo "  99. Download custom model (enter name)"
echo ""

read -p "Select model(s) to download (comma-separated numbers): " choices

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check if required packages are installed
    python3 -c "import einops" 2>/dev/null || {
        echo -e "${YELLOW}Installing einops (required for some models)...${NC}"
        pip install einops >/dev/null 2>&1 || echo -e "${RED}Warning: Failed to install einops${NC}"
    }
    
    python3 -c "import InstructorEmbedding" 2>/dev/null || {
        echo -e "${YELLOW}Installing InstructorEmbedding (required for instructor models)...${NC}"
        pip install InstructorEmbedding >/dev/null 2>&1 || echo -e "${RED}Warning: Failed to install InstructorEmbedding${NC}"
    }
    
    echo ""
}

# Function to download a model
download_model() {
    local model=$1
    
    if check_model_exists "$model"; then
        echo -e "${YELLOW}Model $model already exists. Skipping...${NC}"
        return
    fi
    
    echo -e "${BLUE}Downloading $model...${NC}"
    
    # Check if model needs trust_remote_code
    local trust_remote_code="False"
    if [[ "$model" == "nomic-ai/CodeRankEmbed" ]] || [[ "$model" == "jinaai/jina-embeddings-v3" ]]; then
        trust_remote_code="True"
        echo -e "${YELLOW}Note: This model requires trust_remote_code=True${NC}"
    fi
    
    # Use Python to download
    python3 -c "
from sentence_transformers import SentenceTransformer
import sys
import os

model_name = '$model'
cache_dir = '$CACHE_DIR'
trust_remote = $trust_remote_code

try:
    print(f'Downloading {model_name} from Hugging Face...')
    if trust_remote:
        model = SentenceTransformer(model_name, cache_folder=cache_dir, trust_remote_code=True)
    else:
        model = SentenceTransformer(model_name, cache_folder=cache_dir)
    print(f'Successfully downloaded {model_name}')
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        actual_size=$(get_model_size "$model")
        echo -e "${GREEN}✓ $model downloaded successfully ($actual_size)${NC}"
    else
        echo -e "${RED}✗ Failed to download $model${NC}"
        echo -e "${YELLOW}Tip: You may need to install additional dependencies or check your internet connection${NC}"
    fi
}

# Check dependencies first
check_dependencies

# Process user choices
IFS=',' read -ra SELECTED <<< "$choices"

for choice in "${SELECTED[@]}"; do
    choice=$(echo $choice | xargs)  # Trim whitespace
    
    if [ "$choice" = "0" ]; then
        # Download all specialized models (recommended)
        echo -e "${BLUE}Downloading all specialized models (recommended)...${NC}"
        for i in "${!SPECIALIZED_MODELS[@]}"; do
            IFS=':' read -r model_name _ _ <<< "${SPECIALIZED_MODELS[$i]}"
            download_model "$model_name"
            echo ""
        done
    elif [ "$choice" = "88" ]; then
        # Download all models (specialized + general)
        echo -e "${BLUE}Downloading all models (specialized + general)...${NC}"
        for i in "${!ALL_MODELS[@]}"; do
            IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$i]}"
            download_model "$model_name"
            echo ""
        done
    elif [ "$choice" = "99" ]; then
        # Custom model
        echo ""
        read -p "Enter model name: " custom_model
        if [ -n "$custom_model" ]; then
            download_model "$custom_model"
        fi
    elif [ "$choice" -ge 1 ] && [ "$choice" -le "$total_models" ]; then
        # Download selected
        IFS=':' read -r model_name _ _ <<< "${ALL_MODELS[$((choice-1))]}"
        download_model "$model_name"
        echo ""
    fi
done

# Show final status
echo ""
echo -e "${GREEN}=== Download Summary ===${NC}"
echo -e "${YELLOW}Models stored in: ${CACHE_DIR}${NC}"
echo ""

# List all downloaded models
echo "Downloaded models:"
found_models=false

# Create a temporary file to store model info
temp_file=$(mktemp)

# Find all models
for dir in "$CACHE_DIR"/models--*; do
    if [ -d "$dir/snapshots" ]; then
        dir_name=$(basename "$dir")
        size=$(du -sh "$dir" | cut -f1)
        
        # Extract model name
        model_name=""
        case "$dir_name" in
            models--sentence-transformers--*)
                model_name="${dir_name#models--sentence-transformers--}"
                ;;
            models--microsoft--*)
                model_name="microsoft/${dir_name#models--microsoft--}"
                ;;
            models--Salesforce--*)
                model_name="Salesforce/${dir_name#models--Salesforce--}"
                ;;
            models--intfloat--*)
                model_name="intfloat/${dir_name#models--intfloat--}"
                ;;
            models--BAAI--*)
                model_name="BAAI/${dir_name#models--BAAI--}"
                ;;
            models--nomic-ai--*)
                model_name="nomic-ai/${dir_name#models--nomic-ai--}"
                ;;
            models--jinaai--*)
                model_name="jinaai/${dir_name#models--jinaai--}"
                ;;
            models--hkunlp--*)
                model_name="hkunlp/${dir_name#models--hkunlp--}"
                ;;
        esac
        
        if [ -n "$model_name" ]; then
            echo "  - $model_name ($size)"
            echo "$model_name" >> "$temp_file"
            found_models=true
        fi
    fi
done

if [ "$found_models" = false ]; then
    echo "  (none)"
fi

# Show total disk usage
echo ""
if [ -d "$CACHE_DIR" ]; then
    total_size=$(du -sh "$CACHE_DIR" | cut -f1)
    echo -e "${BLUE}Total disk usage: ${total_size}${NC}"
fi

# Create model list file
echo ""
echo -e "${YELLOW}Creating model list...${NC}"
cp "$temp_file" "$CACHE_DIR/downloaded_models.txt"
echo -e "${GREEN}Model list saved to: $CACHE_DIR/downloaded_models.txt${NC}"

# Offer to update .env
echo ""
read -p "Would you like to set one of these models as default in .env? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Select a model to set as default:"
    
    # Read models from temp file
    i=1
    declare -a available_models=()
    while IFS= read -r model; do
        echo "  $i. $model"
        available_models+=("$model")
        ((i++))
    done < "$temp_file"
    
    if [ ${#available_models[@]} -gt 0 ]; then
        read -p "Enter number: " model_choice
        
        if [ "$model_choice" -ge 1 ] && [ "$model_choice" -le "${#available_models[@]}" ]; then
            selected_model="${available_models[$((model_choice-1))]}"
            
            # Update .env
            if grep -q "EMBEDDING_MODEL=" .env 2>/dev/null; then
                sed -i.bak "s|EMBEDDING_MODEL=.*|EMBEDDING_MODEL=$selected_model|" .env
                echo -e "${GREEN}Updated EMBEDDING_MODEL in .env to: $selected_model${NC}"
            else
                echo "EMBEDDING_MODEL=$selected_model" >> .env
                echo -e "${GREEN}Added EMBEDDING_MODEL to .env: $selected_model${NC}"
            fi
        fi
    fi
fi

# Clean up
rm -f "$temp_file"

echo ""
echo -e "${GREEN}Done!${NC}"
