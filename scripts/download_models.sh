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

# Set cache directory
CACHE_DIR="${SENTENCE_TRANSFORMERS_HOME:-~/Library/Caches/qdrant-mcp/models}"
CACHE_DIR="${CACHE_DIR/#\~/$HOME}"

echo -e "${BLUE}=== Embedding Model Downloader ===${NC}"
echo -e "${YELLOW}Cache directory: ${CACHE_DIR}${NC}"
echo ""

# Available models (name:size:description)
declare -a MODELS=(
    "all-MiniLM-L6-v2:90MB:Fast, good for general use"
    "all-MiniLM-L12-v2:130MB:Better accuracy, still fast"
    "all-mpnet-base-v2:420MB:Excellent quality, general purpose"
    "all-distilroberta-v1:290MB:High quality, robust"
    "multi-qa-MiniLM-L6-cos-v1:90MB:Optimized for Q&A"
    "microsoft/codebert-base:440MB:Best for code understanding"
    "microsoft/unixcoder-base:550MB:Great for multiple programming languages"
    "Salesforce/codet5-small:220MB:Good for code generation"
    "intfloat/e5-large-v2:1.3GB:State-of-the-art quality"
    "BAAI/bge-large-en-v1.5:1.3GB:Excellent for retrieval"
)

# Function to check if model exists
check_model_exists() {
    local model=$1
    local model_safe=$(echo "$model" | sed 's/\/--/--/g')
    
    # Check different possible patterns
    for pattern in "models--sentence-transformers--$model_safe" "models--$model_safe" "models--${model_safe//\//--}"; do
        if [ -d "$CACHE_DIR/$pattern/snapshots" ]; then
            return 0
        fi
    done
    return 1
}

# Function to get model size  
get_model_size() {
    local model=$1
    local model_safe=$(echo "$model" | sed 's/\/--/--/g')
    
    # Find the actual directory
    for pattern in "models--sentence-transformers--$model_safe" "models--$model_safe" "models--${model_safe//\//--}"; do
        if [ -d "$CACHE_DIR/$pattern" ]; then
            size=$(du -sh "$CACHE_DIR/$pattern" | cut -f1)
            echo "$size"
            return
        fi
    done
    echo "Not found"
}

# Display available models
echo -e "${GREEN}Available Models:${NC}"
echo ""

for i in "${!MODELS[@]}"; do
    IFS=':' read -r model_name expected_size description <<< "${MODELS[$i]}"
    
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

echo "   0. Download all models"
echo "  99. Download custom model (enter name)"
echo ""

read -p "Select model(s) to download (comma-separated numbers): " choices

# Function to download a model
download_model() {
    local model=$1
    
    if check_model_exists "$model"; then
        echo -e "${YELLOW}Model $model already exists. Skipping...${NC}"
        return
    fi
    
    echo -e "${BLUE}Downloading $model...${NC}"
    
    # Use Python to download
    python3 -c "
from sentence_transformers import SentenceTransformer
import sys

model_name = '$model'
cache_dir = '$CACHE_DIR'

try:
    print(f'Downloading {model_name} from Hugging Face...')
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
    fi
}

# Process user choices
IFS=',' read -ra SELECTED <<< "$choices"

for choice in "${SELECTED[@]}"; do
    choice=$(echo $choice | xargs)  # Trim whitespace
    
    if [ "$choice" = "0" ]; then
        # Download all
        echo -e "${BLUE}Downloading all models...${NC}"
        for i in "${!MODELS[@]}"; do
            IFS=':' read -r model_name _ _ <<< "${MODELS[$i]}"
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
    elif [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODELS[@]}" ]; then
        # Download selected
        IFS=':' read -r model_name _ _ <<< "${MODELS[$((choice-1))]}"
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
