#!/usr/bin/env python3
"""
Download specialized embedding models with proper configuration
"""
import os
import sys
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Colors for terminal output
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
RED = '\033[0;31m'
NC = '\033[0m'  # No Color

def print_color(text, color=NC):
    print(f"{color}{text}{NC}")

def download_model(model_name, cache_dir, trust_remote_code=False):
    """Download a model with error handling"""
    try:
        print_color(f"Downloading {model_name}...", BLUE)
        
        # Special handling for models that need trust_remote_code
        if model_name in ['nomic-ai/CodeRankEmbed', 'jinaai/jina-embeddings-v3']:
            print_color(f"Note: {model_name} requires trust_remote_code=True", YELLOW)
            model = SentenceTransformer(model_name, 
                                      cache_folder=cache_dir,
                                      trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        print_color(f"✓ Successfully downloaded {model_name}", GREEN)
        return True
    except Exception as e:
        print_color(f"✗ Failed to download {model_name}: {str(e)}", RED)
        return False

def main():
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get cache directory
    cache_dir = os.path.expanduser(
        os.getenv('SENTENCE_TRANSFORMERS_HOME', '~/mcp-servers/qdrant-rag/data/models')
    )
    
    print_color("=== Specialized Embeddings Model Downloader ===", BLUE)
    print_color(f"Cache directory: {cache_dir}", YELLOW)
    print()
    
    # Define specialized models
    specialized_models = [
        {
            'name': os.getenv('QDRANT_CODE_EMBEDDING_MODEL', 'nomic-ai/CodeRankEmbed'),
            'type': 'Code',
            'size': '2.0GB',
            'trust_remote_code': True
        },
        {
            'name': os.getenv('QDRANT_CONFIG_EMBEDDING_MODEL', 'jinaai/jina-embeddings-v3'),
            'type': 'Config',
            'size': '2.0GB',
            'trust_remote_code': True
        },
        {
            'name': os.getenv('QDRANT_DOC_EMBEDDING_MODEL', 'hkunlp/instructor-large'),
            'type': 'Documentation',
            'size': '1.5GB',
            'trust_remote_code': False
        },
        {
            'name': os.getenv('QDRANT_GENERAL_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
            'type': 'General',
            'size': '90MB',
            'trust_remote_code': False
        }
    ]
    
    # Fallback models
    fallback_models = [
        {
            'name': os.getenv('QDRANT_CODE_EMBEDDING_FALLBACK', 'microsoft/codebert-base'),
            'type': 'Code Fallback',
            'size': '440MB',
            'trust_remote_code': False
        },
        {
            'name': os.getenv('QDRANT_CONFIG_EMBEDDING_FALLBACK', 'jinaai/jina-embeddings-v2-base-en'),
            'type': 'Config Fallback',
            'size': '1.0GB',
            'trust_remote_code': False
        },
        {
            'name': os.getenv('QDRANT_DOC_EMBEDDING_FALLBACK', 'sentence-transformers/all-mpnet-base-v2'),
            'type': 'Doc Fallback',
            'size': '420MB',
            'trust_remote_code': False
        }
    ]
    
    print_color("Primary Models:", GREEN)
    for model in specialized_models:
        print(f"  {model['type']}: {model['name']} [{model['size']}]")
    
    print()
    print_color("Fallback Models:", GREEN)
    for model in fallback_models:
        print(f"  {model['type']}: {model['name']} [{model['size']}]")
    
    print()
    response = input("Download all models? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    print()
    
    # Download models
    all_models = specialized_models + fallback_models
    success_count = 0
    
    for model in all_models:
        success = download_model(model['name'], cache_dir, model.get('trust_remote_code', False))
        if success:
            success_count += 1
        print()
    
    # Summary
    print_color("=== Download Summary ===", GREEN)
    print(f"Successfully downloaded: {success_count}/{len(all_models)} models")
    
    if success_count < len(all_models):
        print_color("\nNote: Some models failed to download.", YELLOW)
        print_color("You may need to install additional dependencies:", YELLOW)
        print("  pip install jina-embeddings")
        print("  pip install InstructorEmbedding")

if __name__ == "__main__":
    main()