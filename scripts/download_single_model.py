#!/usr/bin/env python3
"""
Download a single embedding model
"""
import os
import sys
from sentence_transformers import SentenceTransformer

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_single_model.py <model_name>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(project_root, 'data', 'models')
    
    print(f"Downloading {model_name} to {cache_dir}...")
    
    try:
        # Models that need trust_remote_code
        if model_name in ['nomic-ai/CodeRankEmbed', 'jinaai/jina-embeddings-v3']:
            model = SentenceTransformer(model_name, 
                                      cache_folder=cache_dir,
                                      trust_remote_code=True)
        else:
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
        
        print(f"✓ Successfully downloaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()