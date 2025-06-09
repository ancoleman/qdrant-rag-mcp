#!/usr/bin/env python3
"""Test loading jina-embeddings-v3 model"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ['QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED'] = 'true'
os.environ['QDRANT_CONFIG_EMBEDDING_MODEL'] = 'jinaai/jina-embeddings-v3'

try:
    from utils.specialized_embeddings import SpecializedEmbeddingManager
    
    print("Creating SpecializedEmbeddingManager...")
    manager = SpecializedEmbeddingManager()
    
    print("\nTrying to load config model...")
    model, config = manager.load_model('config')
    print(f"Successfully loaded: {config['name']}")
    print(f"Dimension: {config['dimension']}")
    
    print("\nTrying to encode a test string...")
    test_text = '{"test": "config"}'
    embeddings = manager.encode(test_text, content_type='config')
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dtype: {embeddings.dtype}")
    
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()