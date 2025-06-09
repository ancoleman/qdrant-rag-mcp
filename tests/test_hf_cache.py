#!/usr/bin/env python3
"""Test HuggingFace cache configuration"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables BEFORE importing any HF libraries
custom_cache = os.path.expanduser("~/Documents/repos/mcp-servers/qdrant-rag/data/models")
os.environ['HF_HOME'] = custom_cache
os.environ['HF_HUB_CACHE'] = custom_cache
os.environ['TRANSFORMERS_CACHE'] = custom_cache
os.environ['SENTENCE_TRANSFORMERS_HOME'] = custom_cache

print("Environment variables set:")
print(f"HF_HOME: {os.environ.get('HF_HOME')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")

# Now import
from sentence_transformers import SentenceTransformer

try:
    print(f"\nTrying to load jina-embeddings-v3...")
    model = SentenceTransformer(
        'jinaai/jina-embeddings-v3',
        cache_folder=custom_cache,
        trust_remote_code=True,
        device='cpu'
    )
    
    print(f"Model loaded successfully!")
    print(f"Dimension: {model.get_sentence_embedding_dimension()}")
    
    test_embedding = model.encode("test config file")
    print(f"Test embedding shape: {test_embedding.shape}")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()