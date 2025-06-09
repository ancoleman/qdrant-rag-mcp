#!/usr/bin/env python3
"""Debug jina model loading"""

import os
import sys
import logging
from pathlib import Path

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ['QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED'] = 'true'
os.environ['QDRANT_CONFIG_EMBEDDING_MODEL'] = 'jinaai/jina-embeddings-v3'

# Try setting HF environment variables
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print("Environment variables set:")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE', 'not set')}")
print(f"SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'not set')}")

try:
    from sentence_transformers import SentenceTransformer
    
    cache_folder = os.path.expanduser('~/.cache/qdrant-mcp/models')
    print(f"\nTrying to load model from cache: {cache_folder}")
    
    # Try loading with different settings
    model = SentenceTransformer(
        'jinaai/jina-embeddings-v3',
        cache_folder=cache_folder,
        trust_remote_code=True,
        device='cpu'  # Explicitly set device to avoid any GPU issues
    )
    
    print(f"\nModel loaded successfully!")
    print(f"Model path: {model._model_config}")
    print(f"Dimension: {model.get_sentence_embedding_dimension()}")
    
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nTrying fallback model...")
    try:
        model = SentenceTransformer(
            'jinaai/jina-embeddings-v2-base-en',
            cache_folder=cache_folder,
            device='cpu'
        )
        print(f"Fallback model loaded successfully!")
        print(f"Dimension: {model.get_sentence_embedding_dimension()}")
    except Exception as e2:
        print(f"Fallback also failed: {e2}")