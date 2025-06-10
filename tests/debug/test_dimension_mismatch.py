#!/usr/bin/env python3
"""Test to reproduce the dimension mismatch error"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_dimension_mismatch():
    """Try to reproduce the dimension mismatch error"""
    print("=" * 80)
    print("TESTING DIMENSION MISMATCH")
    print("=" * 80)
    
    # Import after adding to path
    from utils.embeddings import get_embeddings_manager
    from utils.specialized_embeddings import get_specialized_embedding_manager
    
    # Get embeddings manager
    embeddings_manager = get_embeddings_manager()
    
    # Force eviction of CodeRankEmbed by loading other models
    print("\n1. Loading models to force eviction...")
    embeddings_manager.encode("test", content_type="general")  # Load general (384D)
    embeddings_manager.encode("test", content_type="config")   # Load config (1024D)
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Now directly check what happens when we encode with code content type
        print("\n2. Testing encoding with 'code' content type...")
        
        # Get the model that will be used
        model_name = actual_manager.get_model_name('code')
        print(f"Model name for 'code': {model_name}")
        
        # Check if model is loaded
        if model_name not in actual_manager.loaded_models:
            print(f"WARNING: {model_name} is NOT loaded!")
            print(f"Currently loaded: {list(actual_manager.loaded_models.keys())}")
        
        # Force a situation where general model might be used instead
        print("\n3. Simulating fallback scenario...")
        
        # Let's check the fallback logic
        print("\nChecking model configs:")
        for content_type in ['code', 'general']:
            config = actual_manager.model_configs.get(content_type, {})
            print(f"{content_type}: {config.get('name')} (dim={config.get('dimension')})")
            if 'fallback' in config:
                print(f"  Fallback: {config['fallback']}")
        
        # Now let's see what dimension we get
        print("\n4. Testing actual encoding...")
        test_code = "def test(): pass"
        
        # Encode with code content type
        code_embedding = embeddings_manager.encode(test_code, content_type="code")
        print(f"Code embedding shape: {code_embedding.shape}")
        print(f"Code embedding dimension: {code_embedding.shape[-1]}")
        
        # Check loaded models again
        print(f"\nLoaded models after encoding: {list(actual_manager.loaded_models.keys())}")
        
        # Let's also check what happens if we manually evict CodeRankEmbed
        print("\n5. Manually testing eviction...")
        if 'nomic-ai/CodeRankEmbed' in actual_manager.loaded_models:
            # Force eviction by loading 2 other models (since max is 2)
            embeddings_manager.encode("test", content_type="documentation")
            embeddings_manager.encode("test", content_type="general")
            print(f"After forced eviction: {list(actual_manager.loaded_models.keys())}")
            
            # Now try encoding as code again
            code_embedding2 = embeddings_manager.encode(test_code, content_type="code")
            print(f"Code embedding after eviction: {code_embedding2.shape[-1]}D")
            print(f"Models after re-encoding: {list(actual_manager.loaded_models.keys())}")

if __name__ == "__main__":
    test_dimension_mismatch()