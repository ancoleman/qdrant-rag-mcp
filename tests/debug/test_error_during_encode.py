#!/usr/bin/env python3
"""Test what happens when encoding fails"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_error_during_encode():
    """Test the error path during encoding"""
    print("=" * 80)
    print("TESTING ERROR DURING ENCODE")
    print("=" * 80)
    
    from utils.embeddings import get_embeddings_manager
    
    # Get embeddings manager
    embeddings_manager = get_embeddings_manager()
    
    print("\n1. Testing normal encoding first...")
    try:
        normal_embedding = embeddings_manager.encode("def test(): pass", content_type="code")
        print(f"   Normal code encoding: {normal_embedding.shape[-1]}D")
    except Exception as e:
        print(f"   Normal encoding failed: {e}")
    
    print("\n2. Testing what causes encoding to fail...")
    
    # Test with extremely long text that might cause OOM
    print("\n   a) Testing with very long text...")
    very_long_text = "x" * 1000000  # 1M characters
    try:
        long_embedding = embeddings_manager.encode(very_long_text, content_type="code")
        print(f"   Long text encoding: {long_embedding.shape[-1]}D")
    except Exception as e:
        print(f"   Long text encoding failed: {type(e).__name__}: {e}")
    
    # Test concurrent access
    print("\n   b) Testing concurrent access...")
    import threading
    results = []
    errors = []
    
    def encode_in_thread(idx):
        try:
            embedding = embeddings_manager.encode(f"def test_{idx}(): pass", content_type="code")
            results.append((idx, embedding.shape[-1]))
        except Exception as e:
            errors.append((idx, str(e)))
    
    threads = []
    for i in range(10):
        t = threading.Thread(target=encode_in_thread, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"   Successful: {len(results)}, Failed: {len(errors)}")
    if errors:
        print(f"   First error: {errors[0]}")
    
    # Check dimensions of successful results
    dimensions = set(dim for _, dim in results)
    print(f"   Unique dimensions: {dimensions}")
    
    print("\n3. Testing our fix behavior...")
    
    # Manually check what happens with our new fallback logic
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        # Check model configs
        code_config = actual_manager.model_configs.get('code', {})
        general_config = actual_manager.model_configs.get('general', {})
        
        print(f"\n   Code model dimension: {code_config.get('dimension')}")
        print(f"   General model dimension: {general_config.get('dimension')}")
        print(f"   Can fall back to general: {code_config.get('dimension') == general_config.get('dimension')}")
        
        # Check fallback model
        fallback = code_config.get('fallback')
        if fallback:
            print(f"   Code fallback model: {fallback}")
            # Try to get fallback dimension
            try:
                # Temporarily set as primary to test
                original = code_config['name']
                code_config['name'] = fallback
                fallback_dim = actual_manager.get_dimension('code')
                code_config['name'] = original
                print(f"   Fallback dimension: {fallback_dim}D")
            except:
                print(f"   Could not determine fallback dimension")

if __name__ == "__main__":
    test_error_during_encode()