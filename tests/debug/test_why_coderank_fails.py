#!/usr/bin/env python3
"""Test to understand why CodeRankEmbed might fail and need fallback"""

import os
import sys
from pathlib import Path
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_coderank_failure_scenarios():
    """Test various scenarios that might cause CodeRankEmbed to fail"""
    print("=" * 80)
    print("TESTING WHY CODERANKEDEMBED MIGHT FAIL")
    print("=" * 80)
    
    from utils.specialized_embeddings import SpecializedEmbeddingManager
    from utils.embeddings import get_embeddings_manager
    
    # Get the manager
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        print("\n1. Checking memory limits...")
        print(f"   Max models in memory: {actual_manager.max_models_in_memory}")
        print(f"   Memory limit GB: {actual_manager.memory_limit_gb}")
        print(f"   Current memory used: {actual_manager.total_memory_used_gb}GB")
        
        # Check model memory estimates
        code_model = actual_manager.model_configs['code']['name']
        estimated_memory = actual_manager._estimate_model_memory(code_model)
        print(f"   CodeRankEmbed estimated memory: {estimated_memory}GB")
        
        print("\n2. Testing model loading directly...")
        try:
            # Clear all models first
            actual_manager.clear_cache()
            print("   Cleared all models from cache")
            
            # Try to load CodeRankEmbed
            print(f"\n   Loading {code_model}...")
            model, config = actual_manager.load_model('code')
            print(f"   ✓ Successfully loaded {code_model}")
            
            # Test encoding
            test_text = "def hello(): pass"
            embedding = model.encode(test_text, show_progress_bar=False)
            print(f"   ✓ Test encoding successful: {embedding.shape}")
            
        except Exception as e:
            print(f"   ✗ Failed to load {code_model}: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        print("\n3. Testing eviction scenario...")
        try:
            # Load models to fill the cache
            print("   Loading config model...")
            actual_manager.load_model('config')
            print(f"   Models loaded: {list(actual_manager.loaded_models.keys())}")
            
            print("   Loading documentation model...")
            actual_manager.load_model('documentation')
            print(f"   Models loaded: {list(actual_manager.loaded_models.keys())}")
            
            # Now try to load code model - should evict something
            print("   Loading code model (should trigger eviction)...")
            model, config = actual_manager.load_model('code')
            print(f"   Models loaded after eviction: {list(actual_manager.loaded_models.keys())}")
            
            # Check if CodeRankEmbed is still there
            if code_model in actual_manager.loaded_models:
                print(f"   ✓ {code_model} is still loaded")
            else:
                print(f"   ✗ {code_model} was evicted!")
                
        except Exception as e:
            print(f"   ✗ Error during eviction test: {type(e).__name__}: {e}")
        
        print("\n4. Testing active model protection...")
        # The active_models set should protect models from eviction during use
        print(f"   Active models: {actual_manager.active_models}")
        
        # Check if active models are protected from eviction
        print("\n5. Testing error scenarios that trigger fallback...")
        
        # Simulate various errors
        print("\n   a) Testing with None text...")
        try:
            result = embeddings_manager.encode(None, content_type="code")
            print(f"   Result: {result.shape}")
        except Exception as e:
            print(f"   Error: {type(e).__name__}: {e}")
        
        print("\n   b) Testing with empty list...")
        try:
            result = embeddings_manager.encode([], content_type="code")
            print(f"   Result: {result.shape}")
        except Exception as e:
            print(f"   Error: {type(e).__name__}: {e}")
        
        print("\n   c) Testing memory pressure...")
        # Check current memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"   Current memory: {mem.available / (1024**3):.1f}GB available, {mem.percent}% used")
        except:
            print("   Could not check memory")
        
        print("\n6. Checking if it's a threading issue...")
        print("   In our concurrent test, some threads got fallback.")
        print("   This suggests a race condition in model loading/eviction.")
        print("   The issue is likely:")
        print("   - Thread 1: Starts loading CodeRankEmbed")
        print("   - Thread 2: Evicts CodeRankEmbed before Thread 1 finishes")
        print("   - Thread 1: Can't find the model it was trying to use")
        print("   - Thread 1: Falls back to general model")

if __name__ == "__main__":
    test_coderank_failure_scenarios()