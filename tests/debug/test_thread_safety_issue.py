#!/usr/bin/env python3
"""Test to identify the exact thread safety issue"""

import os
import sys
from pathlib import Path
import threading
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_thread_safety_issue():
    """Identify the exact thread safety issue"""
    print("=" * 80)
    print("IDENTIFYING THREAD SAFETY ISSUE")
    print("=" * 80)
    
    from utils.embeddings import get_embeddings_manager
    
    # Get the manager
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        print(f"\n1. Initial state:")
        print(f"   Max models: {actual_manager.max_models_in_memory}")
        print(f"   Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        print("\n2. The race condition scenario:")
        print("   When multiple threads try to encode with 'code' content type:")
        print("   - All threads call load_model('code')")
        print("   - load_model checks if CodeRankEmbed is already loaded")
        print("   - If not loaded, it tries to load it")
        print("   - But without proper locking, multiple threads might:")
        print("     a) All see it's not loaded")
        print("     b) All try to load it simultaneously")
        print("     c) Or one loads it while another evicts it")
        
        print("\n3. The specific issue in our code:")
        print("   In load_model() around line 375:")
        print("   - It marks the model as 'active' to prevent eviction")
        print("   - But there's a race between checking loaded_models and marking active")
        
        print("\n4. Testing the race condition...")
        
        # Test what happens when we simulate the race
        results = []
        errors = []
        lock = threading.Lock()
        
        def test_encode(thread_id):
            try:
                # Each thread tries to encode
                result = embeddings_manager.encode(f"def test_{thread_id}(): pass", content_type="code")
                with lock:
                    results.append((thread_id, result.shape[-1]))
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))
        
        # Create threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=test_encode, args=(i,))
            threads.append(t)
        
        print("   Starting 5 threads simultaneously...")
        # Start all threads at once
        for t in threads:
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        print(f"\n   Results: {len(results)} successful, {len(errors)} failed")
        if results:
            dimensions = set(dim for _, dim in results)
            print(f"   Dimensions: {dimensions}")
            if 384 in dimensions:
                print("   ✗ FOUND THE BUG: Some threads got 384D!")
        if errors:
            print(f"   First error: {errors[0]}")
        
        print("\n5. The root cause:")
        print("   Without thread-safe locking in load_model() and encode():")
        print("   - Threads can interfere with each other's model loading")
        print("   - The 'active_models' set isn't thread-safe")
        print("   - The OrderedDict operations aren't thread-safe")
        print("   - Model eviction can happen while another thread is using the model")
        
        print("\n6. Why fallback happens:")
        print("   When load_model() or encode() fails due to race conditions,")
        print("   the exception handler falls back to general model (384D)")

if __name__ == "__main__":
    test_thread_safety_issue()