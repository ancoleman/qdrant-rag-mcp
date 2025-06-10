#!/usr/bin/env python3
"""Test if thread safety fixes prevent dimension mismatch"""

import os
import sys
from pathlib import Path
import threading
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_thread_safety_fixed():
    """Test if our thread safety fixes work"""
    print("=" * 80)
    print("TESTING THREAD SAFETY FIXES")
    print("=" * 80)
    
    from utils.embeddings import get_embeddings_manager
    
    # Get embeddings manager
    embeddings_manager = get_embeddings_manager()
    
    print("\n1. Testing sequential encoding (should work)...")
    results = []
    for i in range(3):
        try:
            embedding = embeddings_manager.encode(f"def test_{i}(): pass", content_type="code")
            results.append((i, embedding.shape[-1]))
            print(f"   Test {i}: {embedding.shape[-1]}D")
        except Exception as e:
            print(f"   Test {i} failed: {e}")
    
    print("\n2. Testing concurrent encoding with thread safety...")
    concurrent_results = []
    errors = []
    lock = threading.Lock()
    
    def encode_with_thread(idx):
        try:
            embedding = embeddings_manager.encode(f"def test_{idx}(): pass", content_type="code")
            with lock:
                concurrent_results.append((idx, embedding.shape[-1]))
        except Exception as e:
            with lock:
                errors.append((idx, str(e)))
    
    # Test with just 3 threads to avoid timeout
    threads = []
    for i in range(3):
        t = threading.Thread(target=encode_with_thread, args=(i,))
        threads.append(t)
        t.start()
    
    # Wait for threads
    for t in threads:
        t.join(timeout=30)  # 30 second timeout per thread
    
    print(f"\n   Results: {len(concurrent_results)} successful, {len(errors)} failed")
    
    # Check dimensions
    if concurrent_results:
        dimensions = set(dim for _, dim in concurrent_results)
        print(f"   Dimensions: {dimensions}")
        if 384 in dimensions:
            print("   ✗ BUG STILL EXISTS: Some threads got 384D!")
        else:
            print("   ✓ All threads got correct dimensions!")
    
    if errors:
        print(f"   Errors: {errors}")
    
    print("\n3. Summary:")
    all_dimensions = set()
    all_dimensions.update(dim for _, dim in results)
    all_dimensions.update(dim for _, dim in concurrent_results)
    
    if 384 in all_dimensions:
        print("   ✗ DIMENSION MISMATCH STILL OCCURS")
    else:
        print("   ✓ THREAD SAFETY FIXES APPEAR TO BE WORKING")

if __name__ == "__main__":
    test_thread_safety_fixed()