#!/usr/bin/env python3
"""Test to check if there's a concurrency issue causing dimension mismatch"""

import os
import sys
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def encode_file(file_info):
    """Encode a file and return its dimension"""
    file_path, content_type, expected_dim = file_info
    
    from utils.embeddings import get_embeddings_manager
    embeddings_manager = get_embeddings_manager()
    
    try:
        # Simulate reading file content
        test_content = f"Test content for {content_type} file: {file_path}"
        
        # Encode
        embedding = embeddings_manager.encode(test_content, content_type=content_type)
        actual_dim = embedding.shape[-1]
        
        success = actual_dim == expected_dim
        
        return {
            "file": file_path,
            "content_type": content_type,
            "expected_dim": expected_dim,
            "actual_dim": actual_dim,
            "success": success,
            "thread_id": threading.get_ident()
        }
    except Exception as e:
        return {
            "file": file_path,
            "content_type": content_type,
            "error": str(e),
            "thread_id": threading.get_ident()
        }

def test_concurrent_encoding():
    """Test concurrent encoding to see if there's a race condition"""
    print("=" * 80)
    print("TESTING CONCURRENT ENCODING FOR DIMENSION ISSUES")
    print("=" * 80)
    
    # Simulate files from a reindex operation
    test_files = [
        # Mix of different file types like in a real reindex
        ("src/config/settings.json", "config", 1024),
        ("src/utils/ast_chunker.py", "code", 768),
        ("docs/README.md", "documentation", 768),
        ("src/utils/embeddings.py", "code", 768),
        ("config/server_config.json", "config", 1024),
        ("src/utils/progressive_context.py", "code", 768),
        ("src/main.py", "code", 768),
        ("package.json", "config", 1024),
        ("docs/api.md", "documentation", 768),
        ("src/utils/memory_manager.py", "code", 768),
        ("src/test.py", "code", 768),
        (".env", "config", 1024),
    ]
    
    print(f"\nTesting with {len(test_files)} files...")
    
    # Test 1: Sequential processing (like the current implementation)
    print("\n1. SEQUENTIAL PROCESSING:")
    sequential_results = []
    start_time = time.time()
    
    from utils.embeddings import get_embeddings_manager
    embeddings_manager = get_embeddings_manager()
    
    for file_info in test_files:
        result = encode_file(file_info)
        sequential_results.append(result)
        if not result.get("success", False):
            print(f"   FAILED: {result['file']} - {result['content_type']} - "
                  f"Expected {result.get('expected_dim')}D, got {result.get('actual_dim')}D")
    
    sequential_time = time.time() - start_time
    sequential_failures = sum(1 for r in sequential_results if not r.get("success", False))
    print(f"   Completed in {sequential_time:.2f}s - {sequential_failures} failures")
    
    # Test 2: Concurrent processing (potential future optimization)
    print("\n2. CONCURRENT PROCESSING (with ThreadPoolExecutor):")
    concurrent_results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(encode_file, file_info) for file_info in test_files]
        
        for future in as_completed(futures):
            result = future.result()
            concurrent_results.append(result)
            if not result.get("success", False):
                print(f"   FAILED: {result['file']} - {result['content_type']} - "
                      f"Expected {result.get('expected_dim')}D, got {result.get('actual_dim')}D")
    
    concurrent_time = time.time() - start_time
    concurrent_failures = sum(1 for r in concurrent_results if not r.get("success", False))
    print(f"   Completed in {concurrent_time:.2f}s - {concurrent_failures} failures")
    
    # Test 3: Rapid sequential with model eviction stress test
    print("\n3. RAPID SEQUENTIAL (stress testing model eviction):")
    stress_results = []
    start_time = time.time()
    
    # Double the test files to stress the system
    stress_files = test_files * 2
    
    for file_info in stress_files:
        result = encode_file(file_info)
        stress_results.append(result)
        if not result.get("success", False):
            print(f"   FAILED: {result['file']} - {result['content_type']} - "
                  f"Expected {result.get('expected_dim')}D, got {result.get('actual_dim')}D")
    
    stress_time = time.time() - start_time
    stress_failures = sum(1 for r in stress_results if not r.get("success", False))
    print(f"   Completed in {stress_time:.2f}s - {stress_failures} failures")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print(f"Sequential: {sequential_failures}/{len(sequential_results)} failures")
    print(f"Concurrent: {concurrent_failures}/{len(concurrent_results)} failures")
    print(f"Stress:     {stress_failures}/{len(stress_results)} failures")
    
    # Check if any Python files got wrong dimensions
    all_results = sequential_results + concurrent_results + stress_results
    python_failures = [r for r in all_results 
                      if r.get("content_type") == "code" 
                      and r.get("actual_dim") != 768]
    
    if python_failures:
        print("\nPYTHON FILE DIMENSION FAILURES:")
        for failure in python_failures:
            print(f"  {failure['file']}: {failure.get('actual_dim')}D instead of 768D")
        print("\nTHIS IS THE BUG! Python files are getting wrong dimensions!")
    else:
        print("\nNo dimension failures detected in this test run.")
        print("The issue might be more complex or require specific timing/memory conditions.")

if __name__ == "__main__":
    test_concurrent_encoding()