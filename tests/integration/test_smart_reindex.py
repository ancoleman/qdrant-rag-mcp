#!/usr/bin/env python3
"""
Test Smart Reindex functionality (v0.2.4)

This test demonstrates the new incremental reindexing feature.
"""

import os
import tempfile
import shutil
from pathlib import Path
import requests
import json
import hashlib
import time

# Test configuration
BASE_URL = "http://localhost:8081"
TEST_PROJECT_NAME = "smart_reindex_test"

def create_test_file(directory: Path, filename: str, content: str) -> str:
    """Create a test file and return its path."""
    file_path = directory / filename
    file_path.write_text(content)
    return str(file_path)

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def test_smart_reindex():
    """Test the smart incremental reindexing feature."""
    print("🧪 Testing Smart Reindex (v0.2.4)")
    print("=" * 50)
    
    # Create temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / TEST_PROJECT_NAME
        test_dir.mkdir()
        
        print(f"📁 Created test directory: {test_dir}")
        
        # === Phase 1: Initial Index ===
        print("\n🚀 Phase 1: Initial Index")
        
        # Create initial files
        file1 = create_test_file(test_dir, "app.py", """
def hello_world():
    return "Hello, World!"

def add_numbers(a, b):
    return a + b
""")
        
        file2 = create_test_file(test_dir, "config.json", """{
    "database": {
        "host": "localhost",
        "port": 5432
    }
}""")
        
        file3 = create_test_file(test_dir, "README.md", """# Test Project

This is a test project for smart reindexing.

## Features
- Smart change detection
- Incremental updates
""")
        
        # Index the directory initially
        print("📊 Indexing initial files...")
        response = requests.post(f"{BASE_URL}/index_directory", json={
            "directory": str(test_dir),
            "recursive": True
        })
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Initial index completed: {result.get('total', 0)} files")
        else:
            print(f"❌ Initial index failed: {response.text}")
            return
        
        # Store initial hashes
        initial_hashes = {
            file1: calculate_file_hash(file1),
            file2: calculate_file_hash(file2),
            file3: calculate_file_hash(file3)
        }
        
        time.sleep(1)  # Ensure timestamp difference
        
        # === Phase 2: No Changes Reindex ===
        print("\n🔄 Phase 2: Reindex with No Changes")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/reindex_directory", json={
            "directory": str(test_dir),
            "incremental": True
        })
        no_change_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ No-change reindex: {result.get('message', 'completed')}")
            print(f"⚡ Duration: {no_change_duration:.2f}s")
            changes = result.get('changes_detected', {})
            print(f"📈 Changes: {changes.get('unchanged', 0)} unchanged, {changes.get('added', 0)} added, {changes.get('modified', 0)} modified, {changes.get('deleted', 0)} deleted")
        else:
            print(f"❌ No-change reindex failed: {response.text}")
        
        # === Phase 3: Modify Files ===
        print("\n✏️  Phase 3: Modify Files and Reindex")
        
        # Modify file1
        create_test_file(test_dir, "app.py", """
def hello_world():
    return "Hello, Smart World!"  # Modified

def add_numbers(a, b):
    return a + b

def multiply_numbers(a, b):  # New function
    return a * b
""")
        
        # Add new file
        file4 = create_test_file(test_dir, "utils.py", """
def format_string(s):
    return s.upper()
""")
        
        # Delete file2
        os.remove(file2)
        
        # Smart reindex
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/reindex_directory", json={
            "directory": str(test_dir),
            "incremental": True
        })
        smart_reindex_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Smart reindex: {result.get('message', 'completed')}")
            print(f"⚡ Duration: {smart_reindex_duration:.2f}s")
            changes = result.get('changes_detected', {})
            print(f"📈 Changes: {changes.get('unchanged', 0)} unchanged, {changes.get('added', 0)} added, {changes.get('modified', 0)} modified, {changes.get('deleted', 0)} deleted")
            print(f"🗂️  Indexed files: {result.get('total_indexed', 0)}")
            print(f"🗑️  Deleted chunks: {result.get('deleted_chunks', 0)}")
        else:
            print(f"❌ Smart reindex failed: {response.text}")
        
        # === Phase 4: Full Reindex Comparison ===
        print("\n🔄 Phase 4: Full Reindex for Comparison")
        
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/reindex_directory", json={
            "directory": str(test_dir),
            "force": True
        })
        full_reindex_duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Full reindex: {result.get('message', 'completed')}")
            print(f"⚡ Duration: {full_reindex_duration:.2f}s")
            print(f"🗂️  Total indexed: {result.get('total_indexed', 0)}")
        else:
            print(f"❌ Full reindex failed: {response.text}")
        
        # === Summary ===
        print("\n📊 Performance Summary")
        print("=" * 30)
        print(f"No-change reindex:  {no_change_duration:.2f}s")
        print(f"Smart reindex:      {smart_reindex_duration:.2f}s")
        print(f"Full reindex:       {full_reindex_duration:.2f}s")
        
        if smart_reindex_duration < full_reindex_duration:
            speedup = full_reindex_duration / smart_reindex_duration
            print(f"🚀 Smart reindex is {speedup:.1f}x faster than full reindex!")
        
        print("\n✅ Smart Reindex test completed successfully!")

def test_search_after_reindex():
    """Test that search works correctly after smart reindex."""
    print("\n🔍 Testing Search After Smart Reindex")
    
    # Search for content that should exist
    response = requests.post(f"{BASE_URL}/search", json={
        "query": "multiply_numbers",
        "n_results": 3
    })
    
    if response.status_code == 200:
        results = response.json()
        if results.get('results'):
            print(f"✅ Search found {len(results['results'])} results for 'multiply_numbers'")
            for i, result in enumerate(results['results'][:2]):
                print(f"   {i+1}. {result.get('file_path', 'unknown')} (score: {result.get('score', 0):.3f})")
        else:
            print("❌ Search found no results for 'multiply_numbers'")
    else:
        print(f"❌ Search failed: {response.text}")

if __name__ == "__main__":
    print("Smart Reindex Test Suite")
    print("Ensure the HTTP server is running on localhost:8081")
    print()
    
    # Test health first
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please start the HTTP server with: python src/http_server.py")
        exit(1)
    
    # Run tests
    test_smart_reindex()
    test_search_after_reindex()
    
    print("\n🎉 All tests completed!")