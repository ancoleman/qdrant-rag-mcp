#!/usr/bin/env python3
"""
Test script for incremental reindex functionality.

This script demonstrates the enhanced reindex_directory function
with incremental mode support.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_incremental_reindex():
    """Test the incremental reindex functionality"""
    from qdrant_mcp_context_aware import reindex_directory, detect_changes
    
    print("Testing incremental reindex functionality...")
    
    # Test with current directory (should use incremental mode by default)
    print("\n1. Testing incremental reindex (default behavior):")
    result = reindex_directory(directory=".", incremental=True)
    print(f"   Mode: {result.get('mode', 'unknown')}")
    if 'changes_detected' in result:
        changes = result['changes_detected']
        print(f"   Changes detected: {changes['added']} added, {changes['modified']} modified, {changes['deleted']} deleted, {changes['unchanged']} unchanged")
        print(f"   Files indexed: {result.get('total_indexed', 0)}")
        print(f"   Chunks deleted: {result.get('deleted_chunks', 0)}")
    elif 'total_indexed' in result:
        print(f"   Total indexed: {result['total_indexed']}")
    
    # Test with force=True (should use full reindex mode)
    print("\n2. Testing full reindex (force=True):")
    result = reindex_directory(directory=".", force=True)
    print(f"   Mode: {result.get('mode', 'unknown')}")
    if 'cleared_collections' in result:
        print(f"   Collections cleared: {len(result['cleared_collections'])}")
        print(f"   Files indexed: {result.get('total_indexed', 0)}")
    
    # Test with incremental=False (should use full reindex mode)
    print("\n3. Testing full reindex (incremental=False):")
    result = reindex_directory(directory=".", incremental=False)
    print(f"   Mode: {result.get('mode', 'unknown')}")
    if 'cleared_collections' in result:
        print(f"   Collections cleared: {len(result['cleared_collections'])}")
        print(f"   Files indexed: {result.get('total_indexed', 0)}")
    
    # Test detect_changes function separately
    print("\n4. Testing detect_changes function:")
    changes = detect_changes(".")
    if 'error' not in changes:
        summary = changes.get('summary', {})
        print(f"   Total indexed files: {summary.get('total_indexed', 0)}")
        print(f"   Total current files: {summary.get('total_current', 0)}")
        print(f"   Added: {summary.get('added_count', 0)}")
        print(f"   Modified: {summary.get('modified_count', 0)}")
        print(f"   Unchanged: {summary.get('unchanged_count', 0)}")
        print(f"   Deleted: {summary.get('deleted_count', 0)}")
    else:
        print(f"   Error: {changes['error']}")
    
    print("\nIncremental reindex test completed!")

if __name__ == "__main__":
    test_incremental_reindex()