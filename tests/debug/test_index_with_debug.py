#!/usr/bin/env python3
"""Test indexing ast_chunker.py with debug output"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.embeddings import get_embeddings_manager
from qdrant_mcp_context_aware import index_code

def test_index_with_debug():
    """Test indexing with debug output"""
    print("=" * 80)
    print("TESTING INDEX OF ast_chunker.py")
    print("=" * 80)
    
    # Get embeddings manager to check models
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        print("\nBEFORE INDEXING:")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Try to encode a test string as code to ensure CodeRankEmbed is loaded
        print("\nPre-loading CodeRankEmbed...")
        test_embedding = embeddings_manager.encode("def test(): pass", content_type="code")
        print(f"Test embedding dimension: {test_embedding.shape[-1]}")
        
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models after pre-load: {list(actual_manager.loaded_models.keys())}")
        
        # Now try to index ast_chunker.py
        print("\nINDEXING ast_chunker.py...")
        result = index_code("/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/utils/ast_chunker.py")
        print(f"Result: {result}")
        
        print("\nAFTER INDEXING:")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")

if __name__ == "__main__":
    test_index_with_debug()