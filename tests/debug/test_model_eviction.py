#!/usr/bin/env python3
"""Test model eviction scenario"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.embeddings import get_embeddings_manager
from qdrant_mcp_context_aware import index_code

def test_model_eviction():
    """Test what happens when CodeRankEmbed gets evicted"""
    print("=" * 80)
    print("TESTING MODEL EVICTION SCENARIO")
    print("=" * 80)
    
    # Get embeddings manager
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        print("\nInitial state:")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Load config model first
        print("\n1. Loading config model...")
        config_embedding = embeddings_manager.encode("test config", content_type="config")
        print(f"Config embedding dimension: {config_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Load documentation model
        print("\n2. Loading documentation model...")
        doc_embedding = embeddings_manager.encode("test doc", content_type="documentation")
        print(f"Doc embedding dimension: {doc_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Load general model (this should evict one of the others)
        print("\n3. Loading general model...")
        general_embedding = embeddings_manager.encode("test general", content_type="general")
        print(f"General embedding dimension: {general_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
            print(f"Max models allowed: {actual_manager.max_models_in_memory}")
        
        # Now try to encode as code - this should load CodeRankEmbed
        print("\n4. Encoding as code (should load CodeRankEmbed)...")
        code_embedding = embeddings_manager.encode("def test(): pass", content_type="code")
        print(f"Code embedding dimension: {code_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Check what model is actually being used for code
        print("\n5. Checking which model is used for code content type...")
        if hasattr(actual_manager, 'get_model_name'):
            code_model = actual_manager.get_model_name('code')
            print(f"Model for 'code' content type: {code_model}")
        
        # Now let's see what happens when we try to index ast_chunker.py
        print("\n6. Attempting to index ast_chunker.py...")
        try:
            # First, make sure the old chunks are deleted
            from qdrant_mcp_context_aware import delete_file_chunks
            delete_result = delete_file_chunks("/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/utils/ast_chunker.py")
            print(f"Deleted {delete_result.get('deleted_points', 0)} old chunks")
            
            # Now index
            result = index_code("/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/utils/ast_chunker.py")
            print(f"Index result: {result}")
        except Exception as e:
            print(f"ERROR: {e}")
        
        print("\n7. Final state:")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")

if __name__ == "__main__":
    test_model_eviction()