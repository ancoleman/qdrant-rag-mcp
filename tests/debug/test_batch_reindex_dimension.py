#!/usr/bin/env python3
"""Test to reproduce and debug the dimension mismatch during batch reindexing"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_batch_reindex_dimension():
    """Test batch reindexing with dimension tracking"""
    print("=" * 80)
    print("TESTING BATCH REINDEX DIMENSION ISSUE")
    print("=" * 80)
    
    # Import after adding to path
    from utils.embeddings import get_embeddings_manager
    from qdrant_mcp_context_aware import index_code, index_config, index_documentation
    
    # Get embeddings manager
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        
        print("\nINITIAL STATE:")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Simulate batch reindexing scenario
        print("\n1. Simulating batch reindex with multiple file types...")
        
        # First, index a config file (should load 1024D model)
        print("\n   Indexing config file...")
        config_embedding = embeddings_manager.encode("test config", content_type="config")
        print(f"   Config embedding dimension: {config_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Then index a doc file (should load 768D model)
        print("\n   Indexing documentation file...")
        doc_embedding = embeddings_manager.encode("test doc", content_type="documentation")
        print(f"   Doc embedding dimension: {doc_embedding.shape[-1]}")
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Loaded models: {list(actual_manager.loaded_models.keys())}")
            print(f"   Max models allowed: {actual_manager.max_models_in_memory}")
        
        # Now try to index a Python file - this should load CodeRankEmbed (768D)
        # But if it's already evicted, what happens?
        print("\n2. Now trying to encode Python code...")
        
        # Check what model will be used
        if hasattr(actual_manager, 'get_model_name'):
            code_model = actual_manager.get_model_name('code')
            print(f"   Model for 'code' content type: {code_model}")
        
        # Check current loaded models
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Currently loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Try encoding
        code_test = "def test_function():\n    pass"
        code_embedding = embeddings_manager.encode(code_test, content_type="code")
        print(f"   Code embedding dimension: {code_embedding.shape[-1]}")
        
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Loaded models after code encoding: {list(actual_manager.loaded_models.keys())}")
        
        # Let's check if there's a fallback happening
        print("\n3. Checking for fallback behavior...")
        
        # Look at the code model config
        if hasattr(actual_manager, 'model_configs'):
            code_config = actual_manager.model_configs.get('code', {})
            print(f"   Code model config:")
            print(f"     Primary: {code_config.get('name')}")
            print(f"     Dimension: {code_config.get('dimension')}")
            print(f"     Fallback: {code_config.get('fallback')}")
        
        # Now let's test what happens if we force eviction and then try again
        print("\n4. Testing after forced eviction...")
        
        # Load general model to force eviction
        general_embedding = embeddings_manager.encode("test general", content_type="general")
        print(f"   General embedding dimension: {general_embedding.shape[-1]}")
        
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Loaded models after general: {list(actual_manager.loaded_models.keys())}")
        
        # Try code again
        code_embedding2 = embeddings_manager.encode(code_test, content_type="code")
        print(f"   Code embedding after eviction: {code_embedding2.shape[-1]}")
        
        if hasattr(actual_manager, 'loaded_models'):
            print(f"   Final loaded models: {list(actual_manager.loaded_models.keys())}")
        
        # Check if dimensions match what we expect
        print("\n5. DIMENSION VALIDATION:")
        expected_dims = {
            'code': 768,  # CodeRankEmbed
            'config': 1024,  # Jina v3
            'documentation': 768,  # Instructor
            'general': 384  # MiniLM
        }
        
        print("\n   Testing all content types again:")
        for content_type in ['code', 'config', 'documentation', 'general']:
            embedding = embeddings_manager.encode("test", content_type=content_type)
            actual_dim = embedding.shape[-1]
            expected_dim = expected_dims[content_type]
            status = "✓" if actual_dim == expected_dim else "✗ MISMATCH!"
            print(f"   {content_type}: {actual_dim}D (expected {expected_dim}D) {status}")
            
            if actual_dim != expected_dim:
                # This is the issue!
                print(f"     WARNING: Dimension mismatch for {content_type}!")
                if hasattr(actual_manager, 'get_model_name'):
                    model_used = actual_manager.get_model_name(content_type)
                    print(f"     Model configured: {model_used}")
                    if hasattr(actual_manager, 'loaded_models') and model_used in actual_manager.loaded_models:
                        print(f"     Model IS loaded")
                    else:
                        print(f"     Model NOT loaded - may be using fallback!")
        
        # Test actual file indexing
        print("\n6. Testing actual file indexing...")
        test_files = [
            "/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/utils/ast_chunker.py",
            "/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag/src/utils/progressive_context.py"
        ]
        
        for file_path in test_files:
            if Path(file_path).exists():
                print(f"\n   Testing {Path(file_path).name}...")
                try:
                    # First check what dimension we'd get
                    with open(file_path, 'r') as f:
                        content = f.read()[:500]  # Just test with first 500 chars
                    
                    test_embedding = embeddings_manager.encode(content, content_type="code")
                    print(f"     Test embedding dimension: {test_embedding.shape[-1]}")
                    
                    if test_embedding.shape[-1] != 768:
                        print(f"     ERROR: Would generate {test_embedding.shape[-1]}D instead of 768D!")
                        print(f"     This would cause the dimension mismatch error!")
                except Exception as e:
                    print(f"     Error testing file: {e}")

if __name__ == "__main__":
    test_batch_reindex_dimension()