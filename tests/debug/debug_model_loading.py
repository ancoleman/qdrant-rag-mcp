#!/usr/bin/env python3
"""Debug tool to show exactly what models are loaded in SpecializedEmbeddingManager"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.specialized_embeddings import get_specialized_embedding_manager
from utils.embeddings import get_embeddings_manager
from config import get_config

def debug_model_loading():
    """Show detailed information about loaded models"""
    print("=" * 80)
    print("SPECIALIZED EMBEDDINGS DEBUG")
    print("=" * 80)
    
    # Get the embeddings manager instance
    embeddings_manager = get_embeddings_manager()
    
    print(f"\n1. Embeddings Manager Type: {type(embeddings_manager).__name__}")
    print(f"   Has 'use_specialized': {hasattr(embeddings_manager, 'use_specialized')}")
    if hasattr(embeddings_manager, 'use_specialized'):
        print(f"   use_specialized = {embeddings_manager.use_specialized}")
    
    # Check if we're using specialized embeddings
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        print(f"\n2. Actual Manager Type: {type(actual_manager).__name__}")
        
        if hasattr(actual_manager, 'loaded_models'):
            print(f"\n3. Loaded Models:")
            if actual_manager.loaded_models:
                for model_name, model in actual_manager.loaded_models.items():
                    print(f"   - {model_name}")
                    print(f"     Type: {type(model).__name__}")
                    print(f"     Device: {getattr(model, 'device', 'unknown')}")
                    if hasattr(actual_manager, 'memory_usage') and model_name in actual_manager.memory_usage:
                        print(f"     Memory: {actual_manager.memory_usage[model_name]:.2f} GB")
            else:
                print("   No models currently loaded")
        
        if hasattr(actual_manager, 'model_configs'):
            print(f"\n4. Model Configurations:")
            for content_type, config in actual_manager.model_configs.items():
                print(f"   {content_type}:")
                print(f"     Model: {config.get('name', 'unknown')}")
                print(f"     Dimension: {config.get('dimension', 'unknown')}")
                if 'fallback' in config:
                    print(f"     Fallback: {config['fallback']}")
        
        if hasattr(actual_manager, 'active_models'):
            print(f"\n5. Active Models: {actual_manager.active_models}")
        
        if hasattr(actual_manager, 'usage_stats'):
            print(f"\n6. Usage Statistics:")
            for content_type, stats in actual_manager.usage_stats.items():
                if stats.get('loads', 0) > 0 or stats.get('encodes', 0) > 0:
                    print(f"   {content_type}: loads={stats.get('loads', 0)}, encodes={stats.get('encodes', 0)}")
        
        # Test encoding with different content types
        print(f"\n7. Testing Encoding:")
        test_text = "def test_function(): pass"
        
        for content_type in ['code', 'config', 'documentation', 'general']:
            try:
                print(f"\n   Testing {content_type}:")
                # Get model name before encoding
                if hasattr(actual_manager, 'get_model_name'):
                    model_name = actual_manager.get_model_name(content_type)
                    print(f"     Will use model: {model_name}")
                
                # Encode
                embedding = embeddings_manager.encode(test_text, content_type=content_type)
                print(f"     Embedding shape: {embedding.shape}")
                print(f"     Dimension: {embedding.shape[-1]}")
                
                # Check loaded models after encoding
                if hasattr(actual_manager, 'loaded_models'):
                    print(f"     Models loaded after encoding: {list(actual_manager.loaded_models.keys())}")
                    
            except Exception as e:
                print(f"     ERROR: {e}")
    
    # Check environment variables
    print(f"\n8. Environment Variables:")
    env_vars = [
        "QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED",
        "QDRANT_CODE_EMBEDDING_MODEL",
        "QDRANT_MAX_MODELS_IN_MEMORY",
        "QDRANT_MEMORY_LIMIT_GB"
    ]
    for var in env_vars:
        value = os.getenv(var, "NOT SET")
        print(f"   {var}: {value}")
    
    # Check system memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n9. System Memory:")
        print(f"   Available: {memory.available / (1024**3):.1f} GB")
        print(f"   Total: {memory.total / (1024**3):.1f} GB")
        print(f"   Used %: {memory.percent}%")
    except ImportError:
        print("\n9. System Memory: psutil not available")

if __name__ == "__main__":
    debug_model_loading()