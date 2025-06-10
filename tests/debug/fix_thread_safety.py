#!/usr/bin/env python3
"""Test fix for thread safety in specialized embeddings"""

import os
import sys
from pathlib import Path
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_thread_safety_fix():
    """Test if adding a lock fixes the dimension issue"""
    print("=" * 80)
    print("TESTING THREAD SAFETY FIX")
    print("=" * 80)
    
    from utils.specialized_embeddings import SpecializedEmbeddingManager
    
    # Check current implementation
    print("\n1. Checking current SpecializedEmbeddingManager for thread safety...")
    
    # Look for threading primitives
    manager = SpecializedEmbeddingManager()
    
    has_lock = hasattr(manager, '_lock') or hasattr(manager, 'lock')
    print(f"   Has lock: {has_lock}")
    
    # Check if encode method is thread-safe
    import inspect
    encode_source = inspect.getsource(manager.encode)
    has_thread_safety = 'lock' in encode_source.lower() or 'Lock' in encode_source
    print(f"   Encode method has thread safety: {has_thread_safety}")
    
    # Check load_model method
    load_model_source = inspect.getsource(manager.load_model)
    load_has_thread_safety = 'lock' in load_model_source.lower() or 'Lock' in load_model_source
    print(f"   Load model method has thread safety: {load_has_thread_safety}")
    
    print("\n2. The issue is that SpecializedEmbeddingManager is NOT thread-safe!")
    print("   Multiple threads can cause race conditions during model loading/eviction.")
    
    print("\n3. Proposed fix locations in specialized_embeddings.py:")
    print("   a) Add threading.RLock() in __init__ method")
    print("   b) Use lock in load_model() method")
    print("   c) Use lock in encode() method")
    print("   d) Use lock in _evict_lru_model() method")
    
    print("\n4. Example fix pattern:")
    print("""
    def __init__(self, ...):
        ...
        # Add thread safety
        self._lock = threading.RLock()
        ...
    
    def load_model(self, content_type: str) -> Tuple[SentenceTransformer, Dict[str, Any]]:
        with self._lock:
            # Existing load_model code
            ...
    
    def encode(self, texts: Union[str, List[str]], ...):
        # Only lock the critical sections
        model, model_config = self.load_model(content_type)  # This already has lock
        model_name = model_config['name']
        
        with self._lock:
            # Mark model as active
            self.active_models.add(model_name)
        
        try:
            # Actual encoding doesn't need lock
            ...
        finally:
            with self._lock:
                # Remove from active set
                self.active_models.discard(model_name)
    """)
    
    print("\n5. Alternative solutions:")
    print("   a) Use a process pool instead of thread pool for parallelism")
    print("   b) Create separate embeddings manager instances per thread")
    print("   c) Pre-load all models and disable eviction during batch operations")
    
    # Test if the issue is specifically with active_models set
    print("\n6. Analyzing the specific race condition...")
    print("   The issue likely occurs when:")
    print("   - Thread 1: Loads CodeRankEmbed for Python file")
    print("   - Thread 2: Loads config model, evicts CodeRankEmbed")
    print("   - Thread 1: Tries to use CodeRankEmbed but it's gone")
    print("   - Thread 1: Falls back to general model (384D)")
    
    # Check if there's a fallback mechanism that could cause 384D
    print("\n7. Checking fallback behavior...")
    from utils.embeddings import get_embeddings_manager
    embeddings_manager = get_embeddings_manager()
    
    if hasattr(embeddings_manager, 'manager'):
        actual_manager = embeddings_manager.manager
        if hasattr(actual_manager, 'model_configs'):
            code_config = actual_manager.model_configs.get('code', {})
            print(f"   Code model: {code_config.get('name')}")
            print(f"   Code dimension: {code_config.get('dimension')}")
            print(f"   Code fallback: {code_config.get('fallback')}")
            
            # Check fallback dimensions
            if code_config.get('fallback'):
                fallback_name = code_config['fallback']
                # Check if fallback is 384D
                if 'codebert' in fallback_name.lower():
                    print(f"   Fallback dimension: ~768D (CodeBERT)")
                else:
                    print(f"   Fallback dimension: Unknown")
            
            # Check general model
            general_config = actual_manager.model_configs.get('general', {})
            print(f"\n   General model: {general_config.get('name')}")
            print(f"   General dimension: {general_config.get('dimension')}")
            
            if general_config.get('dimension') == 384:
                print("\n   FOUND IT! The general model is 384D.")
                print("   If code model fails to load, it falls back to general (384D)!")

if __name__ == "__main__":
    test_thread_safety_fix()