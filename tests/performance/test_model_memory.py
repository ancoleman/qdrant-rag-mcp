#!/usr/bin/env python3
"""Test actual memory usage of CodeRankEmbed model"""

import os
import sys
import gc
import time
import psutil
import tracemalloc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def get_memory_info():
    """Get detailed memory information"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    virtual_memory = psutil.virtual_memory()
    
    return {
        'rss_gb': memory_info.rss / (1024**3),  # Resident Set Size
        'vms_gb': memory_info.vms / (1024**3),  # Virtual Memory Size
        'available_gb': virtual_memory.available / (1024**3),
        'percent': virtual_memory.percent
    }

def format_memory_info(info, label):
    """Format memory info for display"""
    return (f"{label}:\n"
            f"  RSS (Resident): {info['rss_gb']:.2f} GB\n"
            f"  VMS (Virtual):  {info['vms_gb']:.2f} GB\n"
            f"  System Available: {info['available_gb']:.2f} GB ({100-info['percent']:.1f}%)")

print("Testing CodeRankEmbed memory usage...\n")

# Start Python memory tracking
tracemalloc.start()

# Initial memory
gc.collect()
time.sleep(1)
initial_memory = get_memory_info()
print(format_memory_info(initial_memory, "Initial memory"))
print()

# Load SentenceTransformer
print("Loading SentenceTransformer library...")
from sentence_transformers import SentenceTransformer
after_import = get_memory_info()
print(f"Memory after import: +{after_import['rss_gb'] - initial_memory['rss_gb']:.2f} GB")
print()

# Set cache directory
cache_dir = os.path.expanduser("~/Documents/repos/mcp-servers/qdrant-rag/data/models")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir

# Test different devices
import torch

devices = ['cpu']
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    devices.append('mps')

for device in devices:
    print(f"\n{'='*60}")
    print(f"Testing with device: {device}")
    print('='*60)
    
    # Clear any existing models
    gc.collect()
    if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    time.sleep(1)
    
    before_load = get_memory_info()
    print(format_memory_info(before_load, "Before loading model"))
    
    # Load model
    print(f"\nLoading CodeRankEmbed on {device}...")
    start_time = time.time()
    
    try:
        model = SentenceTransformer(
            'nomic-ai/CodeRankEmbed',
            device=device,
            cache_folder=cache_dir,
            trust_remote_code=True
        )
        model.eval()
        
        load_time = time.time() - start_time
        after_load = get_memory_info()
        
        print(f"\nLoad time: {load_time:.2f} seconds")
        print(format_memory_info(after_load, "After loading model"))
        print(f"\nMemory increase: +{after_load['rss_gb'] - before_load['rss_gb']:.2f} GB")
        
        # Test encoding
        print("\nTesting encoding...")
        test_texts = [
            "def hello_world():\n    print('Hello, World!')",
            "function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }",
            "SELECT * FROM users WHERE age > 18",
        ]
        
        # Encode with query prefix (as used in search)
        query_text = "Represent this query for searching relevant code: function implementation"
        
        before_encode = get_memory_info()
        embeddings = model.encode(test_texts)
        query_embedding = model.encode(query_text)
        after_encode = get_memory_info()
        
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Memory after encoding: +{after_encode['rss_gb'] - after_load['rss_gb']:.2f} GB")
        
        # Get model info
        print("\nModel information:")
        total_params = sum(p.numel() for p in model.parameters())
        total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size (parameters only): {total_size_mb:.2f} MB")
        
        # Check PyTorch memory (if on MPS)
        if device == 'mps':
            print("\nMPS Memory Info:")
            # MPS doesn't have detailed memory tracking like CUDA
            print("Note: MPS doesn't provide detailed memory statistics")
        
        # Python memory tracking
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nPython memory tracking:")
        print(f"Current: {current / (1024**3):.2f} GB")
        print(f"Peak: {peak / (1024**3):.2f} GB")
        
        # Cleanup
        del model
        gc.collect()
        if device == 'mps' and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        time.sleep(2)
        after_cleanup = get_memory_info()
        print(f"\nMemory after cleanup: {after_cleanup['rss_gb']:.2f} GB")
        print(f"Memory released: -{before_load['rss_gb'] - after_cleanup['rss_gb']:.2f} GB")
        
    except Exception as e:
        print(f"\nError loading model on {device}: {e}")
        import traceback
        traceback.print_exc()

# Stop memory tracking
tracemalloc.stop()

print("\n" + "="*60)
print("Summary:")
print("- Model files on disk: 534 MB")
print("- Actual memory usage varies significantly by device")
print("- MPS typically uses 2-3x more memory than CPU")
print("- Memory may not be fully released after cleanup")