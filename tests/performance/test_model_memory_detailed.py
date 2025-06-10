#!/usr/bin/env python3
"""Test actual memory usage with system-wide monitoring"""

import os
import sys
import gc
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def get_system_memory():
    """Get system memory usage using vm_stat on macOS"""
    try:
        # Run vm_stat command
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        
        # Parse page size
        page_size_line = lines[0]
        page_size = int(page_size_line.split()[-2])
        
        # Parse memory stats
        stats = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().replace(' ', '_')
                value = value.strip().rstrip('.')
                if value.isdigit():
                    stats[key] = int(value) * page_size / (1024**3)  # Convert to GB
        
        # Calculate used memory
        free = stats.get('Pages_free', 0)
        inactive = stats.get('Pages_inactive', 0)
        speculative = stats.get('Pages_speculative', 0)
        
        # Get total memory
        total_result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        total_gb = int(total_result.stdout.strip()) / (1024**3)
        
        available = free + inactive + speculative
        used = total_gb - available
        
        return {
            'total_gb': total_gb,
            'used_gb': used,
            'available_gb': available,
            'wired_gb': stats.get('Pages_wired_down', 0),
            'compressed_gb': stats.get('Pages_occupied_by_compressor', 0)
        }
    except Exception as e:
        print(f"Error getting system memory: {e}")
        return None

def format_system_memory(info, label):
    """Format system memory info"""
    if not info:
        return f"{label}: Unable to get system memory info"
    
    return (f"{label}:\n"
            f"  Total:      {info['total_gb']:.2f} GB\n"
            f"  Used:       {info['used_gb']:.2f} GB\n"
            f"  Available:  {info['available_gb']:.2f} GB\n"
            f"  Wired:      {info['wired_gb']:.2f} GB\n"
            f"  Compressed: {info['compressed_gb']:.2f} GB")

print("Testing CodeRankEmbed memory usage with system monitoring...\n")

# Set environment
cache_dir = os.path.expanduser("~/Documents/repos/mcp-servers/qdrant-rag/data/models")
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir

# Initial system memory
gc.collect()
time.sleep(2)
initial_sys = get_system_memory()
print(format_system_memory(initial_sys, "Initial system memory"))
print()

# Load the model on MPS
print("Loading CodeRankEmbed on MPS...")
from sentence_transformers import SentenceTransformer
import torch

before_load = get_system_memory()
start_time = time.time()

model = SentenceTransformer(
    'nomic-ai/CodeRankEmbed',
    device='mps' if torch.backends.mps.is_available() else 'cpu',
    cache_folder=cache_dir,
    trust_remote_code=True
)
model.eval()

load_time = time.time() - start_time
after_load = get_system_memory()

print(f"\nLoad time: {load_time:.2f} seconds")
print(format_system_memory(after_load, "After loading model"))

if before_load and after_load:
    memory_increase = after_load['used_gb'] - before_load['used_gb']
    print(f"\nSystem memory increase: +{memory_increase:.2f} GB")

# Test with larger batch
print("\nTesting with larger batch (simulating indexing)...")
test_texts = [
    f"def function_{i}():\n    return {i} * 2"
    for i in range(100)
]

before_encode = get_system_memory()
embeddings = model.encode(test_texts, batch_size=32, show_progress_bar=True)
after_encode = get_system_memory()

print(f"\nEmbedding shape: {embeddings.shape}")
if before_encode and after_encode:
    encode_increase = after_encode['used_gb'] - before_encode['used_gb']
    print(f"Memory increase during encoding: +{encode_increase:.2f} GB")

# Load another model to simulate model switching
print("\n\nSimulating model switching (loading jina-embeddings-v3)...")
before_second = get_system_memory()

model2 = SentenceTransformer(
    'jinaai/jina-embeddings-v3',
    device='mps' if torch.backends.mps.is_available() else 'cpu',
    cache_folder=cache_dir,
    trust_remote_code=True
)

after_second = get_system_memory()
print(format_system_memory(after_second, "After loading second model"))

if before_second and after_second:
    second_increase = after_second['used_gb'] - before_second['used_gb']
    print(f"Memory increase for second model: +{second_increase:.2f} GB")
    
    total_increase = after_second['used_gb'] - initial_sys['used_gb']
    print(f"\nTotal memory increase (both models): +{total_increase:.2f} GB")

# Cleanup
print("\nCleaning up...")
del model
del model2
gc.collect()
if torch.backends.mps.is_available():
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

time.sleep(3)
final_sys = get_system_memory()
print(format_system_memory(final_sys, "After cleanup"))

if initial_sys and final_sys:
    memory_leaked = final_sys['used_gb'] - initial_sys['used_gb']
    print(f"\nMemory not released: +{memory_leaked:.2f} GB")