#!/usr/bin/env python3
"""Test memory usage during search operations"""

import os
import sys
import time
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ['QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED'] = 'true'

def get_memory_usage():
    """Get current process memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)

def log_memory(context):
    """Log memory usage with context"""
    memory_gb = get_memory_usage()
    print(f"{context}: {memory_gb:.2f} GB")
    return memory_gb

# Test search memory usage
from qdrant_mcp_context_aware import search_config, search_code, search_docs

print("Testing search memory usage...\n")

# Initial memory
initial_memory = log_memory("Initial memory")

# First search - will load model
print("\n1. First config search (loading jina-embeddings-v3)...")
result = search_config("database configuration", n_results=5)
after_first_search = log_memory("After first config search")
print(f"Memory increase: +{after_first_search - initial_memory:.2f} GB")

# Second search - model already loaded
print("\n2. Second config search (model cached)...")
result = search_config("api settings", n_results=5)
after_second_search = log_memory("After second config search")
print(f"Memory increase: +{after_second_search - after_first_search:.2f} GB")

# Code search - will load another model
print("\n3. Code search (loading CodeRankEmbed)...")
result = search_code("function implementation", n_results=5)
after_code_search = log_memory("After code search")
print(f"Memory increase: +{after_code_search - after_second_search:.2f} GB")

# Check if models are evicted
print("\n4. Docs search (might evict a model)...")
result = search_docs("installation guide", n_results=5)
after_docs_search = log_memory("After docs search")
print(f"Memory increase: +{after_docs_search - after_code_search:.2f} GB")

# Back to config - check if model needs reloading
print("\n5. Config search again...")
result = search_config("yaml configuration", n_results=5)
final_memory = log_memory("Final memory")
print(f"Memory increase: +{final_memory - after_docs_search:.2f} GB")

print(f"\nTotal memory usage: {final_memory:.2f} GB")
print(f"Total increase: +{final_memory - initial_memory:.2f} GB")