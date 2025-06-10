#!/usr/bin/env python3
"""Debug configuration loading"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config

# Set environment variables
os.environ['QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED'] = 'true'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = 'data/models'

# Load config
config = get_config()

# Print specialized embeddings config
spec_config = config.get('specialized_embeddings', {})
print("Specialized Embeddings Config:")
print(f"  enabled: {spec_config.get('enabled')} (type: {type(spec_config.get('enabled'))})")

# Check memory config
memory_config = spec_config.get('memory', {})
print("\nMemory Config:")
print(f"  max_models_in_memory: {memory_config.get('max_models_in_memory')} (type: {type(memory_config.get('max_models_in_memory'))})")
print(f"  memory_limit_gb: {memory_config.get('memory_limit_gb')} (type: {type(memory_config.get('memory_limit_gb'))})")

# Check performance config
perf_config = spec_config.get('performance', {})
print("\nPerformance Config:")
print(f"  batch_size: {perf_config.get('batch_size')} (type: {type(perf_config.get('batch_size'))})")
print(f"  device: {perf_config.get('device')} (type: {type(perf_config.get('device'))})")

# Test embeddings manager
print("\nTesting embeddings manager creation...")
try:
    from src.utils.embeddings import get_embeddings_manager
    manager = get_embeddings_manager(config)
    print(f"  Created: {type(manager)}")
    print(f"  Has dimension property: {hasattr(manager, 'dimension')}")
    if hasattr(manager, 'dimension'):
        print(f"  Dimension: {manager.dimension}")
except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()