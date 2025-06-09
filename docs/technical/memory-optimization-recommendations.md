# Memory Optimization Recommendations for Qdrant RAG

## Quick Fixes (Environment Variables)

```bash
# Reduce number of models in memory
export QDRANT_MAX_MODELS_IN_MEMORY=1

# Reduce memory limit to force more aggressive eviction
export QDRANT_MEMORY_LIMIT_GB=8.0

# Disable progressive context to reduce caching
export QDRANT_PROGRESSIVE_CONTEXT_ENABLED=false

# Use CPU instead of MPS to avoid MPS memory issues
export QDRANT_DEVICE=cpu
```

## Code Changes for Better Memory Management

### 1. More Aggressive Memory Cleanup in index_directory

```python
# Add after each file group processing
if files_processed > 0:
    embeddings_manager = get_embeddings_manager_instance()
    if hasattr(embeddings_manager, 'use_specialized') and embeddings_manager.use_specialized:
        # Force model eviction between file types
        if hasattr(embeddings_manager, 'clear_all_models'):
            embeddings_manager.clear_all_models()
        
        # Aggressive garbage collection
        import gc
        gc.collect()
        gc.collect()  # Run twice for thorough cleanup
        
        # On macOS, also try to return memory to system
        import ctypes
        import sys
        if sys.platform == 'darwin':
            libc = ctypes.CDLL('libc.dylib')
            libc.malloc_trim(0)
```

### 2. Process Files in Smaller Batches

Instead of grouping all files by type, process in smaller batches:

```python
BATCH_SIZE = 50  # Process 50 files at a time

for group_name, file_list, index_func in file_groups:
    if not file_list:
        continue
    
    # Process in batches
    for i in range(0, len(file_list), BATCH_SIZE):
        batch = file_list[i:i + BATCH_SIZE]
        console_logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {group_name} files...")
        
        for file_path in batch:
            # Process file
            
        # Force cleanup after each batch
        if i % BATCH_SIZE == 0:
            gc.collect()
```

### 3. Fix CodeRankEmbed Memory Estimate

Update the memory estimate in specialized_embeddings.py:

```python
memory_estimates = {
    'nomic-ai/CodeRankEmbed': 8.0,  # More realistic estimate
    'jinaai/jina-embeddings-v3': 5.0,  # Also underestimated
    # ... other models
}
```

### 4. Add Model-Specific Memory Limits

```python
# In SpecializedEmbeddingManager.__init__
self.model_memory_limits = {
    'code': 8.0,  # Don't load code model if less than 8GB available
    'config': 5.0,
    'documentation': 3.0,
    'general': 1.0
}
```

### 5. Use Streaming/Chunked Processing

Instead of loading all files at once, use a generator:

```python
def get_files_to_process(directory, patterns, recursive):
    """Generator that yields files one at a time"""
    for pattern in patterns:
        glob_func = directory.rglob if recursive else directory.glob
        for file_path in glob_func(pattern):
            if should_process_file(file_path):
                yield file_path
```

## Alternative Solutions

1. **Use Docker with Memory Limits**
   ```bash
   docker run --memory=8g --memory-swap=8g qdrant-rag
   ```

2. **Use Smaller Models**
   ```bash
   export QDRANT_CODE_EMBEDDING_MODEL=microsoft/codebert-base
   export QDRANT_CONFIG_EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   ```

3. **Disable Specialized Embeddings**
   ```bash
   export QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=false
   ```

## Monitoring Memory Usage

Add memory tracking to help identify issues:

```python
import psutil
import os

def log_memory_usage(context):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Memory usage at {context}: "
                f"RSS={memory_info.rss / 1024**3:.2f}GB, "
                f"VMS={memory_info.vms / 1024**3:.2f}GB")
```

## Long-term Fixes

1. Implement model quantization to reduce model sizes
2. Use model sharding to load only necessary parts
3. Implement disk-based caching instead of memory caching
4. Use separate processes for different models (process isolation)