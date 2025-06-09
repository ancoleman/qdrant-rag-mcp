# Memory Optimization Fixes for Progressive Context and Caching

## Problem Analysis

The MCP server's memory grows exponentially due to:

1. **Progressive Context Cache**:
   - Stores up to 1000 query results
   - Embeddings cache stores up to 2000 numpy arrays
   - No memory-based eviction (only count-based)
   - Expired entries not cleaned up

2. **Context Tracker**:
   - Stores full file contents in `files_read`
   - Stores all search results
   - Timeline keeps growing indefinitely
   - Session data persists in memory

3. **Model Management**:
   - Models correctly limited to 2, but estimates were too high
   - No periodic garbage collection

## Immediate Fixes

### 1. Fix Progressive Context Memory Leaks

```python
# In src/utils/progressive_context.py

class SemanticCache:
    def __init__(self, config: Optional[Dict[str, Any]] = None, embedding_model=None):
        # Add memory limits
        self.max_memory_mb = self.config.get("max_memory_mb", 500)  # 500MB max
        self.cleanup_interval = self.config.get("cleanup_interval", 100)  # Clean every 100 operations
        self.operation_count = 0
        
    def _estimate_memory_usage(self) -> float:
        """Estimate current cache memory usage in MB"""
        import sys
        
        # Estimate cache memory
        cache_size = sum(sys.getsizeof(v) for v in self.cache.values()) / (1024**2)
        
        # Estimate embeddings memory (768 or 1024 dims * 4 bytes * count)
        avg_embedding_size = 768 * 4 / (1024**2)  # ~3KB per embedding
        embeddings_size = len(self.embeddings_cache) * avg_embedding_size
        
        return cache_size + embeddings_size
    
    def _cleanup_if_needed(self):
        """Cleanup cache if memory limit exceeded or on interval"""
        self.operation_count += 1
        
        # Periodic cleanup
        if self.operation_count % self.cleanup_interval == 0:
            self._cleanup_expired()
        
        # Memory-based cleanup
        if self._estimate_memory_usage() > self.max_memory_mb:
            self._aggressive_cleanup()
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (result, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _aggressive_cleanup(self):
        """Aggressively clean cache when memory limit exceeded"""
        # Remove oldest 50% of cache
        if len(self.cache) > 10:
            items_to_remove = len(self.cache) // 2
            for _ in range(items_to_remove):
                self.cache.popitem(last=False)
        
        # Clear half of embeddings cache
        if len(self.embeddings_cache) > 20:
            items_to_remove = len(self.embeddings_cache) // 2
            keys_to_remove = list(self.embeddings_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.embeddings_cache[key]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.logger.warning(f"Aggressive cleanup performed. Memory usage was above {self.max_memory_mb}MB")
    
    def get_similar(self, query: str, level: str) -> Optional[ProgressiveResult]:
        """Get similar result with cleanup"""
        self._cleanup_if_needed()
        # ... rest of the method
    
    def add(self, query: str, level: str, result: ProgressiveResult):
        """Add result with cleanup"""
        self._cleanup_if_needed()
        # ... rest of the method
```

### 2. Fix Context Tracker Memory Usage

```python
# In src/utils/context_tracking.py

class ContextTracker:
    def __init__(self, session_id: Optional[str] = None):
        # Add limits
        self.max_files_tracked = 100
        self.max_timeline_events = 500
        self.max_content_length = 1000  # Store only first 1000 chars
        
    def track_file_read(self, file_path: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Track file read with content truncation"""
        # Truncate content for storage
        truncated_content = content[:self.max_content_length] + "..." if len(content) > self.max_content_length else content
        
        # Limit files tracked
        if len(self.files_read) >= self.max_files_tracked:
            # Remove oldest file
            oldest_key = next(iter(self.files_read))
            del self.files_read[oldest_key]
        
        # ... rest of the method
    
    def track_search(self, query: str, results: List[Dict[str, Any]], search_type: str = "general"):
        """Track search with result truncation"""
        # Don't store full results, just metadata
        result_metadata = [
            {
                "file_path": r.get("file_path", ""),
                "score": r.get("score", 0),
                "type": r.get("type", "")
            }
            for r in results[:10]  # Only track top 10
        ]
        
        # ... rest of the method
    
    def _cleanup_timeline(self):
        """Keep timeline under control"""
        if len(self.timeline) > self.max_timeline_events:
            # Keep only recent events
            self.timeline = self.timeline[-self.max_timeline_events:]
```

### 3. Add Periodic Garbage Collection

```python
# In src/qdrant_mcp_context_aware.py

# Add at the top
import gc
import threading
import time

class MemoryManager:
    """Manages periodic memory cleanup"""
    
    def __init__(self, interval_seconds: int = 300):  # 5 minutes
        self.interval = interval_seconds
        self.running = False
        self.thread = None
        
    def start(self):
        """Start periodic cleanup"""
        self.running = True
        self.thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop periodic cleanup"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _cleanup_loop(self):
        """Periodic cleanup loop"""
        while self.running:
            time.sleep(self.interval)
            if self.running:
                self._perform_cleanup()
    
    def _perform_cleanup(self):
        """Perform memory cleanup"""
        logger = get_logger()
        logger.debug("Performing periodic memory cleanup")
        
        # Force garbage collection
        gc.collect()
        gc.collect()  # Run twice for thorough cleanup
        
        # Log memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            logger.info(f"Memory usage after cleanup: {memory_mb:.1f} MB")
        except:
            pass

# Initialize memory manager
memory_manager = MemoryManager()
memory_manager.start()

# Add cleanup on exit
import atexit
atexit.register(memory_manager.stop)
```

### 4. Configuration Updates

Add to `server_config.json`:

```json
{
  "progressive_context": {
    "semantic_cache": {
      "max_cache_size": 100,  // Reduce from 1000
      "max_memory_mb": 200,   // 200MB limit
      "cleanup_interval": 50, // Clean every 50 operations
      "ttl_seconds": 1800     // 30 minutes instead of 1 hour
    }
  },
  "context_tracking": {
    "max_files_tracked": 50,
    "max_timeline_events": 200,
    "max_content_length": 500
  },
  "memory_management": {
    "cleanup_interval_seconds": 180,  // 3 minutes
    "aggressive_cleanup_threshold_mb": 8000  // 8GB
  }
}
```

## Long-term Solutions

1. **Use Disk-Based Caching**:
   - Store embeddings in a local SQLite database
   - Use memory-mapped files for large data
   - Implement LRU disk cache

2. **Implement Streaming**:
   - Don't load entire file contents into memory
   - Stream search results instead of storing all

3. **Process Isolation**:
   - Run each model in a separate process
   - Use IPC for communication
   - Allows OS to manage memory better

4. **Better Monitoring**:
   - Add memory usage to health_check endpoint
   - Log memory usage periodically
   - Alert when memory exceeds threshold