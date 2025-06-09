# Unified Memory Management System

## Overview

The Qdrant RAG MCP server v0.3.3 introduces a unified memory management system that centralizes memory tracking and cleanup across all components. This prevents the exponential memory growth issue where Python processes could consume 15-16GB+ during extended sessions.

## Architecture

### Core Components

1. **UnifiedMemoryManager** (`src/utils/memory_manager.py`)
   - Central coordinator for all memory management
   - Runs periodic cleanup and garbage collection threads
   - Monitors system and component memory usage
   - Triggers cleanup based on configurable thresholds

2. **MemoryComponent** (Base Class)
   - Abstract base for all memory-consuming components
   - Provides standard interface for memory tracking
   - Implements cleanup methods

3. **MemoryRegistry**
   - Tracks all registered components
   - Aggregates memory usage statistics
   - Coordinates cleanup operations

### Integrated Components

1. **SpecializedEmbeddingManager** 
   - Manages multiple embedding models with LRU eviction
   - Memory limit: 4GB default (configurable via `QDRANT_EMBEDDINGS_MAX_MEMORY_MB`)
   - Max models in memory: 3 (configurable via `QDRANT_EMBEDDINGS_MAX_MODELS`)
   - Implements its own sophisticated memory management:
     - Model-specific memory estimation
     - System memory checking with psutil
     - GPU/MPS cache clearing on eviction
     - Device-specific cleanup
   - Actual measured usage:
     - CodeRankEmbed: ~1GB on MPS
     - jina-embeddings-v3: ~1.5GB on MPS
     - instructor-large: ~1.2GB on MPS

2. **SemanticCache** (Progressive Context)
   - Extends LRUMemoryCache for semantic similarity caching
   - Memory limit: 200MB default (configurable via `QDRANT_PROGRESSIVE_CACHE_MAX_MEMORY_MB`)
   - Max items: 100 cached results (configurable via `QDRANT_PROGRESSIVE_CACHE_MAX_ITEMS`)
   - Additional features:
     - TTL expiration (30 minutes default)
     - Embeddings cache management
     - Semantic similarity matching
   - Inherits LRU eviction from parent class

3. **SessionContextTracker**
   - Extends MemoryComponent for context tracking
   - Memory limit: 100MB default (configurable via `QDRANT_CONTEXT_TRACKING_MAX_MEMORY_MB`)
   - Max files: 100 (configurable via `QDRANT_CONTEXT_TRACKING_MAX_FILES`)
   - Max timeline events: 500 (configurable via `QDRANT_CONTEXT_TRACKING_MAX_EVENTS`)
   - Automatic trimming of old events when limits reached

## Configuration

### Environment Variables

```bash
# Global Memory Management
QDRANT_MEMORY_MANAGEMENT_ENABLED=true
QDRANT_TOTAL_MEMORY_LIMIT_MB=8000      # 8GB total limit
QDRANT_CLEANUP_THRESHOLD_MB=6000       # Start cleanup at 6GB
QDRANT_AGGRESSIVE_THRESHOLD_MB=7000    # Aggressive cleanup at 7GB
QDRANT_CLEANUP_INTERVAL_SECONDS=180    # Check every 3 minutes
QDRANT_GC_INTERVAL_SECONDS=300         # GC every 5 minutes

# Component Limits
QDRANT_EMBEDDINGS_MAX_MEMORY_MB=4000   # 4GB for models
QDRANT_EMBEDDINGS_MAX_MODELS=3         # Max 3 models loaded
QDRANT_PROGRESSIVE_CACHE_MAX_MEMORY_MB=200
QDRANT_PROGRESSIVE_CACHE_MAX_ITEMS=100
QDRANT_CONTEXT_TRACKING_MAX_MEMORY_MB=100
QDRANT_CONTEXT_TRACKING_MAX_FILES=100
QDRANT_CONTEXT_TRACKING_MAX_EVENTS=500
```

### Server Config (server_config.json)

```json
{
  "memory_management": {
    "enabled": true,
    "total_memory_limit_mb": 8000,
    "cleanup_threshold_mb": 6000,
    "aggressive_threshold_mb": 7000,
    "cleanup_interval_seconds": 180,
    "gc_interval_seconds": 300,
    "component_limits": {
      "specialized_embeddings": {
        "max_memory_mb": 4000,
        "max_items": 3
      },
      "progressive_cache": {
        "max_memory_mb": 200,
        "max_items": 100
      },
      "context_tracking": {
        "max_memory_mb": 100,
        "max_files": 100,
        "max_timeline_events": 500
      }
    }
  }
}
```

## Memory Cleanup Strategy

### Normal Cleanup (at 6GB threshold)
- Remove 20% of oldest cache entries
- Evict least recently used models (keep at least 1)
- Trim oldest context events

### Aggressive Cleanup (at 7GB threshold)
- Remove 50% of cache entries
- Keep only 1 embedding model
- Aggressively trim context history
- Force garbage collection

### Continuous Monitoring
- Cleanup thread runs every 3 minutes
- GC thread runs every 5 minutes
- Components track their own memory usage
- System memory (RSS) monitored via psutil

## Usage

### Health Check
The health_check tool now includes memory status:

```bash
# In Claude Code
"Check system health"

# Returns memory section:
{
  "services": {
    "memory_manager": {
      "status": "healthy|high|critical",
      "process_memory_mb": 2500.5,
      "component_memory_mb": 1200.3,
      "components": {
        "specialized_embeddings": {
          "memory_mb": 1000.0,
          "items": 2
        },
        "progressive_cache": {
          "memory_mb": 150.5,
          "items": 78
        },
        "context_tracking": {
          "memory_mb": 50.0,
          "items": 234
        }
      },
      "thresholds": {
        "cleanup_mb": 6000,
        "aggressive_mb": 7000,
        "total_limit_mb": 8000
      }
    }
  }
}
```

### Memory Report Tool
Check detailed memory usage:

```bash
cd src/utils
./check_memory.py

# Output:
Qdrant RAG Memory Report
==================================================

System Memory:
  Process RSS: 2500.5 MB
  Process VMS: 4500.2 MB
  System Available: 16384.0 MB
  System Usage: 65.5%

Component Memory Usage:
  specialized_embeddings:
    Memory: 1000.0 MB
    Items: 2
    Last Cleanup: 2024-01-15T10:30:00
  progressive_cache:
    Memory: 150.5 MB
    Items: 78
  context_tracking:
    Memory: 50.0 MB
    Items: 234

  Total Component Memory: 1200.5 MB

Memory Limits:
  Total Limit: 8000 MB
  Cleanup Threshold: 6000 MB
  Aggressive Threshold: 7000 MB

✅ Memory usage is normal
```

## Benefits

1. **Prevents Memory Leaks**: Automatic cleanup prevents unbounded growth
2. **Optimized Performance**: LRU eviction keeps frequently used items
3. **Configurable Limits**: Adjust thresholds based on system resources
4. **Transparent Monitoring**: Health check and reports show memory status
5. **Graceful Degradation**: Progressive cleanup maintains functionality

## Design Decisions

### Hybrid Approach
The system uses a hybrid approach where:
1. **UnifiedMemoryManager** provides centralized monitoring and coordination
2. **Components** retain specialized cleanup logic where needed
3. **Configuration** is centralized but components can have custom behavior

This design allows:
- Specialized components (like embeddings) to handle device-specific cleanup
- Simple components to inherit generic LRU behavior
- Central monitoring without sacrificing component autonomy

### Why Not Fully Centralized?
- **Device-specific cleanup**: GPU/MPS cache clearing needs to happen at model eviction
- **Model memory estimation**: Requires domain knowledge about specific models
- **TTL expiration**: Cache-specific feature not needed by all components
- **Performance**: Components can optimize their own cleanup strategies

## Implementation Details

### Memory Estimation
- Models: Based on actual measurements on MPS
- Cache entries: sys.getsizeof() for accurate sizing
- Context events: Aggregate size of stored data

### Thread Safety
- All components use threading locks
- Registry operations are thread-safe
- Cleanup operations are atomic

### Persistence
- Cache data persisted to disk
- Sessions saved periodically
- Graceful shutdown saves state

## Troubleshooting

### High Memory Usage
1. Check health status: `"Check system health"`
2. Review memory report: `./check_memory.py`
3. Adjust thresholds in .env or server_config.json
4. Restart server to clear all memory

### Memory Not Releasing
1. Ensure QDRANT_MEMORY_MANAGEMENT_ENABLED=true
2. Check cleanup intervals aren't too long
3. Verify Python garbage collection is working
4. Consider reducing component limits

### Performance Impact
1. Cleanup operations are lightweight
2. GC runs in separate threads
3. LRU eviction is O(1)
4. Minimal impact on search/index operations