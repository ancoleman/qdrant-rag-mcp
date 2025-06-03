# Progressive Context Code Structure

## Overview

This document shows how Progressive Context Management code will be organized within our existing project structure.

## New Files

### `src/utils/progressive_context.py`

Main module containing all progressive context logic:

```python
"""Progressive Context Management for token-efficient search."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict
import time

from ..utils.embeddings import get_embedding_model
from ..config import get_config

# Main components
class ProgressiveContextManager:
    """Manages multi-level context retrieval and caching."""
    ...

class SemanticCache:
    """Caches query results with semantic similarity matching."""
    ...

class HierarchyBuilder:
    """Builds hierarchical code structure from search results."""
    ...

class QueryIntentClassifier:
    """Classifies query intent to determine appropriate context level."""
    ...

# Data structures
@dataclass
class ProgressiveResult:
    """Result structure for progressive context retrieval."""
    ...

@dataclass
class ExpansionOption:
    """Options for expanding context to more detail."""
    ...

@dataclass
class CodeHierarchy:
    """Hierarchical representation of code structure."""
    ...
```

## Modified Files

### `src/qdrant_mcp_context_aware.py`

Add progressive context support to existing search functions:

```python
# New imports
from utils.progressive_context import (
    ProgressiveContextManager,
    QueryIntentClassifier,
    ProgressiveResult
)

# Initialize progressive components (lazy loading)
_progressive_manager = None
_query_classifier = None

def get_progressive_manager():
    """Get or initialize progressive context manager."""
    global _progressive_manager
    if _progressive_manager is None:
        config = get_config()
        if config.get("progressive_context", {}).get("enabled", False):
            _progressive_manager = ProgressiveContextManager(
                get_qdrant_client(),
                get_embedding_model(),
                config.get("progressive_context", {})
            )
    return _progressive_manager

# Enhanced search function
def search(
    query: str,
    n_results: int = 5,
    cross_project: bool = False,
    search_mode: str = "hybrid",
    include_dependencies: bool = False,
    include_context: bool = True,
    context_chunks: int = 1,
    # New progressive parameters
    context_level: str = "auto",
    progressive_mode: bool = None,
    include_expansion_options: bool = True,
    semantic_cache: bool = True
) -> Dict[str, Any]:
    """
    Search indexed content with optional progressive context.
    
    ... existing docstring ...
    
    New Parameters:
        context_level: Granularity level ("auto", "file", "class", "method", "full")
        progressive_mode: Enable progressive features (None = auto-detect)
        include_expansion_options: Include drill-down options
        semantic_cache: Use semantic similarity caching
    """
    logger = get_logger()
    
    # Determine if progressive mode should be used
    config = get_config()
    progressive_enabled = config.get("progressive_context", {}).get("enabled", False)
    
    if progressive_mode is None:
        # Auto-detect based on context_level
        progressive_mode = progressive_enabled and context_level != "full"
    
    # If progressive mode is disabled or not requested, use existing implementation
    if not progressive_mode or not progressive_enabled:
        # ... existing search implementation ...
        return existing_search_logic()
    
    # Progressive search implementation
    progressive_manager = get_progressive_manager()
    if not progressive_manager:
        # Fallback to regular search if manager not available
        return existing_search_logic()
    
    # Use progressive manager
    progressive_result = progressive_manager.get_progressive_context(
        query=query,
        level=context_level,
        n_results=n_results,
        cross_project=cross_project,
        search_mode=search_mode,
        include_dependencies=include_dependencies,
        semantic_cache=semantic_cache
    )
    
    # Convert to standard response format with progressive metadata
    response = {
        "results": progressive_result.results,
        "query": query,
        "total": len(progressive_result.results),
        "search_mode": search_mode,
        "project_context": get_current_project()["name"] if get_current_project() else None,
        "search_scope": "cross-project" if cross_project else "current project"
    }
    
    # Add progressive metadata
    if include_expansion_options or progressive_result.level != "full":
        response["progressive"] = {
            "level_used": progressive_result.level,
            "token_estimate": progressive_result.token_estimate,
            "token_reduction": progressive_result.token_reduction_percent,
            "expansion_options": progressive_result.expansion_options if include_expansion_options else [],
            "cache_hit": progressive_result.from_cache,
            "query_intent": progressive_result.query_intent
        }
    
    # Track in context tracking
    tracker = get_context_tracker()
    if tracker:
        tracker.track_search(query, response["results"], search_type="progressive")
    
    return response
```

### `src/config.py`

No changes needed - existing config loading will handle new progressive_context section.

### `config/server_config.json`

Add progressive context configuration:

```json
{
  "progressive_context": {
    "enabled": false,
    "default_level": "auto",
    "cache": {
      "enabled": true,
      "similarity_threshold": 0.85,
      "max_cache_size": 1000,
      "ttl_seconds": 3600,
      "persistence_enabled": true,
      "persistence_path": "~/.mcp-servers/qdrant-rag/progressive_cache"
    },
    "levels": {
      "file": {
        "include_summaries": true,
        "max_summary_length": 500,
        "include_structure": true,
        "token_reduction_target": 0.7
      },
      "class": {
        "include_signatures": true,
        "include_docstrings": true,
        "exclude_private": false,
        "token_reduction_target": 0.5
      },
      "method": {
        "include_implementation": true,
        "context_lines": 10,
        "token_reduction_target": 0.2
      }
    },
    "query_classification": {
      "enabled": true,
      "confidence_threshold": 0.7,
      "fallback_level": "class"
    }
  }
}
```

## Integration Points

### 1. With Context Tracking (`utils/context_tracking.py`)

```python
# In ProgressiveContextManager
def get_progressive_context(self, ...):
    # ... perform search ...
    
    # Track token usage
    tracker = get_context_tracker()
    if tracker:
        tracker.track_operation(
            operation_type="progressive_search",
            tokens_used=result.token_estimate,
            metadata={
                "level": level,
                "cache_hit": from_cache,
                "reduction": result.token_reduction_percent
            }
        )
```

### 2. With Enhanced Ranking (`utils/enhanced_ranker.py`)

```python
# Progressive context uses enhanced ranking for better result selection
results = enhanced_ranker.rank_results(
    raw_results,
    query_file_path=context.get("current_file"),
    ranking_weights=config.get("ranking_weights")
)
```

### 3. With AST Chunking (`utils/ast_chunker.py`)

```python
# HierarchyBuilder uses AST metadata to build structure
def build_hierarchy(self, chunks):
    for chunk in chunks:
        if chunk.get("chunk_type") == "class":
            # Use AST metadata for accurate hierarchy
            metadata = chunk["metadata"]
            hierarchy.add_class(
                file_path=chunk["file_path"],
                class_name=metadata["name"],
                methods=metadata.get("methods", [])
            )
```

## Testing Structure

### New Test Files

```
tests/
├── test_progressive_context.py      # Core functionality tests
├── test_semantic_cache.py           # Cache behavior tests
├── test_query_classification.py     # Intent classification tests
└── test_progressive_integration.py  # Integration with search tools
```

### Test Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test with actual search functions
3. **Performance Tests**: Measure token reduction and cache hit rates
4. **Regression Tests**: Ensure existing functionality unchanged

## Deployment Strategy

### Phase 1: Hidden Feature (v0.3.2-alpha)
- Code merged but feature flag disabled
- Internal testing only

### Phase 2: Beta Release (v0.3.2-beta)
- Feature flag enabled for opt-in users
- Collect metrics and feedback

### Phase 3: General Availability (v0.3.2)
- Feature enabled by default
- Can still be disabled via config

### Phase 4: Full Integration (v0.4.0)
- Remove feature flag
- Progressive context becomes standard

This structure ensures clean separation of concerns while allowing deep integration with existing functionality.