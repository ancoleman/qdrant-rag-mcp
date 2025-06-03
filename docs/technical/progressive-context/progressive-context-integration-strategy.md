# Progressive Context Integration Strategy

## Decision: Enhance Existing Search Tools

After analyzing our codebase, we'll integrate Progressive Context Management into our existing search tools rather than creating separate ones.

## Integration Design

### 1. Enhanced Parameters for Existing Tools

Add these parameters to `search()`, `search_code()`, and `search_docs()`:

```python
def search(
    query: str,
    n_results: int = 5,
    # Existing parameters...
    search_mode: str = "hybrid",
    include_context: bool = True,
    context_chunks: int = 1,
    include_dependencies: bool = False,
    # New progressive context parameters
    context_level: str = "auto",  # "auto", "file", "class", "method", "full"
    progressive_mode: bool = None,  # None = auto based on context_level
    include_expansion_options: bool = True,
    semantic_cache: bool = True
) -> Dict[str, Any]:
```

### 2. Context Levels

- **"auto"** (default): Use query intent classification to determine level
- **"file"**: Return file-level summaries (70% token reduction)
- **"class"**: Return class/function signatures (50% token reduction)  
- **"method"**: Return full implementations (20% token reduction)
- **"full"**: Current behavior (no reduction)

### 3. Backward Compatibility

```python
# Old usage still works exactly the same
search(query="authentication", n_results=5)

# New progressive usage
search(query="authentication", context_level="file")

# Explicit progressive mode
search(query="authentication", progressive_mode=True)
```

### 4. Response Structure Enhancement

Current response structure remains, with optional progressive fields:

```python
{
    "results": [...],  # Same as before
    "query": "...",
    "total": 10,
    # New progressive fields (only when progressive_mode=True)
    "progressive": {
        "level_used": "file",
        "token_estimate": 1500,
        "token_reduction": "70%",
        "expansion_options": [
            {
                "type": "class",
                "path": "auth/authenticator.py::Authenticator",
                "estimated_tokens": 800,
                "relevance": 0.92
            }
        ],
        "cache_hit": true,
        "query_intent": {
            "type": "exploration",
            "confidence": 0.85
        }
    }
}
```

### 5. Implementation Strategy

#### Phase 1: Infrastructure (Behind Feature Flag)
```python
# In server_config.json
"progressive_context": {
    "enabled": false,  # Feature flag
    "default_level": "auto",
    ...
}

# In search functions
if config.get("progressive_context", {}).get("enabled", False):
    # Use progressive manager
else:
    # Current implementation
```

#### Phase 2: Gradual Rollout
1. Start with `search()` function only
2. Add to `search_code()` after validation
3. Finally add to `search_docs()`

#### Phase 3: Make Default
Once proven stable, flip the default to enabled

### 6. Query Intent Auto-Detection

When `context_level="auto"` (default), use patterns:

```python
# High-level exploration → file level
"what does X do", "explain", "overview", "architecture"

# Debugging/implementation → method level  
"bug in", "error", "line", "implementation of"

# Navigation → class level
"find", "where is", "show me"
```

### 7. Synergy with Existing Features

```python
# Progressive + context expansion
search(query="auth", context_level="class", include_context=True)
# Returns: Class signatures + surrounding file context

# Progressive + dependencies
search_code(query="login", context_level="file", include_dependencies=True)
# Returns: File summaries + imported file summaries

# Progressive + search modes
search(query="config", context_level="file", search_mode="keyword")
# Returns: File summaries from keyword search
```

## Benefits of This Approach

1. **No Learning Curve**: Users continue using familiar tools
2. **Gradual Adoption**: Can enable progressively with feature flags
3. **Backward Compatible**: Existing code/scripts continue working
4. **Natural Evolution**: Follows established enhancement pattern
5. **Unified Experience**: One set of tools for all search needs

## Code Architecture

```
src/
├── utils/
│   ├── progressive_context.py  # New module
│   │   ├── ProgressiveContextManager
│   │   ├── SemanticCache
│   │   ├── HierarchyBuilder
│   │   └── QueryIntentClassifier
│   └── ...existing utils...
└── qdrant_mcp_context_aware.py
    ├── search()  # Enhanced with progressive
    ├── search_code()  # Enhanced with progressive
    └── search_docs()  # Enhanced with progressive
```

## Migration Path

1. **v0.3.2-alpha**: Feature flag off, infrastructure in place
2. **v0.3.2-beta**: Feature flag on for early adopters
3. **v0.3.2**: Feature enabled by default, can be disabled
4. **v0.4.0**: Remove feature flag, fully integrated

This integration strategy provides the best balance of innovation and stability, allowing us to deliver Progressive Context Management without disrupting existing users.