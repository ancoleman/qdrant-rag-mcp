# Search Context Improvement Plan

## Current Limitations

Based on the example search session, the main issues are:

1. **Chunk Size Limitations**:
   - Code chunks: 1500 chars (often splits functions/classes)
   - Config chunks: 1000 chars (may split nested structures)
   - AST chunks: 2000 chars (better but still limited)

2. **Limited Context**: Search returns only the matched chunk without surrounding code

3. **No Full-File Option**: Can't easily retrieve complete file context when needed

## Practical Improvements for v0.2.0

### 1. Implement Context Expansion in Search

Add parameters to `search` and `search_code` functions:

```python
def search(
    query: str,
    n_results: int = 5,
    include_context: bool = True,  # NEW
    context_lines: int = 50,        # NEW - lines before/after
    ...
):
```

Implementation approach:
- When a chunk matches, also fetch adjacent chunks from the same file
- Combine them intelligently to show surrounding context
- Mark the matched section clearly

### 2. Add File-Aware Search Mode

New parameter for searching with file context:

```python
def search_code(
    query: str,
    file_context_mode: bool = False,  # NEW
    ...
):
```

When enabled:
- Group results by file
- Show file summary with all matching chunks
- Include file-level metadata (imports, classes, functions)

### 3. Increase Default Chunk Sizes

Update defaults based on modern token limits:

```python
# Code Indexer
chunk_size: 1500 → 3000  # Double the context
chunk_overlap: 300 → 600  # Better continuity

# Config Indexer  
chunk_size: 1000 → 2500  # Handle larger config sections

# AST Chunker (already good at 2000)
# Keep as is - it's structure-aware
```

### 4. Add Smart Result Expansion

Implement `expand_result()` method that:
- Detects if the chunk is incomplete (e.g., truncated function)
- Automatically includes the complete logical unit
- Uses AST data to find boundaries

### 5. New MCP Tool: get_file_context

Add a dedicated tool for when full file context is needed:

```python
@mcp.tool()
def get_file_context(
    file_path: str,
    focus_lines: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """Get full file with optional focus area highlighted"""
```

## Implementation Priority

1. **Quick Win**: Increase chunk sizes (config change)
2. **High Impact**: Add context expansion to search results
3. **Better UX**: Implement get_file_context tool
4. **Advanced**: Smart result expansion using AST data

## Example: Improved Search Flow

Before:
```
1. search("uv requirements") → Limited results
2. grep("uv") → Find files
3. read(file1) → Get context
4. read(file2) → Get more context
```

After:
```
1. search("uv requirements", include_context=True) → Complete context
   OR
2. get_file_context("install_global.sh", focus_lines=(60, 70)) → Full file
```

## Configuration

Add to `.env`:
```bash
# Search context settings
SEARCH_CONTEXT_LINES=50
SEARCH_INCLUDE_CONTEXT=true
CODE_CHUNK_SIZE=3000
CONFIG_CHUNK_SIZE=2500
```

## Benefits

1. **Fewer Operations**: 50-70% reduction in follow-up reads
2. **Better Context**: See complete functions/configurations
3. **Faster Workflow**: Get answers in one search
4. **Token Efficiency**: Better information density per operation

## Backward Compatibility

- All changes are additive with sensible defaults
- Existing searches continue to work unchanged
- New parameters are optional