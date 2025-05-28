# Search Improvements for v0.2.0

## Overview

This document outlines the improvements made to the qdrant-rag search functionality to provide more comprehensive results and reduce the need for follow-up file reads.

## Problem Statement

The current search implementation returns limited context, requiring multiple follow-up operations:
- Search results contain only the matched chunk (typically 1500 chars)
- Users need to perform additional Read operations to understand surrounding code
- This increases token usage and slows down the workflow

## Implemented Solutions

### 1. Increased Default Chunk Sizes

Doubled the default chunk sizes to capture more content:

```python
# Code chunks
chunk_size: 1500 → 3000 characters
chunk_overlap: 300 → 600 characters

# Config chunks  
chunk_size: 1000 → 2000 characters
chunk_overlap: 200 → 400 characters
```

### 2. Context Expansion Feature

Added automatic context expansion to search results:

```python
def search(
    query: str, 
    n_results: int = 10,  # Increased from 5
    include_context: bool = True,  # NEW
    context_chunks: int = 1,  # NEW (0-3)
    ...
)
```

When `include_context=True`, each search result includes:
- The matched chunk
- N chunks before (based on `context_chunks`)
- N chunks after (based on `context_chunks`)

### 3. Enhanced Result Format

Search results now include:

```json
{
  "content": "...",  // Original matched chunk
  "expanded_content": "=== Context Before (chunk 2) ===\n...\n=== Matched Section (chunk 3) ===\n...\n=== Context After (chunk 4) ===\n...",
  "has_context": true,
  "total_line_range": {"start": 45, "end": 187},
  // ... other fields
}
```

### 4. New Tool: get_file_chunks

Added a dedicated tool for retrieving complete file context:

```python
@mcp.tool()
def get_file_chunks(
    file_path: str,
    start_chunk: Optional[int] = None,
    end_chunk: Optional[int] = None
) -> Dict[str, Any]
```

This allows:
- Fetching all chunks for a file
- Getting specific chunk ranges
- Viewing how files are chunked

## Usage Examples

### Basic Search with Context
```python
# Default: Returns results with 1 chunk before/after
results = search("authentication middleware")
```

### Extended Context Search
```python
# Get more surrounding context (2 chunks each direction)
results = search("error handling", context_chunks=2)
```

### Traditional Search (No Context)
```python
# Disable context expansion for faster, smaller results
results = search("quick lookup", include_context=False)
```

### Full File Context
```python
# Get all chunks for deep analysis
chunks = get_file_chunks("/src/auth.py")
```

## Performance Considerations

1. **Token Usage**: Expanded context increases token usage but reduces follow-up queries
2. **Search Speed**: Minimal impact as chunk fetching is fast
3. **Memory**: Larger chunks use more memory but within reasonable limits

## Configuration

The chunk sizes can be adjusted via environment variables or config:

```bash
# In .env or environment
CODE_CHUNK_SIZE=3000
CODE_CHUNK_OVERLAP=600
CONFIG_CHUNK_SIZE=2000
CONFIG_CHUNK_OVERLAP=400
```

## Benefits

1. **Fewer Operations**: One search often provides enough context
2. **Better Understanding**: See how code fits in its surroundings
3. **Flexibility**: Choose context level based on needs
4. **Backward Compatible**: Existing searches work unchanged

## Future Enhancements

Consider for future versions:
1. Smart context boundaries (complete functions/classes)
2. Relevance-based context expansion
3. File-type specific context strategies
4. Context summarization for very large expansions