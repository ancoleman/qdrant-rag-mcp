# Reindex Implementation Plan

## Problem Statement

The current `index_directory` function only adds new content to the vector database but doesn't remove stale data from deleted, renamed, or moved files. This leads to search results containing outdated information that no longer exists in the codebase.

## Solution Overview

Implement a new `reindex_directory` MCP tool that performs a clean reindex by:
1. Clearing existing collections for the current project
2. Re-indexing all files from scratch
3. Ensuring no stale data remains

## Implementation Tasks

### High Priority

#### 1. Design reindex function that clears collections before indexing
- Create a new MCP tool `reindex_directory`
- Design the API to be similar to `index_directory` for consistency
- Ensure it respects project boundaries (context-aware)

#### 2. Implement clear_project_collections() helper function
```python
def clear_project_collections() -> Dict[str, Any]:
    """
    Clear all collections for the current project.
    Returns info about what was cleared.
    """
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found"}
    
    cleared = []
    for collection_type in ['code', 'config']:
        collection_name = f"{current_project['collection_prefix']}_{collection_type}"
        # Clear collection logic
        cleared.append(collection_name)
    
    return {"cleared_collections": cleared, "project": current_project['name']}
```

#### 3. Add reindex_directory MCP tool
```python
@mcp.tool()
def reindex_directory(
    directory: str,
    patterns: Optional[List[str]] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Reindex a directory by first clearing existing project data.
    
    This prevents stale data from deleted/moved files.
    Use this when files have been renamed, moved, or deleted.
    
    Args:
        directory: Directory to reindex
        patterns: Optional file patterns to include
        force: Skip confirmation (for automation)
    
    Returns:
        Reindex results including what was cleared and indexed
    """
    # Implementation here
```

### Medium Priority

#### 4. Update reindex to handle both project-scoped and global collections
- Add parameter to control scope: `scope: Literal["project", "global", "all"] = "project"`
- Ensure global collections are only cleared when explicitly requested
- Maintain backward compatibility with existing behavior

#### 5. Add safety confirmation or force flag
- By default, show what will be cleared and ask for confirmation
- Add `force=True` parameter to skip confirmation
- Log all clear operations for audit trail

#### 6. Test reindex function with various scenarios
Test cases:
- File renamed: `old_name.py` → `new_name.py`
- File moved: `src/old.py` → `lib/old.py`
- File deleted: Remove `temp.py`
- Directory restructure: Move entire module
- Mixed operations: Multiple changes at once

### Low Priority

#### 7. Update documentation
Create user-facing documentation explaining:
- When to use `reindex_directory` vs `index_directory`
- Performance implications
- Best practices

Example documentation:
```markdown
## When to Use Reindex

Use `reindex_directory` when:
- Files have been renamed or moved
- Files have been deleted
- You suspect stale search results
- Major refactoring has occurred

Use `index_directory` when:
- Adding new files only
- Making content changes to existing files
- Performance is critical (incremental indexing)
```

#### 8. Update CHANGELOG.md for v0.1.1
```markdown
## [0.1.1] - 2024-XX-XX

### Added
- New `reindex_directory` MCP tool for clean reindexing
- `clear_project_collections()` helper function
- Force flag for automated reindexing workflows

### Fixed
- Stale data persisting after file deletions/renames
- Search results showing non-existent files

### Changed
- Improved documentation on indexing strategies
```

## Implementation Details

### Collection Management

The reindex operation must:
1. Identify current project context
2. List all collections for the project
3. Clear collections safely (with error handling)
4. Perform fresh indexing
5. Report statistics

### Error Handling

Handle these edge cases:
- No project context (working in /tmp, etc.)
- Collection doesn't exist
- Qdrant connection issues
- Partial index failures

### Performance Considerations

- Clearing collections is fast (< 1 second)
- Reindexing is proportional to codebase size
- Consider progress reporting for large codebases
- May want to implement batch operations

## Code Structure

Add to `src/qdrant_mcp_context_aware.py`:
```python
# After existing imports
from typing import Literal

# New helper function
def clear_project_collections() -> Dict[str, Any]:
    # Implementation

# New MCP tool
@mcp.tool()
def reindex_directory(...):
    # Implementation
```

## Testing Strategy

1. Unit tests for `clear_project_collections()`
2. Integration tests for `reindex_directory`
3. Manual testing with real projects
4. Performance benchmarks

## Success Criteria

- No stale data after file operations
- Clear user feedback on what was cleared
- Minimal performance impact
- Backward compatibility maintained
- Well-documented feature

## Future Enhancements

- Selective reindexing (only clear specific files)
- Dry-run mode to preview what would be cleared
- Automatic detection of moved/renamed files
- Background reindexing with progress updates