# Smart Reindex Implementation Plan (v0.2.4)

## Overview

Implement a hybrid incremental reindexing system that avoids clearing collections by tracking file changes and only updating what's necessary. This will make reindexing 90%+ faster for typical use cases while preserving existing embeddings.

## Goals

1. **Preserve Unchanged Data**: Keep existing embeddings for unchanged files
2. **Fast Incremental Updates**: Only process changed files
3. **Clean State Management**: Remove stale data from deleted/moved files
4. **Zero Downtime**: No service interruption during reindex
5. **Backward Compatible**: Support existing full reindex when needed

## Technical Approach: Hybrid Strategy

### 1. File State Tracking

Store additional metadata in each chunk:
```python
{
    "file_path": "/path/to/file.py",
    "file_hash": "sha256:abc123...",  # Content hash
    "file_mtime": 1735425600.0,        # Modification time
    "index_time": 1735425700.0,        # When indexed
    "chunk_index": 0,
    "total_chunks": 5,
    # ... existing metadata
}
```

### 2. Change Detection Algorithm

```python
def detect_changes(directory: str, collection_name: str) -> Dict[str, FileStatus]:
    """
    Compare current filesystem state with indexed state.
    Returns dict of file_path -> status (added/modified/unchanged/deleted)
    """
    # Get all indexed files from Qdrant
    indexed_files = get_indexed_files_metadata(collection_name)
    
    # Scan current directory
    current_files = scan_directory_with_hashes(directory)
    
    changes = {
        "added": [],
        "modified": [],
        "unchanged": [],
        "deleted": []
    }
    
    # Find additions and modifications
    for file_path, current_hash in current_files.items():
        if file_path not in indexed_files:
            changes["added"].append(file_path)
        elif indexed_files[file_path]["file_hash"] != current_hash:
            changes["modified"].append(file_path)
        else:
            changes["unchanged"].append(file_path)
    
    # Find deletions
    for file_path in indexed_files:
        if file_path not in current_files:
            changes["deleted"].append(file_path)
    
    return changes
```

### 3. Incremental Update Process

```python
def smart_reindex_directory(
    directory: str,
    incremental: bool = True,
    force_full: bool = False
) -> Dict[str, Any]:
    """Smart reindex with incremental updates."""
    
    if force_full or not incremental:
        # Fall back to current behavior
        return full_reindex_directory(directory)
    
    # Detect changes
    changes = detect_changes(directory, get_collection_name())
    
    # Apply changes
    results = {
        "added": 0,
        "modified": 0,
        "deleted": 0,
        "unchanged": len(changes["unchanged"]),
        "errors": []
    }
    
    # Delete chunks for removed/modified files
    for file_path in changes["deleted"] + changes["modified"]:
        delete_file_chunks(file_path)
        if file_path in changes["deleted"]:
            results["deleted"] += 1
    
    # Index new/modified files
    for file_path in changes["added"] + changes["modified"]:
        try:
            index_file_with_hash(file_path)
            if file_path in changes["added"]:
                results["added"] += 1
            else:
                results["modified"] += 1
        except Exception as e:
            results["errors"].append({
                "file": file_path,
                "error": str(e)
            })
    
    return results
```

### 4. Hash Calculation

Use SHA256 for reliable change detection:
```python
def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return f"sha256:{sha256_hash.hexdigest()}"
```

### 5. Stale Data Cleanup

```python
def delete_file_chunks(file_path: str, collection_name: str):
    """Delete all chunks for a specific file."""
    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_path",
                        match=models.MatchValue(value=file_path)
                    )
                ]
            )
        )
    )
```

## Implementation Steps

### Phase 1: Core Infrastructure (Day 1-2)

1. **Update Metadata Schema**
   - Add file_hash field to all indexers
   - Add index_time tracking
   - Ensure backward compatibility

2. **Implement Hash Calculation**
   - SHA256 hashing function
   - Integrate into all indexers
   - Handle large files efficiently

3. **Create Change Detection**
   - Query indexed files metadata
   - Compare with filesystem state
   - Return change report

### Phase 2: Incremental Logic (Day 2-3)

1. **Implement Smart Reindex**
   - Add incremental parameter
   - Detect changes before processing
   - Apply incremental updates

2. **Update Delete Operations**
   - Create delete_file_chunks function
   - Handle collection filtering
   - Ensure complete cleanup

3. **Progress Tracking**
   - Report files processed
   - Show time savings
   - Log detailed operations

### Phase 3: Optimizations (Day 3-4)

1. **Git Integration**
   - Use git status for faster change detection
   - Handle git-ignored files properly
   - Optimize for git workflows

2. **Batch Operations**
   - Batch deletions for efficiency
   - Parallel file processing
   - Memory-efficient scanning

3. **Error Handling**
   - Graceful fallback to full reindex
   - Detailed error reporting
   - Recovery mechanisms

### Phase 4: Testing & Polish (Day 4-5)

1. **Comprehensive Testing**
   - Unit tests for change detection
   - Integration tests for reindex
   - Performance benchmarks

2. **Documentation**
   - Update API documentation
   - Add usage examples
   - Performance comparisons

3. **User Experience**
   - Clear progress messages
   - Helpful error messages
   - Performance statistics

## API Changes

### Updated reindex_directory

```python
@mcp.tool()
async def reindex_directory(
    directory: str = ".",
    patterns: List[str] = None,
    recursive: bool = True,
    force: bool = False,
    incremental: bool = True  # NEW parameter
) -> Dict[str, Any]:
    """
    Reindex a directory with smart incremental updates.
    
    Args:
        directory: Directory to reindex
        patterns: File patterns to include
        recursive: Search subdirectories
        force: Skip confirmation
        incremental: Use smart incremental reindex (default: True)
    
    Returns:
        Reindex results with statistics
    """
```

## Performance Expectations

### Typical Reindex Scenarios

1. **No Changes**: <1 second (metadata check only)
2. **Few Files Changed**: 5-10 seconds (only changed files)
3. **Major Refactor**: 30-60 seconds (many files)
4. **Full Reindex**: Same as current (all files)

### Benchmarks

- 1000 files, 10 changed: 95% faster
- 5000 files, 50 changed: 90% faster
- 10000 files, 500 changed: 80% faster

## Risk Mitigation

1. **Data Integrity**
   - Verify hash calculations
   - Test deletion logic thoroughly
   - Add data validation checks

2. **Performance**
   - Monitor memory usage
   - Optimize for large directories
   - Add progress indicators

3. **Compatibility**
   - Maintain backward compatibility
   - Support force full reindex
   - Handle edge cases gracefully

## Success Criteria

1. **Functional**
   - ✓ Correctly detects all file changes
   - ✓ Updates only changed files
   - ✓ Removes stale data properly
   - ✓ Maintains data integrity

2. **Performance**
   - ✓ 90%+ faster for typical updates
   - ✓ Sub-second for no changes
   - ✓ Memory efficient for large repos

3. **User Experience**
   - ✓ Clear progress reporting
   - ✓ Helpful error messages
   - ✓ Seamless upgrade path

## Next Steps After v0.2.4

This smart reindex capability enables:
- Continuous indexing workflows
- Watch mode for real-time updates
- Integration with file watchers
- Background index maintenance