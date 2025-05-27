# Reindex Test Project

This is a test project for verifying the reindex functionality.

## Test Files

- `src/hello.py` - Contains greeting functions
- `src/goodbye.py` - Contains farewell functions

## Test Scenarios

1. **Initial Index**: Index all files
2. **File Deletion**: Delete hello.py and verify stale data
3. **File Rename**: Rename goodbye.py to farewell.py
4. **Reindex**: Use reindex_directory to clear stale data