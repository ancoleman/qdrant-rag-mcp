# Quick Reindex Test Instructions

## Step 1: Ensure Claude Code Uses Latest Version

```bash
# From the qdrant-rag directory:
./install_global.sh

# Then restart Claude Code completely
# (Close the app and reopen)
```

## Step 2: Test Commands to Run in Claude Code

Once Claude Code is restarted, navigate to the test directory and run these commands:

```
cd ~/Documents/repos/mcp-servers/qdrant-rag/tests/reindex-test

# 1. Initial index
"Index this directory"

# 2. Search for our test functions
"Search for hello function"
"Search for goodbye function"

# 3. Simulate file changes (run in terminal)
rm src/hello.py
mv src/goodbye.py src/farewell.py
echo "def new_function(): return 'new'" > src/new_file.py

# 4. Test regular index (should still find deleted files)
"Index this directory"
"Search for hello function"  # Should still find it (stale data)

# 5. Test reindex (should clear stale data)
"Reindex this directory"
"Search for hello function"  # Should NOT find it
"Search for new_function"    # Should find it
```

## Step 3: Verify Results

Ask Claude Code:
- "What files are currently indexed for this project?"
- "Show me the current project context"
- "Search for all functions in src directory"

## Expected Behavior

✅ After reindex:
- `hello.py` should NOT appear in search results
- `goodbye.py` should NOT appear (renamed)
- `farewell.py` should appear (new name)
- `new_file.py` should appear

❌ If regular index (not reindex):
- `hello.py` would still appear (stale)
- Both `goodbye.py` and `farewell.py` might appear