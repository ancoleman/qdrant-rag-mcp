# Reindex Function Testing Plan

## Making Claude Code Use Latest Version

### Method 1: Restart Claude Code (Recommended)
```bash
# 1. Close Claude Code completely
# 2. Ensure MCP server is up to date
cd ~/Documents/repos/mcp-servers/qdrant-rag
git pull

# 3. Reinstall globally (updates the symlink)
./install_global.sh

# 4. Start Claude Code fresh
claude
```

### Method 2: Force Reload MCP Servers
In Claude Code:
1. Type: `/reload` (if available)
2. Or ask: "reload your MCP servers"
3. Or simply close and reopen Claude Code

### Method 3: Verify Current Version
Ask Claude Code:
- "What version of qdrant-rag are you running?"
- "Can you use the reindex_directory function?"
- "Show me available MCP tools for qdrant-rag"

## Test Scenarios

### Test 1: Basic Reindex with Stale Data
```bash
# Setup
cd /tmp
mkdir -p reindex-test/src
cd reindex-test

# Create initial files
echo "def hello(): return 'world'" > src/hello.py
echo "def goodbye(): return 'farewell'" > src/goodbye.py

# In Claude Code:
# "Index the current directory"
# "Search for 'hello' function"  # Should find it

# Simulate file changes
rm src/hello.py
mv src/goodbye.py src/farewell.py
echo "def greet(): return 'hi'" > src/greet.py

# Test regular index (should show stale data)
# "Index the current directory again"
# "Search for 'hello' function"  # Should still find deleted file

# Test reindex (should clear stale data)
# "Reindex the current directory"
# "Search for 'hello' function"  # Should NOT find it
# "Search for 'greet' function"  # Should find new file
```

### Test 2: Project-Scoped Reindex
```bash
# Create two projects
mkdir -p /tmp/project-a/src
mkdir -p /tmp/project-b/src

# Project A
cd /tmp/project-a
echo "class ProjectA: pass" > src/main.py

# Project B
cd /tmp/project-b
echo "class ProjectB: pass" > src/main.py

# In Claude Code:
# Navigate to project-a: "cd /tmp/project-a"
# "Index current directory"
# "Search for ProjectA"  # Should find it

# Navigate to project-b: "cd /tmp/project-b"
# "Index current directory"
# "Search for ProjectB"  # Should find it
# "Search for ProjectA"  # Should NOT find it (different project)

# Test reindex doesn't affect other projects
# "Reindex current directory"
# Navigate back: "cd /tmp/project-a"
# "Search for ProjectA"  # Should still find it
```

### Test 3: Force Flag Usage
```bash
cd /tmp/reindex-test

# In Claude Code:
# "Reindex with force flag"
# or
# "Force reindex this directory"

# Should clear and reindex without confirmation
```

### Test 4: Error Handling
```bash
# Test non-existent directory
# "Reindex /tmp/does-not-exist"  # Should error gracefully

# Test no project context
cd /tmp
# "Reindex"  # Should error about no project context
```

## Verification Steps

1. **Check Collection Status**
   - Before reindex: Note number of indexed files
   - After reindex: Verify count matches current files
   - Search for deleted files: Should return no results

2. **Monitor Logs**
   ```bash
   # Watch server logs during testing
   tail -f ~/Documents/repos/mcp-servers/qdrant-rag/logs/*.log
   ```

3. **Direct API Testing** (Optional)
   ```bash
   # Start HTTP server for debugging
   cd ~/Documents/repos/mcp-servers/qdrant-rag
   python src/http_server.py

   # Check collection stats
   curl http://localhost:6333/collections/code_collection
   ```

## Expected Results

✅ **Success Criteria:**
- Deleted files no longer appear in search results after reindex
- Renamed files only appear with new names
- File counts match actual directory contents
- Project isolation is maintained
- Force flag skips confirmation

❌ **Failure Indicators:**
- Stale files still appear in searches
- Collections not properly cleared
- Cross-project data leakage
- Errors during reindex operation

## Troubleshooting

If reindex function is not available:
1. Check MCP server logs for errors
2. Verify git is on latest commit
3. Ensure install_global.sh was run
4. Try manual MCP reload:
   ```bash
   claude mcp remove qdrant-rag
   claude mcp add qdrant-rag -s user ~/.mcp-servers/qdrant-rag-global.sh
   ```

If tests fail:
1. Check Qdrant is running: `docker ps | grep qdrant`
2. Verify collections exist: `curl http://localhost:6333/collections`
3. Check server logs for detailed errors
4. Test with HTTP API directly for debugging