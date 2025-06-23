# Troubleshooting MCP Server Connection Issues

This guide helps resolve intermittent connection issues with the Qdrant RAG MCP server in Claude Code.

## Common Issues and Solutions

### 1. Multiple Server Processes

**Problem**: Multiple MCP server processes running simultaneously can cause connection failures.

**Symptoms**:
- MCP server doesn't connect on new Claude sessions
- `claude --resume` fails to connect to the server
- Intermittent connection failures

**Solution**:
```bash
# Check for multiple processes
ps aux | grep qdrant_mcp_context_aware.py | grep -v grep

# Kill all MCP server processes
pkill -f qdrant_mcp_context_aware.py

# Restart Claude
claude --new
```

### 2. Stale Process Cleanup

The updated global runner script now automatically kills stale processes before starting. To apply this fix:

```bash
cd ~/Documents/repos/mcp-servers/qdrant-rag
./install_global.sh
```

### 3. Diagnostic Tools

Use the diagnostic script to check system state:

```bash
./scripts/diagnose_mcp_connection.sh
```

This script checks:
- Running MCP processes
- Claude configuration
- Environment variables
- Qdrant connection
- Recent log activity
- Stale lock files

### 4. Manual Cleanup

For persistent issues, use the cleanup script:

```bash
./scripts/ensure_clean_startup.sh
```

This script:
- Kills all MCP server processes
- Removes stale lock files
- Prepares for clean startup

## Connection Lifecycle

The MCP server connection follows this lifecycle:

1. **Claude starts** → Launches MCP servers configured in `claude mcp list`
2. **Global runner script** → Kills stale processes, sets up environment
3. **MCP server starts** → Initializes memory manager, connects to Qdrant
4. **Server ready** → Waits for commands from Claude via stdio
5. **Claude session ends** → MCP server should terminate

## Best Practices

### 1. Clean Sessions

Start fresh Claude sessions when experiencing issues:
```bash
# Kill stale processes first
pkill -f qdrant_mcp_context_aware.py

# Start new session
claude --new
```

### 2. Check Logs

Monitor server startup in logs:
```bash
# Watch startup logs
./scripts/qdrant-logs -f | grep -E "(Starting|Process ID|Ready)"
```

### 3. Environment Variables

For debugging, set explicit environment:
```bash
export MCP_CLIENT_CWD=$(pwd)
export QDRANT_LOG_LEVEL=DEBUG
claude
```

## Startup Timing Issues

The server now logs detailed startup information:
- Version number
- Process ID
- Working directory
- Python executable
- Startup timestamp
- Qdrant connection status

This helps diagnose timing-related connection failures.

## Prevention

1. **Use the updated install script** - It now includes automatic process cleanup
2. **Monitor process count** - Only one MCP server process should run per Claude session
3. **Check logs regularly** - Look for startup/shutdown patterns

## Emergency Recovery

If all else fails:

```bash
# 1. Complete cleanup
pkill -9 -f qdrant_mcp_context_aware.py
pkill -9 -f "uv run.*qdrant"

# 2. Clear any cache/state
rm -rf ~/.mcp-servers/qdrant-rag/sessions/*
rm -rf ~/.mcp-servers/qdrant-rag/logs/errors/*

# 3. Reinstall
cd ~/Documents/repos/mcp-servers/qdrant-rag
./install_global.sh

# 4. Test with new session
claude --new
```

## Verification

After troubleshooting, verify the fix:

1. In Claude, ask: "Check system health"
2. Should see: "✅ Qdrant connection: healthy"
3. Check process count: `ps aux | grep qdrant_mcp | grep -v grep | wc -l` (should be 2: uv + python)

## Future Improvements

The server could benefit from:
- Process lock files to prevent multiple instances
- Automatic stale process detection and cleanup
- Connection retry with exponential backoff
- Health check endpoint for Claude to verify readiness