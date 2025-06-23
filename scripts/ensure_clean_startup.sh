#!/bin/bash
# Ensure clean MCP server startup by killing stale processes

echo "🧹 Cleaning up stale MCP server processes..."

# Kill any existing MCP server processes
KILLED_COUNT=0
for PID in $(ps aux | grep -E "qdrant_mcp_context_aware.py" | grep -v grep | awk '{print $2}'); do
    echo "   Killing process $PID..."
    kill -9 $PID 2>/dev/null
    KILLED_COUNT=$((KILLED_COUNT + 1))
done

if [ $KILLED_COUNT -gt 0 ]; then
    echo "   ✅ Killed $KILLED_COUNT stale processes"
    sleep 1  # Give processes time to fully terminate
else
    echo "   ✅ No stale processes found"
fi

# Clear any stale lock files (if they exist in future versions)
LOCK_DIR="$HOME/.mcp-servers/qdrant-rag"
if [ -d "$LOCK_DIR" ]; then
    find "$LOCK_DIR" -name "*.pid" -o -name "*.lock" -o -name "*.socket" -type f -delete 2>/dev/null
fi

echo "✨ Ready for clean startup!"