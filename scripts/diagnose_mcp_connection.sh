#!/bin/bash
# Diagnostic script for MCP server connection issues

echo "🔍 Qdrant RAG MCP Server Connection Diagnostics"
echo "==============================================="
echo ""

# Check for multiple processes
echo "1. Checking for running MCP processes:"
PROCESSES=$(ps aux | grep -E "qdrant_mcp_context_aware.py" | grep -v grep)
if [ -z "$PROCESSES" ]; then
    echo "   ✅ No MCP server processes running"
else
    echo "   ⚠️  Found running MCP processes:"
    echo "$PROCESSES" | awk '{print "      PID: " $2 " Started: " $9 " " $10}'
    echo ""
    echo "   Multiple processes can cause connection issues!"
fi
echo ""

# Check Claude configuration
echo "2. Checking Claude MCP configuration:"
CLAUDE_CONFIG=$(claude mcp list 2>/dev/null | grep qdrant-rag)
if [ -z "$CLAUDE_CONFIG" ]; then
    echo "   ❌ qdrant-rag not configured in Claude"
else
    echo "   ✅ Found configuration: $CLAUDE_CONFIG"
fi
echo ""

# Check global script
echo "3. Checking global runner script:"
GLOBAL_SCRIPT="$HOME/.mcp-servers/qdrant-rag-global.sh"
if [ -f "$GLOBAL_SCRIPT" ]; then
    echo "   ✅ Global script exists"
    if [ -x "$GLOBAL_SCRIPT" ]; then
        echo "   ✅ Script is executable"
    else
        echo "   ❌ Script is not executable"
    fi
else
    echo "   ❌ Global script not found at $GLOBAL_SCRIPT"
fi
echo ""

# Check environment
echo "4. Checking environment:"
echo "   MCP_CLIENT_CWD: ${MCP_CLIENT_CWD:-<not set>}"
echo "   QDRANT_RAG_AUTO_INDEX: ${QDRANT_RAG_AUTO_INDEX:-<not set>}"
echo "   QDRANT_HOST: ${QDRANT_HOST:-localhost}"
echo "   QDRANT_PORT: ${QDRANT_PORT:-6333}"
echo ""

# Check Qdrant connection
echo "5. Checking Qdrant connection:"
if curl -s http://localhost:6333/health > /dev/null 2>&1; then
    echo "   ✅ Qdrant is running and accessible"
else
    echo "   ❌ Cannot connect to Qdrant at localhost:6333"
fi
echo ""

# Check recent logs
echo "6. Recent log activity:"
LOG_DIR="$HOME/.mcp-servers/qdrant-rag/logs"
if [ -d "$LOG_DIR" ]; then
    RECENT_LOGS=$(find "$LOG_DIR" -name "*.log" -type f -mmin -60 2>/dev/null | wc -l)
    echo "   Found $RECENT_LOGS log files modified in the last hour"
    
    # Check for startup messages in recent logs
    RECENT_STARTUP=$(find "$LOG_DIR" -name "*.log" -type f -mmin -60 -exec grep -l "Starting Qdrant RAG MCP Server\|Started unified memory manager" {} \; 2>/dev/null | wc -l)
    echo "   Found $RECENT_STARTUP logs with startup messages"
    
    # Check for errors
    RECENT_ERRORS=$(find "$LOG_DIR" -name "*.log" -type f -mmin -60 -exec grep -c "ERROR\|CRITICAL" {} \; 2>/dev/null | awk '{sum+=$1} END {print sum}')
    if [ "$RECENT_ERRORS" -gt 0 ]; then
        echo "   ⚠️  Found $RECENT_ERRORS error messages in recent logs"
    else
        echo "   ✅ No errors in recent logs"
    fi
else
    echo "   ❌ Log directory not found"
fi
echo ""

# Check for lock files or stale state
echo "7. Checking for stale state:"
STALE_FILES=$(find "$HOME/.mcp-servers/qdrant-rag" -name "*.pid" -o -name "*.lock" -o -name "*.socket" 2>/dev/null)
if [ -z "$STALE_FILES" ]; then
    echo "   ✅ No stale lock/pid files found"
else
    echo "   ⚠️  Found stale files:"
    echo "$STALE_FILES"
fi
echo ""

# Test MCP server startup
echo "8. Testing MCP server startup:"
echo "   Attempting to start server in test mode..."
TEMP_LOG=$(mktemp)
timeout 5 bash -c "cd '$HOME/.mcp-servers/qdrant-rag' && uv run python src/qdrant_mcp_context_aware.py" > "$TEMP_LOG" 2>&1 &
TEST_PID=$!

sleep 3

if ps -p $TEST_PID > /dev/null 2>&1; then
    echo "   ✅ Server started successfully (PID: $TEST_PID)"
    kill $TEST_PID 2>/dev/null
else
    echo "   ❌ Server failed to start or exited early"
    if [ -s "$TEMP_LOG" ]; then
        echo "   Error output:"
        head -20 "$TEMP_LOG" | sed 's/^/      /'
    fi
fi
rm -f "$TEMP_LOG"
echo ""

# Recommendations
echo "📋 Recommendations:"
echo ""

if [ -n "$PROCESSES" ]; then
    echo "• Kill existing processes: pkill -f qdrant_mcp_context_aware.py"
fi

if [ -z "$CLAUDE_CONFIG" ]; then
    echo "• Reinstall the server: cd ~/Documents/repos/mcp-servers/qdrant-rag && ./install_global.sh"
fi

if [ "$RECENT_ERRORS" -gt 0 ]; then
    echo "• Check error logs: ./scripts/qdrant-logs --errors"
fi

echo "• Try a fresh Claude session: claude --new"
echo "• Or resume with explicit context: MCP_CLIENT_CWD=\$(pwd) claude --resume"
echo ""
echo "✨ For persistent issues, try:"
echo "   1. pkill -f qdrant_mcp_context_aware.py"
echo "   2. cd ~/Documents/repos/mcp-servers/qdrant-rag"
echo "   3. ./install_global.sh"
echo "   4. claude --new"