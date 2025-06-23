#!/bin/bash
# Unified global installation script for Qdrant RAG MCP Server
# Maintains context awareness with optional file watching

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GLOBAL_DIR="$HOME/.mcp-servers"
GLOBAL_SCRIPT="$GLOBAL_DIR/qdrant-rag-global.sh"

echo "🚀 Qdrant RAG MCP Server - Global Installation"
echo "============================================="
echo ""

# Create global directory
mkdir -p "$GLOBAL_DIR"

# Check if qdrant-rag already exists in ~/.mcp-servers
if [ -L "$GLOBAL_DIR/qdrant-rag" ] || [ -d "$GLOBAL_DIR/qdrant-rag" ]; then
    echo "📦 Updating existing installation..."
else
    echo "📦 Creating new installation..."
    ln -sf "$PROJECT_ROOT" "$GLOBAL_DIR/qdrant-rag"
fi

# Create the global runner script
cat > "$GLOBAL_SCRIPT" << 'EOF'
#!/bin/bash
# Global context-aware runner for Qdrant RAG MCP Server
# Supports optional file watching via environment variable

# Get the real path of the MCP server (macOS compatible)
if [[ "$OSTYPE" == "darwin"* ]]; then
    MCP_SERVER_DIR="$(cd "$(dirname "$HOME/.mcp-servers/qdrant-rag")" && pwd -P)/$(basename "$HOME/.mcp-servers/qdrant-rag")"
    MCP_SERVER_DIR="$(cd "$MCP_SERVER_DIR" && pwd -P)"
else
    MCP_SERVER_DIR="$(readlink -f "$HOME/.mcp-servers/qdrant-rag")"
fi

# Clean up any stale processes before starting
# This prevents connection issues from multiple server instances
for PID in $(ps aux | grep -E "qdrant_mcp_context_aware.py" | grep -v grep | grep -v $$ | awk '{print $2}'); do
    kill -9 $PID 2>/dev/null
done

# IMPORTANT: Save current directory for context detection
CURRENT_DIR="$(pwd)"

# Change to server directory to load environment
cd "$MCP_SERVER_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# CRITICAL: Return to original directory so Python sees correct context
cd "$CURRENT_DIR"

# Check if auto-indexing is requested via environment variable
if [ "$QDRANT_RAG_AUTO_INDEX" = "true" ] || [ "$QDRANT_RAG_AUTO_INDEX" = "1" ]; then
    # Run with auto-indexing
    echo "🔄 Starting with auto-indexing enabled..." >&2
    exec uv run --directory "$MCP_SERVER_DIR" python "$MCP_SERVER_DIR/src/qdrant_mcp_context_aware.py" \
        --watch \
        --watch-dir "$CURRENT_DIR" \
        --debounce "${QDRANT_RAG_DEBOUNCE:-3.0}" \
        --initial-index
else
    # Run normally
    exec uv run --directory "$MCP_SERVER_DIR" python "$MCP_SERVER_DIR/src/qdrant_mcp_context_aware.py" "$@"
fi
EOF

chmod +x "$GLOBAL_SCRIPT"

# Update or add to Claude Code
echo ""
echo "🔧 Configuring Claude Code..."

# Check if claude command exists
if ! command -v claude &> /dev/null; then
    echo "❌ Claude CLI not found!"
    echo ""
    echo "Please install Claude Code first, then manually run:"
    echo "  claude mcp add qdrant-rag -s user $GLOBAL_SCRIPT"
    echo ""
    exit 0
fi

# Remove any existing configuration
echo "Removing any existing qdrant-rag configuration..."
if claude mcp remove qdrant-rag 2>&1 | grep -q "not found"; then
    echo "  No existing configuration found"
else
    echo "  Removed existing configuration"
fi

# Add with global scope
echo "Adding qdrant-rag with global scope..."
if claude mcp add qdrant-rag -s user "$GLOBAL_SCRIPT"; then
    echo "  ✅ Successfully configured!"
else
    echo "  ❌ Failed to configure. Please run manually:"
    echo "     claude mcp add qdrant-rag -s user $GLOBAL_SCRIPT"
fi

# Verify installation
echo ""
echo "Verifying installation..."
if claude mcp list 2>/dev/null | grep -q "qdrant-rag.*$GLOBAL_SCRIPT"; then
    echo "  ✅ MCP server correctly configured"
else
    echo "  ⚠️  MCP server may not be configured correctly"
    echo "  Please check with: claude mcp list"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "📍 IMPORTANT: This is a GLOBAL installation"
echo "   • Works in ALL projects automatically"
echo "   • No need to run this script again"
echo "   • The MCP server is available everywhere"
echo ""
echo "📍 Usage:"
echo ""
echo "1. Normal mode (manual indexing):"
echo "   Just use Claude Code normally in ANY project"
echo ""
echo "2. With auto-indexing:"
echo "   Option A - Per session:"
echo "   export QDRANT_RAG_AUTO_INDEX=true"
echo "   claude"
echo ""
echo "   Option B - Always on:"
echo "   Add to your ~/.bashrc or ~/.zshrc:"
echo "   export QDRANT_RAG_AUTO_INDEX=true"
echo "   export QDRANT_RAG_DEBOUNCE=5.0  # Optional: custom debounce"
echo ""
echo "3. Test current project context:"
echo "   In Claude Code, ask: 'What is my current project context?'"
echo ""

# Check if watchdog is installed
if ! uv pip show watchdog >/dev/null 2>&1; then
    echo "⚠️  Note: Watchdog not installed. Auto-indexing won't work."
    echo "   Install with: uv pip install watchdog"
    echo ""
fi