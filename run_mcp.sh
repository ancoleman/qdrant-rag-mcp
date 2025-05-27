#!/bin/bash
# Context-aware runner that preserves working directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# IMPORTANT: Save current directory for context detection
CURRENT_DIR="$(pwd)"

# Change to script directory to load environment
cd "$SCRIPT_DIR"

# Export environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# CRITICAL: Return to original directory so Python sees correct context
cd "$CURRENT_DIR"

# Run the context-aware server with any passed arguments
exec uv run --directory "$SCRIPT_DIR" python "$SCRIPT_DIR/src/qdrant_mcp_context_aware.py" "$@"