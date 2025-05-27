#!/bin/bash
# Quick context loader for Claude Code sessions
# Ensures the project is fully indexed and ready for editing

set -e

echo "🚀 Preparing Claude Code session with full context..."
echo ""

# Get the directory to index (default to current)
TARGET_DIR="${1:-.}"
cd "$TARGET_DIR"

echo "📍 Project: $(basename $(pwd))"
echo "📂 Path: $(pwd)"
echo ""

# Check if Claude Code is available
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code CLI not found. Please install it first."
    exit 1
fi

# Create a context file with project overview
CONTEXT_FILE=".claude-context.md"
echo "📝 Generating project context..."

cat > "$CONTEXT_FILE" << EOF
# Project Context for Claude Code

## Project: $(basename $(pwd))
**Path**: $(pwd)
**Generated**: $(date)

## Quick Start Commands

1. **Check current context**: "What is my current project context?"
2. **Search codebase**: "Search for [term] in the codebase"
3. **Reindex if needed**: "Reindex this directory"
4. **Health check**: "Run health check"

## Project Structure
\`\`\`
$(find . -type f -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.yaml" -o -name "*.yml" 2>/dev/null | grep -v node_modules | grep -v .git | grep -v __pycache__ | sort | head -20)
...
\`\`\`

## Recent Changes
\`\`\`
$(git log --oneline -10 2>/dev/null || echo "No git history")
\`\`\`

## Key Files
$(find . -name "README.md" -o -name "package.json" -o -name "pyproject.toml" -o -name "Cargo.toml" -o -name "go.mod" 2>/dev/null | grep -v node_modules | head -10)

---
*This file is auto-generated. Add to .gitignore if needed.*
EOF

echo "✅ Context file created: $CONTEXT_FILE"
echo ""

# Start Claude with auto-indexing
echo "🔄 Starting Claude Code with auto-indexing..."
echo ""
echo "💡 First message suggestions:"
echo "   - 'Reindex this directory and show me the project structure'"
echo "   - 'What are the main components of this codebase?'"
echo "   - 'Search for the main entry point'"
echo ""

# Set auto-indexing for this session
export QDRANT_RAG_AUTO_INDEX=true
export QDRANT_RAG_DEBOUNCE=3.0

# Start Claude
exec claude "$@"