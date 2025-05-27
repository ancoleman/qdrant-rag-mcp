# Qdrant RAG MCP Server

A context-aware Model Context Protocol (MCP) server that provides semantic search capabilities across your codebase using Qdrant vector database. Designed to work seamlessly with Claude Code.

## 🌟 Features

- **🎯 Context-Aware**: Automatically detects and scopes to your current project
- **🔍 Semantic Search**: Find code by meaning, not just keywords
- **📁 Multi-Project Support**: Keep different projects' knowledge separate
- **🚀 Fast Local Execution**: Supports Apple Silicon MPS acceleration
- **🔧 Specialized Indexers**: Language-aware code parsing and config file understanding
- **🔄 Optional Auto-Indexing**: Keep your index up-to-date automatically as files change

## 📚 Documentation Overview

Complete documentation for setting up and using the Qdrant RAG server with Claude Code.

## 🚀 Quick Start

### Prerequisites
- Claude Code CLI
- Docker
- Python 3.10+ with `uv`

### Installation

```bash
# 1. Clone the repository
git clone <repository-url> ~/mcp-servers/qdrant-rag
cd ~/mcp-servers/qdrant-rag

# 2. Run setup (starts Qdrant, installs dependencies)
./scripts/setup.sh

# 3. Install globally with context awareness (ONE TIME ONLY)
./install_global.sh

# That's it! The MCP server is now available in ALL projects

# 4. Test in any project
cd ~/any-project
claude
# Ask: "What's my current project context?"
```

### Optional: Enable Auto-Indexing

```bash
# For current session only
export QDRANT_RAG_AUTO_INDEX=true
claude

# For permanent auto-indexing
echo 'export QDRANT_RAG_AUTO_INDEX=true' >> ~/.bashrc
# or for zsh users:
echo 'export QDRANT_RAG_AUTO_INDEX=true' >> ~/.zshrc
```

## 📖 Documentation

### Core Guides
- **[Complete Setup & Usage Guide](docs/complete-setup-and-usage-guide.md)** 📚 - Comprehensive setup and usage instructions
- **[Context-Aware Guide](docs/context-aware-guide.md)** 🎯 - How the context-aware system works
- **[MCP Scope Configuration Guide](docs/mcp-scope-configuration-guide.md)** 🔧 - Understanding local vs global configuration
- **[Practical Usage Examples](docs/rag-usage-examples.md)** 💡 - Real-world examples with Claude Code

### Reference Documentation
- **[Qdrant Quick Reference](docs/reference/qdrant-quick-reference.md)** - Quick commands for Qdrant operations
- **[MPS Quick Reference](docs/reference/mps-quick-reference.md)** - Apple Silicon optimization guide
- **[Troubleshooting Guide](docs/reference/troubleshooting.md)** - Common issues and solutions

### Technical Documentation
- **[Enhanced RAG Guide](docs/technical/enhanced-qdrant-rag-guide.md)** - Technical implementation details
- **[MCP Protocol Research](docs/technical/qdrant_mcp_research.md)** - Research on MCP protocol and race conditions
- **[Installation Playbook](docs/technical/installation-playbook.md)** - Detailed breakdown of how installation works
- **[Context Awareness Playbook](docs/technical/context-awareness-playbook.md)** - Deep dive into context detection

## 📋 Quick Decision Guide

### I want to...

- **Get started quickly** → [Complete Setup & Usage Guide](docs/complete-setup-and-usage-guide.md)
- **Use RAG in ALL my projects** → [MCP Scope Configuration Guide](docs/mcp-scope-configuration-guide.md)
- **Understand project isolation** → [Context-Aware Guide](docs/context-aware-guide.md)
- **See practical examples** → [Usage Examples](docs/rag-usage-examples.md)
- **Troubleshoot issues** → [Troubleshooting Guide](docs/reference/troubleshooting.md)
- **Optimize for Apple Silicon** → [MPS Quick Reference](docs/reference/mps-quick-reference.md)

## ⚡ Quick Commands

### Global Setup (One-Time)
```bash
# Run the installer - it handles everything
./install_global.sh
```

### Daily Usage

**Manual Indexing (Default)**
```bash
# Navigate to any project
cd ~/projects/my-app
claude

# In Claude:
# - Index: "Index all code files in this project"
# - Search: "Find authentication functions"
# - Context: "What's my current project context?"
```

**With Auto-Indexing**
```bash
# Enable for this session
export QDRANT_RAG_AUTO_INDEX=true
cd ~/projects/my-app
claude

# Files are automatically indexed as you work!
# Just search - no manual indexing needed
```

**Reindexing (Clean Index)**
```bash
# Use reindex when files have been:
# - Renamed or moved
# - Deleted
# - You see stale results

# In Claude:
# "Reindex this project" - Clears old data before indexing
# "Reindex the src directory" - Clean reindex of specific directory

# Regular index only adds new content
# Reindex removes old + adds new content
```

## ✨ Key Features

### Enhanced Indexing
- **Specialized Code Indexer**: Language-specific parsing for 10+ programming languages
- **Advanced Config Indexer**: Support for JSON, XML, YAML, TOML, INI, ENV files
- **Structure-Aware Chunking**: Functions, classes, and config sections
- **Rich Metadata**: Line numbers, imports, dependencies, schema extraction

### Context Awareness
- **Automatic Project Detection**: Based on .git, package.json, etc.
- **Project Isolation**: Each project gets separate collections
- **Smart Scoping**: Searches default to current project only
- **Cross-Project Option**: Available when needed

### Apple Silicon Optimization
- **MPS Acceleration**: Metal Performance Shaders support for M1/M2/M3 chips
- **Local Mode**: Native macOS execution for maximum performance
- **Smart Caching**: Efficient model storage and loading

### Project-Aware Logging
- **Automatic Log Separation**: Logs organized by project, no mixing
- **Structured JSON Format**: Rich metadata for every operation
- **Performance Tracking**: Operation timing and success metrics
- **Log Viewer Utility**: Search, filter, and tail logs easily
- **Configurable Levels**: Debug specific operations as needed

## 🔧 Architecture

```
Claude Code ←→ MCP Protocol ←→ Context-Aware RAG Server ←→ Qdrant Vector DB
                                        ↓
                               Project Detection
                                        ↓
                         Project-Specific Collections
                                        ↓
                              Specialized Indexers
                               (Code + Config)
```

### Server Components

This project includes two distinct servers:

1. **MCP Server** (`src/qdrant_mcp_context_aware.py`) - The main server
   - Integrates directly with Claude Code via stdio
   - No network port - communicates via stdin/stdout
   - Started automatically by Claude Code when needed
   - This is what you use for normal operation

2. **HTTP Test Server** (`src/http_server.py`) - Optional testing interface
   - Provides REST API endpoints on port 8081
   - Only for testing indexing/search without Claude Code
   - Not required for normal Claude Code usage
   - Useful for debugging and integration testing

## 🎯 Use Cases

### With Claude Code
- **Semantic Code Search**: "Find authentication functions similar to UserService"
- **Configuration Discovery**: "Where is the database connection configured?"
- **Pattern Analysis**: "Show me error handling patterns in this codebase"
- **Code Understanding**: "How does the logging system work?"
- **Cross-Project Insights**: "Show JWT implementations across all my projects"

## 📊 Logging & Debugging

The server includes comprehensive project-aware logging for debugging and monitoring.

### View Logs

```bash
# View logs for current project
./scripts/qdrant-logs

# Follow logs in real-time
./scripts/qdrant-logs -f

# Filter by log level
./scripts/qdrant-logs --level ERROR

# Search logs
./scripts/qdrant-logs --search "index.*failed"

# View logs for specific project
./scripts/qdrant-logs --project /path/to/project

# Export logs for analysis
./scripts/qdrant-logs --export json > debug-logs.json
```

### Log Location

Logs are stored in `~/.mcp-servers/qdrant-rag/logs/`:
- `global/` - Server startup and non-project operations
- `projects/` - Separated by project with friendly names (e.g., `qdrant-rag_70e24d/`)
- `errors/` - Critical errors across all projects

### Configuration

Control logging via environment variables:
```bash
export QDRANT_LOG_LEVEL=DEBUG        # Set log level
export QDRANT_LOG_DIR=/custom/path   # Custom log directory
```

## 🐛 Common Issues & Solutions

| Issue | Solution | Reference |
|-------|----------|-----------|
| MCP not available globally | Add with `-s user` flag | [MCP Scope Guide](mcp-scope-configuration-guide.md) |
| Wrong project detected | Check project markers | [Context-Aware Guide](context-aware-guide.md) |
| MPS not working | Use local mode | [MPS Guide](mps-quick-reference.md) |
| No search results | Index first, check scope | [Troubleshooting](claude-code-troubleshooting.md) |

## 📈 Recent Improvements

- ✅ **Context-Aware System**: Automatic project detection and scoping
- ✅ **Global Installation**: Works across all projects with proper setup
- ✅ **Integrated Specialized Indexers**: Enhanced code and config parsing
- ✅ **Added MPS Support**: Apple Silicon GPU acceleration
- ✅ **Optional Auto-Indexing**: File watching integrated into MCP server
- ✅ **Unified Installation**: Single script handles all configuration
- ✅ **Comprehensive Documentation**: Multiple guides for different use cases

Happy coding with context-aware semantic search! 🎉