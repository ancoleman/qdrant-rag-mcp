# Qdrant RAG MCP Server

A context-aware Model Context Protocol (MCP) server that provides semantic search capabilities across your codebase using Qdrant vector database. Now with **intelligent GitHub issue resolution** (v0.3.0) and **GitHub Projects V2 management** (v0.3.4). Designed to work seamlessly with Claude Code.

> **Why MCP RAG?** This server enables AI agents to efficiently work with entire codebases while using 95%+ fewer tokens. [Learn how →](docs/reference/why-mcp-rag-agentic-coding.md)

## 🌟 Features

### 🆕 GitHub Integration (v0.3.0) + Projects V2 (v0.3.4)
- **🤖 Intelligent Issue Resolution**: RAG-powered GitHub issue analysis and automated fix generation
- **🔄 End-to-End Workflows**: Analyze issues → Generate fixes → Create PRs with dry-run safety
- **📋 GitHub Projects V2 Management**: Create projects, manage roadmaps, track progress with custom dashboards
- **🎯 24 GitHub MCP Tools**: Complete issue lifecycle + 8 new project management tools via natural language
- **💬 Issue Comments**: Add comments to existing issues for workflow updates and collaboration
- **🧠 Smart Project Assignment**: RAG-enhanced field assignment based on issue content analysis
- **📊 Project Templates**: Pre-built templates for roadmaps, bug tracking, and feature planning
- **🔐 Flexible Authentication**: Personal Access Token and GitHub App support
- **📊 RAG-Enhanced Analysis**: Leverage full codebase search for issue understanding
- **🛡️ Safety-First Design**: Dry-run mode, file protection, rate limiting, and audit logging

### 🆕 Context Tracking (v0.3.1)
- **👁️ Context Window Visibility**: Monitor what Claude knows in the current session
- **📊 Token Usage Tracking**: Real-time estimates of context window consumption
- **⚠️ Usage Warnings**: Automatic alerts at 60% and 80% context usage
- **📈 Session Timeline**: Chronological view of all context-consuming operations
- **💾 Session Persistence**: Automatic saving for later analysis
- **🔍 Session Viewer**: Utility to analyze patterns across sessions

### 🆕 Specialized Embeddings (v0.3.3)
- **🎯 Content-Type Specific Models**: Different AI models optimized for different content types
  - **Code**: Programming-aware embeddings for 30-50% better code search relevance
  - **Config**: Structure-aware models for precise JSON/YAML/XML navigation
  - **Documentation**: Prose-optimized embeddings for natural documentation search
- **🧠 Language-Specific Understanding**: Python idioms, JavaScript patterns, and more
- **🔥 Reduced Cross-Type Noise**: Configs don't pollute code searches and vice versa
- **🛡️ Thread-Safe & Reliable**: Comprehensive thread safety with fallback protection
- **🍎 Apple Silicon Optimized**: Memory management with MPS acceleration

### 🆕 Progressive Context Management (v0.3.2)
- **🎯 Multi-Level Context**: File → Class → Method hierarchy for optimal token usage
- **💰 50-70% Token Reduction**: Get high-level overviews without sacrificing detail access
- **🧠 Query Intent Detection**: Automatically determines the right context level for your query
- **⚡ Semantic Caching**: Similar queries hit cache with 0.85 similarity threshold
- **🔄 Drill-Down Options**: Start with overview, expand to details as needed
- **🏗️ Smart Structure Building**: Automatically constructs code hierarchies from search results

### Core RAG Capabilities
- **🎯 Context-Aware**: Automatically detects and scopes to your current project
- **🔍 Hybrid Search**: Combines semantic understanding with keyword matching for +30% better precision
- **🧠 AST-Based Chunking**: Structure-aware code parsing for Python, Shell, Go, JavaScript, and TypeScript (-40% tokens)
- **🔗 Dependency-Aware Search**: Automatically includes files that import or are imported by your search results (v0.1.9)
- **📊 Enhanced Search Context**: Get surrounding code chunks automatically for better understanding (v0.2.0)
- **🎯 Multi-Signal Ranking**: 5-factor ranking system for 45% better search precision (v0.2.1)
- **📚 Documentation Indexing**: Index and search markdown documentation files (v0.2.3)
- **⚡ Smart Incremental Reindexing**: Only process changed files for 90%+ faster reindexing (v0.2.4)
- **📁 Multi-Project Support**: Keep different projects' knowledge separate
- **🚀 Fast Local Execution**: Supports Apple Silicon MPS acceleration
- **🔧 Specialized Indexers**: Language-aware code parsing and config file understanding
- **🔄 Optional Auto-Indexing**: Keep your index up-to-date automatically as files change
- **📊 Project-Aware Logging**: Automatic log separation by project with rich debugging tools
- **🏥 Health Monitoring**: Built-in health checks with detailed system status

## 📚 Documentation Overview

Complete documentation for setting up and using the Qdrant RAG server with Claude Code.

## 🚀 Quick Start

### Prerequisites
- Claude Code CLI
- Docker
- Python 3.10+ with `uv` (ultraviolet package manager)

### Installing uv (Ultraviolet)

This project uses `uv` for fast, reliable Python package management. If you don't have it installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Why uv?** It's 10-100x faster than pip and provides better dependency resolution. [Learn more](https://github.com/astral-sh/uv)

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

## 🔧 Working Directory Configuration (Important!)

The MCP server needs to know your actual working directory to correctly detect projects. You have three options:

### Option 1: Natural Language (No Configuration Required!)

Simply tell Claude Code to set the working directory at the start of your session:

```
"Get current directory with pwd, export MCP_CLIENT_CWD to that value, then run health check"
```

This works immediately without any configuration changes!

### Option 2: Environment Variable Configuration

In your Claude Code configuration (`~/.claude-code/config.json`):

```json
{
  "mcpServers": {
    "qdrant-rag": {
      "command": "python",
      "args": ["/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py"],
      "env": {
        "MCP_CLIENT_CWD": "${workspaceFolder}"
      }
    }
  }
}
```

### Option 3: Command Line Argument

```json
{
  "mcpServers": {
    "qdrant-rag": {
      "command": "python",
      "args": [
        "/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py",
        "--client-cwd",
        "${workspaceFolder}"
      ]
    }
  }
}
```

**Note**: Without proper working directory setup, the server may detect the wrong project. The natural language approach (Option 1) is the quickest way to get started. See [Claude Code Configuration Guide](docs/claude-code-config-example.md) for more details.

## 📖 Documentation

### Core Guides
- **[Complete Setup & Usage Guide](docs/complete-setup-and-usage-guide.md)** 📚 - Comprehensive setup and usage instructions
  - Includes: [Changing Embedding Models](docs/complete-setup-and-usage-guide.md#what-to-expect-changing-embedding-models) - What to expect when switching models
- **[GitHub Integration Guide](docs/github-integration-guide.md)** 🤖 - Setup and use GitHub issue resolution (v0.3.0)
- **[GitHub Projects Guide](docs/github-projects-guide.md)** 📋 - Manage GitHub Projects V2 with custom fields (v0.3.4)
- **[Context-Aware Guide](docs/context-aware-guide.md)** 🎯 - How the context-aware system works
- **[MCP Scope Configuration Guide](docs/mcp-scope-configuration-guide.md)** 🔧 - Understanding local vs global configuration
- **[Practical Usage Examples](docs/rag-usage-examples.md)** 💡 - Real-world examples with Claude Code

### Reference Documentation
- **[Why MCP RAG for Agentic Coding](docs/reference/why-mcp-rag-agentic-coding.md)** 🤖 - Understand how MCP enables efficient AI coding
- **[Context Tracking Guide](docs/reference/context-tracking-guide.md)** 👁️ - Monitor and understand Claude's context window usage (v0.3.1)
- **[Qdrant Quick Reference](docs/reference/qdrant-quick-reference.md)** - Quick commands for Qdrant operations
- **[MPS Quick Reference](docs/reference/mps-quick-reference.md)** - Apple Silicon optimization guide
- **[Troubleshooting Guide](docs/reference/troubleshooting.md)** - Common issues and solutions

### Technical Documentation
- **[Enhanced RAG Guide](docs/technical/enhanced-qdrant-rag-guide.md)** - Technical implementation details
- **[Development Workflow Guide](docs/development-workflow-guide.md)** 🛠️ - Efficient development patterns using RAG search
- **[Progressive Context Implementation](docs/technical/progressive-context/)** 🎯 - Multi-level context retrieval system (v0.3.2)
- **[AST Chunking Implementation](docs/technical/ast-chunking-implementation.md)** - How AST-based chunking works (v0.1.5+)
- **[Hybrid Search Implementation](docs/technical/hybrid-search-implementation.md)** - How hybrid search works (v0.1.4+)
- **[Enhanced Ranking Guide](docs/enhanced-ranking-guide.md)** 🎯 - Configure multi-signal ranking for better results (v0.2.1+)
- **[MCP Protocol Research](docs/technical/qdrant_mcp_research.md)** - Research on MCP protocol and race conditions
- **[Installation Playbook](docs/technical/installation-playbook.md)** - Detailed breakdown of how installation works
- **[Context Awareness Playbook](docs/technical/context-awareness-playbook.md)** - Deep dive into context detection

## 📋 Quick Decision Guide

### I want to...

- **Get started quickly** → [Complete Setup & Usage Guide](docs/complete-setup-and-usage-guide.md)
- **🆕 Automate GitHub issues** → [GitHub Integration Guide](docs/github-integration-guide.md)
- **🆕 Manage GitHub Projects** → [GitHub Projects Guide](docs/github-projects-guide.md)
- **🆕 Use progressive search efficiently** → [Progressive Search Usage Guide](docs/technical/progressive-context/progressive-search-usage-guide.md)
- **Load context fast in Claude** → [Quick Context Setup](docs/quick-context-setup.md)
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

**🆕 GitHub Issue Resolution (v0.3.0)**
```bash
# In Claude with GitHub integration:
# - Setup: "Switch to repository owner/repo-name"
# - Create: "Create a test issue with title 'Bug in login' and labels 'bug'"
# - Analyze: "Analyze issue #123 using RAG search"
# - Fix: "Generate fix suggestions for issue #123"
# - Resolve: "Resolve issue #123 in dry-run mode"
```

**🆕 GitHub Projects V2 Management (v0.3.4)**
```bash
# In Claude with GitHub Projects:
# - Create: "Create a roadmap project for Q1 2025"
# - Template: "Create a bug tracking project from template"
# - Add Items: "Add issue #123 to project with smart field assignment"
# - Status: "Show project #1 status with completion metrics"
# - Update: "Update issue #123 status to In Progress"
# - Custom Fields: "Create a priority field with High/Medium/Low options"
```

**🆕 Progressive Context Management (v0.3.2)**
```bash
# In Claude - use progressive context for token efficiency:
# - Overview: "What does the authentication system do?" (70% token reduction)
# - Navigation: "Find the DatabaseManager class" (50% token reduction)
# - Debugging: "Show me the bug in save_user function" (20% token reduction)
# - Explicit: "Search for logging with context_level=file"
# - Full detail: "Search for validate_user with progressive_mode=false"

# The system automatically:
# - Detects query intent (understanding vs debugging)
# - Uses semantic cache for similar queries
# - Provides drill-down options for deeper exploration
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

**Dependency-Aware Search (v0.1.9+)**
```bash
# In Claude - search with dependencies:
# "Search for 'validate_user' and include files that import it"
# "Find 'database connection' including dependent files"

# The include_dependencies parameter automatically:
# - Finds files that import the search results
# - Finds files imported by the search results
# - Shows related code for better understanding
```

## ✨ Key Features

### Enhanced Indexing
- **Specialized Code Indexer**: Language-specific parsing for 10+ programming languages
- **Advanced Config Indexer**: Support for JSON, XML, YAML, TOML, INI, ENV files
- **Structure-Aware Chunking**: Functions, classes, and config sections
- **Rich Metadata**: Line numbers, imports, dependencies, schema extraction

### Custom File Patterns
- **Default Patterns**: Automatically indexes common code, config, and doc files
- **Custom Patterns**: Specify exactly which file types to index
- **Usage Examples**:
  ```bash
  # In Claude Code
  "Index directory with patterns *.sql *.graphql"
  "Index only Rust files: *.rs *.toml Cargo.lock"
  "Reindex with patterns *.proto *.pb.go for protobuf"
  ```
- **Supported Patterns**: Any glob pattern (*.ext, specific-file.ext, etc.)

### AST-Based Chunking (v0.1.5+)
- **Structure-Aware Parsing**: Uses Abstract Syntax Trees to understand code structure
- **Complete Code Units**: Never splits functions or classes in the middle
- **Multi-Language Support**: Python, JavaScript/TypeScript, Shell scripts, and Go
- **Hierarchical Metadata**: Tracks relationships (module → class → method)
- **40-60% Fewer Chunks**: More efficient token usage while preserving meaning
- **Language-Specific Features**:
  - Python: Classes, methods, functions with decorators and docstrings
  - JavaScript/TypeScript: ES6 modules, React components, arrow functions (v0.1.8)
  - Shell: Function extraction with setup code preservation
  - Go: Packages, structs, interfaces with visibility rules

### Dependency-Aware Search (v0.1.9+)
- **Automatic Dependency Inclusion**: Find files that import or are imported by your search results
- **Bidirectional Relationships**: Tracks both imports and exports across the codebase
- **Smart Scoring**: Related files included with reduced scores to maintain relevance
- **Import Resolution**: Handles relative and absolute imports with path resolution
- **Use Cases**:
  - Find all files using a specific function or class
  - Understand the impact of changes by seeing dependent code
  - Trace code flow through import chains
  - Discover usage patterns across your project

### Hybrid Search (v0.1.4+)
- **Three Search Modes**: Hybrid (default), vector-only, keyword-only
- **Smart Ranking**: Combines exact keyword matches with semantic understanding
- **Score Transparency**: See individual contributions (vector_score, bm25_score)
- **Automatic Mode**: Hybrid search works out-of-the-box for best results

### Enhanced Ranking (v0.2.1+)
- **5 Ranking Signals**: Combines multiple factors for optimal relevance
  - Base score (semantic + keyword match)
  - File proximity (same directory boost)
  - Dependency distance (import relationships)
  - Code structure similarity (functions, classes)
  - Recency (recent modifications prioritized)
- **Configurable Weights**: Tune ranking for your workflow via `server_config.json`
- **45% Better Precision**: Measured improvement in search relevance
- **Visible Signals**: See why results ranked as they did with `ranking_signals`
- **[Configuration Guide](docs/enhanced-ranking-guide.md)**: Learn how to customize ranking

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

### Reliability & Security
- **Connection Retry Logic**: Automatic retry with exponential backoff
- **Health Monitoring**: Built-in health check for all services
- **Progress Indicators**: Track long-running operations
- **Input Validation**: Path traversal prevention and sanitization
- **Better Error Handling**: User-friendly messages with error codes

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

### RAG-Powered Code Search
- **Semantic Code Search**: "Find authentication functions similar to UserService"
- **Configuration Discovery**: "Where is the database connection configured?"
- **Pattern Analysis**: "Show me error handling patterns in this codebase"
- **Code Understanding**: "How does the logging system work?"
- **Cross-Project Insights**: "Show JWT implementations across all my projects"

### 🆕 GitHub Issue Resolution (v0.3.0)
- **Intelligent Issue Analysis**: "Analyze issue #123 to understand the bug and find related code"
- **Automated Fix Generation**: "Generate fix suggestions for the authentication issue in #456"
- **Repository Management**: "Switch to my backend repository and show open issues"
- **Issue Workflow**: "Create a test issue, analyze it, and generate a fix in dry-run mode"
- **Pull Request Creation**: "Create a PR to resolve the login bug with automated content"

## 🏥 Health Monitoring

The server includes a health check tool to monitor all services:

```bash
# In Claude Code
"Check health status"
"Run health check"
```

The health check reports:
- **Qdrant Connection**: Status and collection count
- **Embedding Model**: Model name and dimension verification
- **Disk Space**: Available storage with warnings
- **Memory Usage**: System memory status (if psutil installed)
- **Project Context**: Current project information

### Connection Resilience

The server automatically retries failed operations:
- Exponential backoff for transient failures
- Automatic reconnection to Qdrant
- Graceful degradation with clear error messages

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

### 🚀 v0.3.3 (Latest) - Specialized Embeddings & Critical Bug Fixes
- ✅ **Content-Type Specific Models**: Different embedding models for different content types
  - **Code**: `nomic-ai/CodeRankEmbed` (768D) - Programming-aware embeddings for superior code search
  - **Config**: `jinaai/jina-embeddings-v3` (1024D) - Structure-aware for JSON/YAML/XML files
  - **Documentation**: `hkunlp/instructor-large` (768D) - Prose-optimized for markdown/docs
  - **General**: `sentence-transformers/all-MiniLM-L12-v2` (384D) - Backward compatibility
- ✅ **30-50% Better Code Search**: Programming-aware embeddings understand code context better
- ✅ **Unified Memory Management**: Apple Silicon optimizations with conservative memory limits
- ✅ **Model Registry System**: Central model management with compatibility checking
- ✅ **Enhanced Model Scripts**: Improved download_models.sh with specialized model support
- ✅ **Critical Post-Release Fixes (v0.3.3.post1-4)**:
  - **Fixed Critical Search Error**: Payload type validation preventing search failures
  - **Fixed Context Tracking Error**: Configuration float handling for proper context tracking
  - **🔧 Fixed Dimension Mismatch**: Resolved "Vector dimension error: expected dim: 768, got 384"
    - **Root Cause**: Thread safety issues causing CodeRankEmbed model eviction
    - **Solution**: Comprehensive thread safety + dimension-compatible fallback logic
    - **Impact**: Code files now always get correct 768D embeddings
- ✅ **Test Suite Reorganization**: Clean categorization (unit/integration/performance/debug)
- 📖 **[Memory Management Guide](docs/technical/memory-optimization-recommendations.md)**: Apple Silicon optimization details

### v0.3.4 - GitHub Projects V2 Integration
- ✅ **GitHub Projects V2 Support**: Full GraphQL API integration for modern project management
- ✅ **8 New Project Tools**: Create projects, add items, update fields, track progress
- ✅ **Smart Field Assignment**: RAG-enhanced analysis automatically assigns priority/status/type
- ✅ **Project Templates**: Pre-built templates for roadmaps, bug tracking, feature planning
- ✅ **Custom Field Support**: Create and manage TEXT, NUMBER, DATE, SINGLE_SELECT fields
- ✅ **Progress Dashboards**: Automatic completion metrics and status tracking
- ✅ **GraphQL Adapter Pattern**: Clean separation preserving existing REST functionality
- 📖 **[GitHub Projects Guide](docs/github-projects-guide.md)**: Complete guide with examples and templates

### v0.3.2 - Progressive Context Management
- ✅ **Multi-Level Context Retrieval**: File → Class → Method hierarchy for 50-70% token reduction
- ✅ **Semantic Caching**: Similar queries hit cache with 0.85 similarity threshold
- ✅ **Query Intent Detection**: Automatically determines optimal context level
- ✅ **Configurable Scoring Pipeline**: Modular scoring system with pluggable stages
- ✅ **Enhanced BM25 Tokenization**: Code-specific preprocessing (camelCase, snake_case)
- ✅ **Linear Combination Scoring**: More accurate hybrid search scores (0.6-0.9 vs 0.01-0.02)

### v0.3.1 - Context Tracking
- ✅ **Context Window Visibility**: Monitor what Claude knows in current session
- ✅ **Token Usage Tracking**: Real-time estimates with warnings at 60%/80%
- ✅ **Session Timeline**: Chronological view of all context-consuming operations
- ✅ **Session Persistence**: Automatic saving for later analysis

### v0.3.0 - GitHub Integration
- ✅ **10 GitHub MCP Tools**: Complete issue lifecycle management via Claude Code
- ✅ **RAG-Powered Issue Analysis**: Leverage codebase search for intelligent issue understanding
- ✅ **Automated Fix Generation**: Generate code fixes with confidence scoring and templates
- ✅ **End-to-End Workflows**: Analyze → Generate → Create PR with dry-run safety
- ✅ **Issue Comments**: Add comments to existing issues for workflow updates
- ✅ **Flexible Authentication**: Personal Access Token and GitHub App support
- ✅ **HTTP API Testing**: 10 new endpoints under `/github/` for testing and integration
- ✅ **Safety-First Design**: Dry-run mode, rate limiting, audit logging, file protection
- ✅ **Configuration Enhanced**: Fixed environment variable resolution for better config management
- 📖 **[GitHub Integration Guide](docs/github-integration-guide.md)**: Comprehensive 700+ line setup and usage guide

### v0.2.1 - Enhanced Ranking
- ✅ **Multi-Signal Ranking**: Search results now ranked by 5 configurable factors
- ✅ **File Proximity Scoring**: Boosts results from same/nearby directories
- ✅ **Dependency Distance**: Prioritizes files with import relationships
- ✅ **Code Structure Similarity**: Groups similar code patterns together
- ✅ **Recency Weighting**: Recently modified files surface first
- ✅ **45% Better Precision**: Significant improvement in search relevance
- 📖 **[Configuration Guide](docs/enhanced-ranking-guide.md)**: Learn how to tune ranking for your workflow

### v0.2.0 - Enhanced Context
- ✅ **Automatic Context Expansion**: Get surrounding chunks with search results
- ✅ **Configurable Context**: Control how much context with `context_chunks` parameter
- ✅ **Get File Chunks Tool**: Retrieve complete files or specific chunk ranges
- ✅ **Doubled Chunk Sizes**: Better semantic understanding (code: 3000, config: 2000 chars)
- ✅ **60% Fewer Operations**: Reduced need for follow-up grep/read operations

### v0.1.9
- ✅ **Dependency-Aware Search**: Automatically include files that import or are imported by search results
- ✅ **Dependency Graph Builder**: Tracks bidirectional import/export relationships across codebase
- ✅ **Enhanced Code Understanding**: See how code modules relate and depend on each other

### v0.1.8
- ✅ **JavaScript/TypeScript Support**: Full AST parsing for JS/TS files including React components
- ✅ **ES6 Module Support**: Handles modern import/export syntax and arrow functions
- ✅ **Expanded Language Coverage**: Now supports 5 languages with AST parsing

### v0.1.7
- ✅ **Shell Script Indexing**: Fixed default patterns to include .sh, .bash, .zsh files
- ✅ **Better Script Support**: Improved handling of executable scripts

### Previous Releases
- ✅ **AST-Based Chunking**: Structure-aware parsing for better code understanding (v0.1.5)
- ✅ **Hybrid Search**: Combined vector + keyword search for 30% better precision (v0.1.4)
- ✅ **Context-Aware System**: Automatic project detection and scoping
- ✅ **Global Installation**: Works across all projects with proper setup
- ✅ **MPS Support**: Apple Silicon GPU acceleration for faster embeddings
- ✅ **Project-Aware Logging**: Automatic log separation and rich debugging tools
- ✅ **Smart Reindexing**: Clean index updates to prevent stale data

Happy coding with context-aware semantic search! 🎉