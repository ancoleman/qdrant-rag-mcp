# Quick Context Setup for Claude Code Sessions

## 🎯 Fastest Way to Get Full Context

### Option 1: One-Line Setup (Recommended)
Start your Claude Code session with this single command:
```
"Reindex this directory, then search for the main files and give me an overview of the project structure"
```

This will:
1. Clear any stale data
2. Index all current files
3. Search and summarize key components
4. Give you a project overview

### Option 2: Auto-Index on Start
Add to your shell config (`~/.bashrc` or `~/.zshrc`):
```bash
# Auto-index every Claude session
export QDRANT_RAG_AUTO_INDEX=true
alias claude-dev='QDRANT_RAG_AUTO_INDEX=true claude'
```

### Option 3: Use the Session Starter
```bash
# From any project directory
~/path/to/qdrant-rag/scripts/start-session.sh
```

## 📋 Essential First Commands

After starting Claude Code, use these commands for instant context:

1. **Full Reindex + Overview**:
   ```
   "Reindex the current directory and tell me about the project structure, main files, and recent changes"
   ```

2. **Quick Health Check**:
   ```
   "Run a health check and show me the current project context"
   ```

3. **Find Key Files**:
   ```
   "Search for README, main, index, app, and config files to understand the project"
   ```

4. **Understand Dependencies**:
   ```
   "Search for package.json, requirements.txt, pyproject.toml, or go.mod and summarize the dependencies"
   ```

## 🚀 Pro Tips

### Create a Project-Specific Alias
Add to your project's README or documentation:
```bash
# Quick start for this project
alias claude-this='cd /path/to/project && QDRANT_RAG_AUTO_INDEX=true claude'
```

### Use Directory-Specific Environment
Create `.envrc` in your project (using direnv):
```bash
# .envrc
export QDRANT_RAG_AUTO_INDEX=true
export QDRANT_RAG_DEBOUNCE=2.0
```

### Combine with Git
Start with recent changes context:
```
"Reindex this directory, then search for files modified in the last week and summarize recent changes"
```

## 🎪 Advanced Context Loading

For large projects, index strategically:
```
"First index just the src directory, then search for the main entry points"
"Index all Python files and search for class definitions"
"Index configuration files and tell me about the project setup"
```

## 💡 Context Verification

Always verify your context is loaded:
```
"What files are currently indexed for this project?"
"Show me the health check status"
"Search for a known file to verify indexing worked"
```