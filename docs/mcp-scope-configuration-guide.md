# MCP Scope Configuration Guide

This guide explains how to properly configure the Qdrant RAG MCP Server for different usage scenarios using Claude Code's scope system.

## Understanding MCP Configuration Scopes

Claude Code supports three configuration scopes for MCP servers:

### 1. Local Scope (Default)
- **Availability**: Only in the current project/directory where you add it
- **Storage**: Project-specific user settings
- **Use Case**: Testing or project-specific servers
- **Command**: `claude mcp add <name> <command>` (no scope flag)

### 2. Project Scope
- **Availability**: For all team members working on the project
- **Storage**: `.mcp.json` file in project root
- **Use Case**: Shared team servers
- **Command**: `claude mcp add <name> -s project <command>`

### 3. User Scope (Global)
- **Availability**: Across ALL projects on your machine
- **Storage**: User-level configuration
- **Use Case**: Personal tools you want everywhere
- **Command**: `claude mcp add <name> -s user <command>`

## Installation Instructions

### For Global Usage (Recommended)

If you want the Qdrant RAG server available in all your projects:

```bash
# 1. First, run the installation script to set up the global runner
cd /Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag
./install_global_cli.sh

# 2. Remove any existing local-scoped server
claude mcp remove qdrant-rag

# 3. Add with user scope for global availability
claude mcp add qdrant-rag -s user ~/.mcp-servers/qdrant-rag-global.sh

# 4. Verify it's available globally
claude mcp list -s user
```

### For Project-Specific Usage

If you only want it in the current project:

```bash
# Add to current project only (local scope)
claude mcp add qdrant-rag /path/to/qdrant-rag/run_mcp.sh

# Or share with your team (project scope)
claude mcp add qdrant-rag -s project /path/to/qdrant-rag/run_mcp.sh
```

## Verifying Your Configuration

### Check Scope-Specific Servers

```bash
# List all servers in current context
claude mcp list

# List user-scoped (global) servers
claude mcp list -s user

# List project-scoped servers
claude mcp list -s project
```

### Understanding the Output

When you run `claude mcp list` in different directories:

1. **In the original project**: Shows local + project + user servers
2. **In a different project**: Shows only user-scoped servers (unless that project has its own)
3. **Outside any project**: Shows only user-scoped servers

## Common Issues and Solutions

### Issue: "No MCP servers configured" in other projects

**Cause**: Server was added with local scope (default)

**Solution**: Remove and re-add with user scope:
```bash
claude mcp remove qdrant-rag
claude mcp add qdrant-rag -s user ~/.mcp-servers/qdrant-rag-global.sh
```

### Issue: Server works in one project but not another

**Cause**: Local scope configuration

**Solution**: Use user scope for global availability

### Issue: Team can't see the MCP server

**Cause**: Using local or user scope instead of project scope

**Solution**: Add with project scope and commit `.mcp.json`:
```bash
claude mcp add qdrant-rag -s project ./run_mcp.sh
git add .mcp.json
git commit -m "Add Qdrant RAG MCP server for team"
```

## Best Practices

### 1. Choose the Right Scope

- **User Scope**: For personal productivity tools you want everywhere
- **Project Scope**: For project-specific tools the whole team needs
- **Local Scope**: For testing or personal project-specific needs

### 2. Global Context-Aware Setup

For the Qdrant RAG server, **user scope** is recommended because:
- It automatically detects project context
- You want it available in all your projects
- Each project's data stays separate

### 3. Environment Variables

When adding servers, you can include environment variables:
```bash
claude mcp add qdrant-rag -s user -e EMBEDDING_MODEL=all-mpnet-base-v2 ~/.mcp-servers/qdrant-rag-global.sh
```

## Quick Reference

| Scope | Command Flag | Available In | Use Case |
|-------|-------------|--------------|----------|
| Local | (none) | Current project only | Testing |
| Project | `-s project` | All team members in project | Team tools |
| User | `-s user` | All your projects | Personal tools |

## Migration Guide

If you've already added the server locally and want to make it global:

```bash
# 1. Check current configuration
claude mcp list

# 2. Remove local instance
claude mcp remove qdrant-rag

# 3. Add with user scope
claude mcp add qdrant-rag -s user ~/.mcp-servers/qdrant-rag-global.sh

# 4. Verify in different directories
cd ~/some-other-project
claude mcp list  # Should show qdrant-rag
```

## Summary

The key to global MCP server configuration is the **`-s user`** flag. Without it, servers are only available in the directory where you added them. For a context-aware RAG server that should work across all your projects, user scope is the correct choice.