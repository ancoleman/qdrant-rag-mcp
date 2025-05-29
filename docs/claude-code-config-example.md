# Claude Code Configuration for Qdrant RAG MCP Server

This guide shows how to properly configure the Qdrant RAG MCP server in Claude Code to ensure correct working directory context.

## Configuration Options

### Option 1: Using Environment Variables (Recommended)

Add to your Claude Code configuration (`~/.claude-code/config.json` or `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qdrant-rag": {
      "command": "python",
      "args": [
        "/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py"
      ],
      "env": {
        "MCP_CLIENT_CWD": "${workspaceFolder}",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "EMBEDDING_MODEL": "all-MiniLM-L12-v2"
      }
    }
  }
}
```

The key is setting `MCP_CLIENT_CWD` to `${workspaceFolder}`, which Claude Code will replace with the actual workspace directory.

### Option 2: Using Working Directory (cwd)

```json
{
  "mcpServers": {
    "qdrant-rag": {
      "command": "python",
      "args": [
        "/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py"
      ],
      "cwd": "${workspaceFolder}",
      "env": {
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

However, this changes the server's working directory, which might affect where it finds its configuration files.

### Option 3: Command Line Arguments (Future Enhancement)

We could modify the server to accept working directory as a command-line argument:

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

## Available Variables in Claude Code

Claude Code supports these variables in configuration:
- `${workspaceFolder}` - The path of the folder opened in Claude Code
- `${workspaceFolderBasename}` - The name of the folder opened in Claude Code without any slashes
- `${file}` - The current opened file
- `${fileBasename}` - The current opened file's basename
- `${fileDirname}` - The current opened file's dirname
- `${fileExtname}` - The current opened file's extension
- `${cwd}` - The task runner's current working directory on startup
- `${lineNumber}` - The current selected line number in the active file
- `${selectedText}` - The current selected text in the active file
- `${execPath}` - The location of the Claude Code executable
- `${pathSeparator}` - `/` on macOS or linux, `\` on Windows

## Troubleshooting

### Verifying Configuration

1. Run a health check and look for the `client_cwd` field:
   ```
   > run health check
   ```

2. Check if the project detection is correct:
   ```
   > get context
   ```

### Common Issues

1. **Wrong project detected**: Ensure `MCP_CLIENT_CWD` is set to `${workspaceFolder}`
2. **Server can't find config files**: Use absolute paths in the configuration
3. **Environment variables not working**: Check Claude Code logs for variable expansion

## Best Practices

1. Always use `${workspaceFolder}` for the `MCP_CLIENT_CWD` environment variable
2. Use absolute paths for the server command and arguments
3. Test the configuration with a health check after changes
4. Keep server configuration files in a fixed location (not relative to working directory)

## Example: Multi-Project Setup

If you work with multiple projects, you can create project-specific configurations:

```json
{
  "mcpServers": {
    "qdrant-rag-projectA": {
      "command": "python",
      "args": ["/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py"],
      "env": {
        "MCP_CLIENT_CWD": "/path/to/projectA",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    },
    "qdrant-rag-projectB": {
      "command": "python",
      "args": ["/path/to/qdrant-rag/src/qdrant_mcp_context_aware.py"],
      "env": {
        "MCP_CLIENT_CWD": "/path/to/projectB",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333"
      }
    }
  }
}
```

This ensures each project has its own properly scoped MCP server instance.