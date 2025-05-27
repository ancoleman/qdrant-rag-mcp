# Claude Code Integration Troubleshooting Guide

## Common Issues and Solutions

### 1. ~/.claude-code Directory Not Found

**Problem**: The `~/.claude-code` directory doesn't exist on your system.

**Solutions**:

```bash
# Option 1: Let the setup script create it
./scripts/setup.sh
# The script will detect the missing directory and create it

# Option 2: Create manually
mkdir -p ~/.claude-code
touch ~/.claude-code/config.json
touch ~/.claude-code/mcp-servers.json

# Option 3: Install Claude Code first
# Visit: https://anthropic.com/claude-code
# After installation, the directory will be created automatically
```

### 2. Claude Code Not Installed

**Problem**: Claude Code hasn't been installed on your system yet.

**Solution**:

1. Visit the official Claude Code page: https://anthropic.com/claude-code
2. Follow the installation instructions for your platform:
   - **macOS**: Download the .dmg file and drag to Applications
   - **Linux**: Use the AppImage or .deb package
   - **Windows**: Download the installer

3. After installation, run Claude Code once to create the configuration directory
4. Then run our setup script: `./scripts/setup.sh`

### 3. MCP Server Not Showing in Claude Code

**Problem**: The Qdrant RAG server doesn't appear in Claude Code's MCP servers list.

**Solutions**:

```bash
# Check if configuration exists
cat ~/.claude-code/mcp-servers.json

# If missing or incorrect, reconfigure
./install_global.sh

# Restart Claude Code after configuration changes
```

### 4. Manual Configuration

If automatic configuration doesn't work, you can manually configure Claude Code:

1. Open `~/.claude-code/mcp-servers.json` in a text editor
2. Add this configuration:

```json
{
  "servers": [
    {
      "name": "qdrant-rag",
      "type": "http",
      "config": {
        "url": "http://localhost:8080",
        "headers": {
          "Content-Type": "application/json"
        }
      },
      "start_command": "cd ~/mcp-servers/qdrant-rag && ./scripts/start_server.sh",
      "health_check": {
        "endpoint": "/health",
        "interval": 30
      },
      "auto_start": true,
      "auto_use": {
        "enabled": true,
        "triggers": {
          "code_patterns": ["implement", "create", "refactor", "similar to"],
          "config_patterns": ["configuration", "settings", "parameter"],
          "search_patterns": ["find", "search", "look for", "where is"]
        }
      }
    }
  ]
}
```

3. Restart Claude Code

### 5. Verifying the Integration

To verify Claude Code can communicate with your MCP server:

```bash
# 1. Check if MCP server is running
curl http://localhost:8080/health

# 2. Check if Qdrant is running
curl http://localhost:6333/health

# 3. In Claude Code, check MCP status
# Look for the MCP indicator in the UI
# It should show "qdrant-rag" as connected
```

### 6. Directory Structure Issues

If Claude Code was installed in a non-standard location:

```bash
# Find Claude Code config directory
find ~ -name "claude-code" -type d 2>/dev/null

# Common locations:
# macOS: ~/Library/Application Support/claude-code
# Linux: ~/.config/claude-code
# Windows: %APPDATA%\claude-code

# Update the CLAUDE_CODE_CONFIG_PATH in your .env file
echo "CLAUDE_CODE_CONFIG_PATH=/actual/path/to/claude-code" >> .env
```

### 7. Permission Issues

If you encounter permission errors:

```bash
# Fix permissions for Claude Code directory
chmod -R 755 ~/.claude-code

# Fix permissions for MCP server
chmod -R 755 ~/mcp-servers/qdrant-rag
chmod +x ~/mcp-servers/qdrant-rag/scripts/*.sh
```

### 8. Server Start Command Issues

If the MCP server doesn't start automatically:

```bash
# Update the start command in mcp-servers.json
"start_command": "/absolute/path/to/start_server.sh"

# Or use a more complex command
"start_command": "bash -c 'cd /home/user/mcp-servers/qdrant-rag && ./scripts/start_server.sh'"
```

### 9. Network/Port Conflicts

If ports are already in use:

```bash
# Check what's using the ports
lsof -i :6333  # Qdrant port
# Note: MCP server uses stdio, not a network port

# Change ports in docker-compose.yml and config files
# Update SERVER_PORT in .env
# Update the URL in mcp-servers.json
```

### 10. Docker Issues

If Docker services won't start:

```bash
# Check Docker daemon
docker ps

# Reset Docker services
docker-compose -f docker/docker-compose.yml down -v
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs
```

## Testing the Integration

### Basic Test Workflow

1. **Start all services**:
   ```bash
   cd ~/mcp-servers/qdrant-rag
   docker-compose -f docker/docker-compose.yml up -d
   ./scripts/start_server.sh
   ```

2. **Index a test file**:
   ```bash
   echo "def test_function(): return 'Hello World'" > test.py
   curl -X POST http://localhost:8080/index_code \
     -H "Content-Type: application/json" \
     -d '{"file_path": "./test.py"}'
   ```

3. **Test in Claude Code**:
   ```bash
   claude-code "Find functions that return Hello World"
   ```

4. **Check logs**:
   ```bash
   tail -f logs/rag_server.log
   ```

## Environment Variables

Make sure these are set correctly in your `.env` file:

```bash
# Required for server
QDRANT_HOST=localhost
QDRANT_PORT=6333
SERVER_PORT=8080

# Optional but recommended
LOG_LEVEL=DEBUG  # For troubleshooting
CLAUDE_CODE_CONFIG_PATH=~/.claude-code
```

## Getting Help

1. **Check logs**:
   - Server logs: `logs/rag_server.log`
   - Docker logs: `docker-compose logs`
   - Claude Code logs: Check Claude Code's help menu

2. **Verify configuration**:
   ```bash
   # Print configuration
   cat ~/.claude-code/mcp-servers.json | jq '.'
   
   # Validate JSON
   python -m json.tool ~/.claude-code/mcp-servers.json
   ```

3. **Community resources**:
   - Claude Code documentation
   - Anthropic's Discord/Forums
   - GitHub issues page

## Quick Fixes Script

Create a `scripts/fix_claude_code.sh`:

```bash
#!/bin/bash

echo "Attempting to fix Claude Code integration..."

# 1. Create directory if missing
mkdir -p ~/.claude-code

# 2. Fix permissions
chmod -R 755 ~/.claude-code

# 3. Create/update configuration
./install_global.sh

# 4. Restart services
docker-compose -f docker/docker-compose.yml restart

# 5. Test connection
curl http://localhost:8080/health

echo "Done! Please restart Claude Code to apply changes."
```

Remember: Claude Code must be restarted after any configuration changes to pick up the new settings.
