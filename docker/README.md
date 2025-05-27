# Docker Configuration

This directory contains Docker configurations for the Qdrant RAG MCP Server.

## Current Setup

With the context-aware MCP server, the typical setup is:
- **Qdrant**: Runs in Docker
- **MCP Server**: Runs locally via Claude CLI for context detection

## Files

### `docker-compose-qdrant-only.yml`
Simplified compose file that only runs Qdrant. This is what you'll typically use.

```bash
# Start Qdrant
docker-compose -f docker-compose-qdrant-only.yml up -d

# Or use the helper script
./start_qdrant.sh
```

### `docker-compose.yml`
Full compose file with optional RAG server service (commented out). The RAG server section is kept for:
- Testing the server in isolation
- Running in environments where local execution isn't possible
- Development/debugging

### `Dockerfile`
Builds a container image for the RAG server. Updated to use the context-aware server.
- Only needed if you uncomment the rag-server in docker-compose.yml
- Not used in normal MCP operation

### `start_qdrant.sh`
Simple script to start Qdrant with health checks.

### `start_with_env.sh`
Legacy script that supports both Docker and local modes. Kept for backwards compatibility.

## Normal Usage

For typical usage with Claude Code:

1. **Start Qdrant**:
   ```bash
   ./start_qdrant.sh
   ```

2. **Use Claude Code**:
   ```bash
   cd ~/any-project
   claude
   ```

The MCP server runs locally and connects to Qdrant in Docker.

## Why This Setup?

- **Qdrant in Docker**: Consistent, isolated database environment
- **MCP Server Local**: Can detect your current working directory for context awareness
- **Best Performance**: Local execution allows MPS acceleration on Apple Silicon

## Advanced Usage

If you need to run everything in Docker (loses context awareness):

1. Uncomment the `rag-server` section in `docker-compose.yml`
2. Run: `docker-compose up -d`
3. The server will be available at http://localhost:8080

Note: This mode doesn't support project context detection since the server runs in an isolated container.