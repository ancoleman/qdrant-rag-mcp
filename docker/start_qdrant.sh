#!/bin/bash
# Simple script to start only Qdrant for use with the MCP server

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting Qdrant Vector Database...${NC}"

# Navigate to docker directory
cd "$(dirname "$0")"

# Check if Qdrant is already running
if docker ps | grep -q qdrant_mcp; then
    echo -e "${YELLOW}Qdrant is already running${NC}"
    echo "To restart: docker-compose -f docker-compose-qdrant-only.yml restart"
    echo "To stop: docker-compose -f docker-compose-qdrant-only.yml down"
else
    # Start Qdrant
    echo "Starting Qdrant container..."
    docker-compose -f docker-compose-qdrant-only.yml up -d
    
    # Wait for Qdrant to be ready
    echo "Waiting for Qdrant to be ready..."
    max_attempts=30
    attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s http://localhost:6333/health | grep -q "ok"; then
            echo -e "${GREEN}✓ Qdrant is ready!${NC}"
            break
        fi
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    if [ $attempt -eq $max_attempts ]; then
        echo -e "${RED}✗ Qdrant failed to start${NC}"
        echo "Check logs: docker logs qdrant_mcp"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Qdrant is running at:${NC}"
echo "  - REST API: http://localhost:6333"
echo "  - gRPC API: http://localhost:6334"
echo "  - Dashboard: http://localhost:6333/dashboard"
echo ""
echo "The MCP server will connect to this when you use Claude Code."