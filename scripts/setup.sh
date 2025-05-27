#!/bin/bash

# Enhanced setup.sh with Claude Code directory handling

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Qdrant MCP RAG Server...${NC}"

# Function to check if Claude Code is installed
check_claude_code() {
    if [ ! -d "$HOME/.claude-code" ]; then
        echo -e "${YELLOW}Claude Code configuration directory not found.${NC}"
        echo -e "${BLUE}Creating ~/.claude-code directory...${NC}"
        
        # Create the directory structure
        mkdir -p "$HOME/.claude-code"
        
        # Create a basic configuration
        cat > "$HOME/.claude-code/config.json" <<EOF
{
  "version": "1.0",
  "mcp_servers": {
    "enabled": true,
    "servers": []
  }
}
EOF
        
        echo -e "${GREEN}Created ~/.claude-code directory with basic configuration.${NC}"
        echo -e "${YELLOW}Note: You'll need to install Claude Code to use this MCP server.${NC}"
        echo -e "${YELLOW}Visit: https://anthropic.com/claude-code for installation instructions.${NC}"
        echo ""
        
        # Ask if user wants to add MCP server config now
        read -p "Would you like to add this MCP server to Claude Code config now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            add_mcp_config
        else
            echo -e "${BLUE}You can add the MCP server configuration later by running:${NC}"
            echo -e "${BLUE}./install_global.sh${NC}"
        fi
    else
        echo -e "${GREEN}Claude Code directory found.${NC}"
        add_mcp_config
    fi
}

# Function to add MCP server configuration
add_mcp_config() {
    MCP_CONFIG_FILE="$HOME/.claude-code/mcp-servers.json"
    
    # Check if mcp-servers.json exists
    if [ ! -f "$MCP_CONFIG_FILE" ]; then
        echo -e "${BLUE}Creating MCP servers configuration...${NC}"
        cat > "$MCP_CONFIG_FILE" <<EOF
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
EOF
        echo -e "${GREEN}Created MCP servers configuration.${NC}"
    else
        echo -e "${YELLOW}MCP servers configuration already exists.${NC}"
        
        # Check if our server is already configured
        if ! grep -q "qdrant-rag" "$MCP_CONFIG_FILE"; then
            echo -e "${BLUE}Adding qdrant-rag server to existing configuration...${NC}"
            
            # Backup existing config
            cp "$MCP_CONFIG_FILE" "$MCP_CONFIG_FILE.backup"
            
            # Use Python to add our server to the existing config
            python3 <<EOF
import json
import os

config_file = os.path.expanduser("$MCP_CONFIG_FILE")
with open(config_file, 'r') as f:
    config = json.load(f)

# Add our server
new_server = {
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
    "auto_start": True,
    "auto_use": {
        "enabled": True,
        "triggers": {
            "code_patterns": ["implement", "create", "refactor", "similar to"],
            "config_patterns": ["configuration", "settings", "parameter"],
            "search_patterns": ["find", "search", "look for", "where is"]
        }
    }
}

if "servers" not in config:
    config["servers"] = []

config["servers"].append(new_server)

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print("Added qdrant-rag server to configuration.")
EOF
            echo -e "${GREEN}Successfully added qdrant-rag to MCP configuration.${NC}"
        else
            echo -e "${GREEN}qdrant-rag server already configured.${NC}"
        fi
    fi
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker compose &> /dev/null; then
    echo -e "${YELLOW}Docker Compose V1 not found. Checking for Docker Compose V2...${NC}"
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    else
        echo -e "${GREEN}Docker Compose V2 found.${NC}"
        # Create an alias for consistency
        alias docker-compose="docker compose"
    fi
fi

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    if [ -f config/.env.example ]; then
        cp config/.env.example .env
    else
        # Create a basic .env file
        cat > .env <<EOF
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=
QDRANT_GRPC_PORT=6334

# Server Configuration
SERVER_PORT=8080
LOG_LEVEL=INFO

# Embeddings Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Claude Code Integration
CLAUDE_CODE_CONFIG_PATH=~/.claude-code
EOF
        echo -e "${GREEN}Created basic .env file${NC}"
    fi
fi

# Check for Python and required version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Install Python dependencies (for local development)
echo -e "${YELLOW}Installing Python dependencies...${NC}"
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}requirements.txt not found. Creating with basic dependencies...${NC}"
    cat > requirements.txt <<EOF
qdrant-client==1.7.3
sentence-transformers==2.3.1
langchain==0.1.4
tiktoken==0.5.2
mcp==0.1.0
python-dotenv==1.0.0
watchdog==3.0.0
pydantic==2.5.3
uvicorn==0.25.0
httpx==0.25.2
EOF
    pip install -r requirements.txt
fi

# Check/create necessary directories
echo -e "${YELLOW}Creating necessary directories...${NC}"
mkdir -p src/{indexers,utils} config docker scripts tests data/qdrant_storage logs

# Check for Claude Code configuration
check_claude_code

# Start services with Docker Compose
echo -e "${YELLOW}Starting Docker services...${NC}"
if [ -f docker/docker-compose.yml ]; then
    cd docker && docker compose up -d && cd ..
else
    echo -e "${RED}docker-compose.yml not found!${NC}"
    echo -e "${YELLOW}Creating basic docker-compose.yml...${NC}"
    mkdir -p docker
    cat > docker/docker-compose.yml <<EOF
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_mcp
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ../data/qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 5s
      timeout: 5s
      retries: 5
EOF
    cd docker && docker compose up -d && cd ..
fi

# Wait for Qdrant to be ready
echo -e "${YELLOW}Waiting for Qdrant to be ready...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:6333/health | grep -q "ok"; then
        echo -e "${GREEN}Qdrant is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Qdrant failed to start after $max_attempts seconds!${NC}"
    echo -e "${YELLOW}Check Docker logs: docker compose -f docker/docker-compose.yml logs qdrant${NC}"
    exit 1
fi

# Note: Claude Code configuration is now handled by install_global.sh

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${GREEN}Services are running:${NC}"
echo -e "  - Qdrant: http://localhost:6333"
echo -e "  - MCP Server: http://localhost:8080 (will start when Claude Code requests it)"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. If Claude Code is not installed, install it from: https://anthropic.com/claude-code"
echo -e "  2. Start the MCP server: ./scripts/start_server.sh"
echo -e "  3. Index your project: ./scripts/index_project.sh /path/to/your/project"
echo ""
echo -e "${GREEN}Happy coding with Claude Code + RAG!${NC}"
