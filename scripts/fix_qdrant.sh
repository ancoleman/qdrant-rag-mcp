#!/bin/bash

# scripts/fix_qdrant.sh - Fix Qdrant container health check

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Fixing Qdrant Container ===${NC}"

# Update the docker-compose.yml file to use the correct health check
echo -e "${YELLOW}Updating Docker Compose health check...${NC}"

# Create a temporary file
tmp_file=$(mktemp)

# Replace health check line in docker-compose.yml
awk 'BEGIN { replacing = 0 }
{
    if ($0 ~ /healthcheck:/) {
        replacing = 1
        print $0
    } else if (replacing && $0 ~ /test:/) {
        print "      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:6333/collections\"]"
        replacing = 0
    } else {
        print $0
    }
}' docker/docker-compose.yml > "$tmp_file"

# Copy back to original
cp "$tmp_file" docker/docker-compose.yml
rm "$tmp_file"

echo -e "${GREEN}Docker Compose updated.${NC}"

# Restart the containers
echo -e "${YELLOW}Restarting containers...${NC}"
cd docker && docker compose down && docker compose up -d && cd ..

# Wait for Qdrant to be ready
echo -e "${YELLOW}Waiting for Qdrant to be ready...${NC}"
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:6333/collections > /dev/null; then
        echo -e "${GREEN}Qdrant is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
    attempt=$((attempt + 1))
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Qdrant failed to start after $max_attempts seconds!${NC}"
    echo -e "${YELLOW}Trying direct check...${NC}"
    
    # Try direct connection without health check
    curl -v http://localhost:6333/collections
    
    echo -e "${RED}Please check Docker logs: docker compose -f docker/docker-compose.yml logs qdrant${NC}"
    exit 1
fi

# Start the RAG server
echo -e "${YELLOW}Starting RAG server...${NC}"
docker compose -f docker/docker-compose.yml start rag-server || docker compose -f docker/docker-compose.yml up -d rag-server

echo -e "${GREEN}Fix complete!${NC}"
echo -e "${YELLOW}Test with: curl http://localhost:8080/health${NC}"
