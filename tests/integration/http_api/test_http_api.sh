#!/bin/bash
# test_http_api.sh - Quick HTTP API testing script

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Qdrant RAG HTTP API Test Script${NC}"
echo ""

# Check if HTTP server is running
echo -e "${YELLOW}Checking HTTP server status...${NC}"
if curl -s http://localhost:8081/health > /dev/null; then
    echo -e "${GREEN}✅ HTTP server is running on port 8081${NC}"
else
    echo -e "${RED}❌ HTTP server not running. Start it with:${NC}"
    echo -e "${BLUE}   python src/http_server.py${NC}"
    exit 1
fi

echo ""

# Test 1: Health check
echo -e "${YELLOW}Test 1: Health Check${NC}"
curl -s http://localhost:8081/health | jq '.'
echo ""

# Test 2: Check collections
echo -e "${YELLOW}Test 2: Available Collections${NC}"
curl -s http://localhost:8081/collections | jq '.'
echo ""

# Test 3: Index the RAG server itself
echo -e "${YELLOW}Test 3: Index the RAG server code${NC}"
curl -X POST http://localhost:8081/index_code \
  -H "Content-Type: application/json" \
  -d "{\"file_path\": \"$(pwd)/src/qdrant_mcp_server.py\"}" | jq '.'
echo ""

# Test 4: Index a config file
echo -e "${YELLOW}Test 4: Index environment configuration${NC}"
curl -X POST http://localhost:8081/index_config \
  -H "Content-Type: application/json" \
  -d "{\"file_path\": \"$(pwd)/.env\"}" | jq '.'
echo ""

# Test 5: General search
echo -e "${YELLOW}Test 5: General Search - 'embedding model'${NC}"
curl -X POST http://localhost:8081/search \
  -H "Content-Type: application/json" \
  -d '{"query": "embedding model", "n_results": 2}' | jq '.results[] | {score, type, file_path, preview}'
echo ""

# Test 6: Code-specific search
echo -e "${YELLOW}Test 6: Code Search - 'initialize' in Python${NC}"
curl -X POST http://localhost:8081/search_code \
  -H "Content-Type: application/json" \
  -d '{"query": "initialize", "language": "python", "n_results": 2}' | jq '.results[] | {score, chunk_type, line_range, preview}'
echo ""

# Test 7: Config search
echo -e "${YELLOW}Test 7: Config Search - 'embedding'${NC}"
curl -X POST http://localhost:8081/search_config \
  -H "Content-Type: application/json" \
  -d '{"query": "embedding", "n_results": 2}' | jq '.results[] | {score, path, value, file_path}'
echo ""

echo -e "${GREEN}🎉 HTTP API testing complete!${NC}"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo -e "   • View Qdrant dashboard: ${BLUE}open http://localhost:6333/dashboard${NC}"
echo -e "   • Check server logs: ${BLUE}tail -f /tmp/http_server.log${NC}"
echo -e "   • Index more files: ${BLUE}curl -X POST http://localhost:8081/index_directory${NC}"
echo ""