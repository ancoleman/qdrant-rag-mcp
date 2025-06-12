#!/bin/bash
# Test script for GitHub HTTP API endpoints (v0.3.0)

set -e

BASE_URL="http://localhost:8081"
echo "🧪 Testing GitHub HTTP API endpoints at $BASE_URL"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test an endpoint
test_endpoint() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    local description="$4"
    
    echo -e "\n${BLUE}Testing: $description${NC}"
    echo "Endpoint: $method $endpoint"
    
    if [ "$method" = "GET" ]; then
        if curl -s -f -X GET "$BASE_URL$endpoint" -H "Content-Type: application/json" > /tmp/response.json; then
            echo -e "${GREEN}✓ Success${NC}"
            # Show abbreviated response
            head -c 200 /tmp/response.json
            if [ $(wc -c < /tmp/response.json) -gt 200 ]; then
                echo "..."
            fi
            echo
        else
            echo -e "${RED}✗ Failed${NC}"
            cat /tmp/response.json 2>/dev/null || echo "No response body"
        fi
    else
        if curl -s -f -X "$method" "$BASE_URL$endpoint" \
           -H "Content-Type: application/json" \
           -d "$data" > /tmp/response.json; then
            echo -e "${GREEN}✓ Success${NC}"
            # Show abbreviated response
            head -c 200 /tmp/response.json
            if [ $(wc -c < /tmp/response.json) -gt 200 ]; then
                echo "..."
            fi
            echo
        else
            echo -e "${RED}✗ Failed${NC}"
            cat /tmp/response.json 2>/dev/null || echo "No response body"
        fi
    fi
}

# Check if server is running
echo -e "${BLUE}Checking if server is running...${NC}"
if ! curl -s -f "$BASE_URL/health" > /dev/null; then
    echo -e "${RED}❌ Server is not running at $BASE_URL${NC}"
    echo "Start the server with: python src/http_server.py"
    exit 1
fi
echo -e "${GREEN}✓ Server is running${NC}"

# Test basic health
test_endpoint "GET" "/health" "" "Basic server health check"

# Test GitHub health (will show if GitHub integration is available)
test_endpoint "GET" "/github/health" "" "GitHub integration health check"

echo -e "\n${YELLOW}📝 Note: The following tests require GitHub authentication${NC}"
echo "Set GITHUB_TOKEN in your .env file to test authenticated endpoints"

# Test GitHub endpoints (these will fail if not authenticated, but that's expected)
test_endpoint "GET" "/github/repositories" "" "List repositories (requires auth)"

test_endpoint "POST" "/github/switch_repository" '{
    "owner": "octocat",
    "repo": "Hello-World"
}' "Switch repository context (requires auth)"

test_endpoint "GET" "/github/issues" "" "Fetch issues from current repository (requires auth + repo context)"

test_endpoint "GET" "/github/issues/1" "" "Get specific issue #1 (requires auth + repo context)"

test_endpoint "POST" "/github/issues/1/analyze" "" "Analyze issue #1 (requires auth + repo context)"

test_endpoint "POST" "/github/issues/1/suggest_fix" "" "Generate fix suggestions for issue #1 (requires auth + repo context)"

test_endpoint "POST" "/github/issues/1/resolve?dry_run=true" "" "Test issue resolution workflow (dry run)"

test_endpoint "POST" "/github/pull_requests" '{
    "title": "Test PR",
    "body": "This is a test pull request",
    "head": "test-branch",
    "base": "main"
}' "Create pull request (requires auth + repo context + branch)"

echo -e "\n${BLUE}🏁 GitHub HTTP API Testing Complete${NC}"
echo "=================================================="

echo -e "\n${YELLOW}📋 To test with authentication:${NC}"
echo "1. Set GITHUB_TOKEN in your .env file"
echo "2. Restart the HTTP server"
echo "3. Run this script again"

echo -e "\n${YELLOW}📋 Example manual testing with curl:${NC}"
echo
echo "# List repositories"
echo "curl -X GET $BASE_URL/github/repositories"
echo
echo "# Switch to a repository"
echo "curl -X POST $BASE_URL/github/switch_repository \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"owner\": \"your-username\", \"repo\": \"your-repo\"}'"
echo
echo "# Fetch issues"
echo "curl -X GET '$BASE_URL/github/issues?state=open&limit=5'"
echo
echo "# Analyze an issue"
echo "curl -X POST $BASE_URL/github/issues/123/analyze"
echo
echo "# Suggest fixes for an issue"  
echo "curl -X POST $BASE_URL/github/issues/123/suggest_fix"
echo
echo "# Test resolution workflow (dry run)"
echo "curl -X POST '$BASE_URL/github/issues/123/resolve?dry_run=true'"

echo -e "\n${GREEN}🎯 Ready for GitHub issue resolution testing!${NC}"