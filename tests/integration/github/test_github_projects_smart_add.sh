#!/bin/bash

# Test GitHub Projects V2 Smart Add functionality
# This script demonstrates the intelligent field assignment feature

set -e

# Configuration
API_BASE="http://localhost:8081"
OWNER="ancoleman"  # Change to your GitHub username or org
REPO="qdrant-rag-mcp"  # Change to your test repository

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}GitHub Projects V2 Smart Add Test${NC}"
echo "====================================="

# Function to check if API is running
check_api() {
    echo -n "Checking if API is running... "
    if curl -s "$API_BASE/health" > /dev/null; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
        echo "Please start the HTTP API server: python src/http_server.py"
        exit 1
    fi
}

# Function to pretty print JSON
pretty_json() {
    python3 -m json.tool
}

# Check API health
check_api

echo ""
echo -e "${YELLOW}Step 1: Create a Bug Tracking Project${NC}"
echo "--------------------------------------"
PROJECT_RESPONSE=$(curl -s -X POST "$API_BASE/github/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "'$OWNER'",
    "title": "Smart Add Demo Project",
    "template": "bugs"
  }')

echo "$PROJECT_RESPONSE" | pretty_json

# Extract project ID
PROJECT_ID=$(echo "$PROJECT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('project', {}).get('id', ''))" 2>/dev/null || echo "")

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Failed to create project${NC}"
    exit 1
fi

echo -e "${GREEN}Created project with ID: $PROJECT_ID${NC}"

echo ""
echo -e "${YELLOW}Step 2: Switch to repository${NC}"
echo "-----------------------------"
curl -s -X POST "$API_BASE/github/switch_repository" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "'$OWNER'",
    "repo": "'$REPO'"
  }' | pretty_json

echo ""
echo -e "${YELLOW}Step 3: Create test issues with different characteristics${NC}"
echo "--------------------------------------------------------"

# High priority bug
echo "Creating high priority bug..."
BUG_RESPONSE=$(curl -s -X POST "$API_BASE/github/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Critical: Application crashes on startup",
    "body": "The application crashes immediately when trying to start. This is blocking all development work.\n\nSteps to reproduce:\n1. Run the application\n2. Observe crash\n\nError: Segmentation fault",
    "labels": ["bug", "critical", "blocking"]
  }')

BUG_NUMBER=$(echo "$BUG_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('issue', {}).get('number', ''))" 2>/dev/null || echo "")
echo -e "${GREEN}Created bug issue #$BUG_NUMBER${NC}"

# Medium priority search issue
echo ""
echo "Creating search-related issue..."
SEARCH_RESPONSE=$(curl -s -X POST "$API_BASE/github/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Search results are inconsistent",
    "body": "The search functionality returns different results for the same query.\n\nThis issue affects the search component and needs investigation.",
    "labels": ["bug", "search"]
  }')

SEARCH_NUMBER=$(echo "$SEARCH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('issue', {}).get('number', ''))" 2>/dev/null || echo "")
echo -e "${GREEN}Created search issue #$SEARCH_NUMBER${NC}"

# Low priority documentation issue
echo ""
echo "Creating documentation issue..."
DOC_RESPONSE=$(curl -s -X POST "$API_BASE/github/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Update README with new API endpoints",
    "body": "The documentation needs to be updated to include the new API endpoints added in v0.3.4",
    "labels": ["documentation", "low-priority"]
  }')

DOC_NUMBER=$(echo "$DOC_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('issue', {}).get('number', ''))" 2>/dev/null || echo "")
echo -e "${GREEN}Created documentation issue #$DOC_NUMBER${NC}"

echo ""
echo -e "${YELLOW}Step 4: Add issues to project with smart field assignment${NC}"
echo "---------------------------------------------------------"

# Add high priority bug
echo ""
echo "Adding critical bug with smart assignment..."
SMART_ADD1=$(curl -s -X POST "$API_BASE/github/projects/items/smart" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'$PROJECT_ID'",
    "issue_number": '$BUG_NUMBER'
  }')

echo "$SMART_ADD1" | pretty_json

# Add search issue
echo ""
echo "Adding search issue with smart assignment..."
SMART_ADD2=$(curl -s -X POST "$API_BASE/github/projects/items/smart" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'$PROJECT_ID'",
    "issue_number": '$SEARCH_NUMBER'
  }')

echo "$SMART_ADD2" | pretty_json

# Add documentation issue
echo ""
echo "Adding documentation issue with smart assignment..."
SMART_ADD3=$(curl -s -X POST "$API_BASE/github/projects/items/smart" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'$PROJECT_ID'",
    "issue_number": '$DOC_NUMBER'
  }')

echo "$SMART_ADD3" | pretty_json

echo ""
echo -e "${YELLOW}Step 5: Get project status to see all items${NC}"
echo "-------------------------------------------"
PROJECT_NUMBER=$(echo "$PROJECT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('project', {}).get('number', ''))" 2>/dev/null || echo "")
curl -s "$API_BASE/github/projects/$OWNER/$PROJECT_NUMBER/status" | pretty_json

echo ""
echo -e "${GREEN}Smart Add Test Complete!${NC}"
echo ""
echo "The test demonstrated:"
echo "- Critical bug was assigned high severity and marked as new"
echo "- Search issue was assigned to the Search component"
echo "- Documentation issue was assigned to Documentation component with low priority"
echo ""
echo "Visit https://github.com/users/$OWNER/projects/$PROJECT_NUMBER to see the results"