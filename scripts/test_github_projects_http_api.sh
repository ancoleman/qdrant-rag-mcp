#!/bin/bash

# Test GitHub Projects V2 HTTP API Endpoints
# This script tests all the new GitHub Projects endpoints added in v0.3.4

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

echo -e "${YELLOW}GitHub Projects V2 HTTP API Test Suite${NC}"
echo "========================================"

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
echo -e "${YELLOW}Test 1: Create a new GitHub Project${NC}"
echo "------------------------------------"
PROJECT_RESPONSE=$(curl -s -X POST "$API_BASE/github/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "'$OWNER'",
    "title": "Test Project from HTTP API",
    "body": "Testing GitHub Projects V2 integration via HTTP API"
  }')

echo "$PROJECT_RESPONSE" | pretty_json

# Extract project ID and number from response
PROJECT_ID=$(echo "$PROJECT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('project', {}).get('id', ''))" 2>/dev/null || echo "")
PROJECT_NUMBER=$(echo "$PROJECT_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('project', {}).get('number', ''))" 2>/dev/null || echo "")

if [ -z "$PROJECT_ID" ] || [ -z "$PROJECT_NUMBER" ]; then
    echo -e "${RED}Failed to create project or extract ID/number${NC}"
    exit 1
fi

echo -e "${GREEN}Created project #$PROJECT_NUMBER with ID: $PROJECT_ID${NC}"

echo ""
echo -e "${YELLOW}Test 2: Get project details${NC}"
echo "----------------------------"
curl -s "$API_BASE/github/projects/$OWNER/$PROJECT_NUMBER" | pretty_json

echo ""
echo -e "${YELLOW}Test 3: Create custom fields${NC}"
echo "-----------------------------"

# Create Task Status field (Status is reserved by GitHub)
echo "Creating Task Status field..."
STATUS_FIELD=$(curl -s -X POST "$API_BASE/github/projects/fields" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'$PROJECT_ID'",
    "name": "Task Status",
    "data_type": "SINGLE_SELECT",
    "options": [
      {"name": "📋 Todo", "color": "GRAY"},
      {"name": "🚧 In Progress", "color": "YELLOW"},
      {"name": "✅ Done", "color": "GREEN"}
    ]
  }')

echo "$STATUS_FIELD" | pretty_json
STATUS_FIELD_ID=$(echo "$STATUS_FIELD" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")

# Create Priority field
echo ""
echo "Creating Priority field..."
PRIORITY_FIELD=$(curl -s -X POST "$API_BASE/github/projects/fields" \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "'$PROJECT_ID'",
    "name": "Priority",
    "data_type": "SINGLE_SELECT",
    "options": [
      {"name": "🔥 High", "color": "RED"},
      {"name": "📌 Medium", "color": "YELLOW"},
      {"name": "📎 Low", "color": "BLUE"}
    ]
  }')

echo "$PRIORITY_FIELD" | pretty_json

echo ""
echo -e "${YELLOW}Test 4: Create a test issue and add to project${NC}"
echo "----------------------------------------------"

# First, switch to the repository context
echo "Switching to repository context..."
curl -s -X POST "$API_BASE/github/switch_repository" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "'$OWNER'",
    "repo": "'$REPO'"
  }' | pretty_json

# Create a test issue
echo ""
echo "Creating test issue..."
ISSUE_RESPONSE=$(curl -s -X POST "$API_BASE/github/issues" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Issue for Projects V2",
    "body": "This issue tests GitHub Projects V2 integration",
    "labels": ["test", "projects-v2"]
  }')

echo "$ISSUE_RESPONSE" | pretty_json
ISSUE_NUMBER=$(echo "$ISSUE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('number', ''))" 2>/dev/null || echo "")

if [ -n "$ISSUE_NUMBER" ]; then
    echo -e "${GREEN}Created issue #$ISSUE_NUMBER${NC}"
    
    # Add issue to project
    echo ""
    echo "Adding issue to project..."
    ITEM_RESPONSE=$(curl -s -X POST "$API_BASE/github/projects/items" \
      -H "Content-Type: application/json" \
      -d '{
        "project_id": "'$PROJECT_ID'",
        "issue_number": '$ISSUE_NUMBER'
      }')
    
    echo "$ITEM_RESPONSE" | pretty_json
    ITEM_ID=$(echo "$ITEM_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null || echo "")
    
    if [ -n "$ITEM_ID" ] && [ -n "$STATUS_FIELD_ID" ]; then
        echo ""
        echo "Updating item status..."
        
        # Get the option ID for "In Progress"
        STATUS_OPTIONS=$(echo "$STATUS_FIELD" | python3 -c "import sys, json; data=json.load(sys.stdin); options=data.get('options', []); print(next((opt['id'] for opt in options if 'In Progress' in opt['name']), ''))" 2>/dev/null || echo "")
        
        if [ -n "$STATUS_OPTIONS" ]; then
            curl -s -X PUT "$API_BASE/github/projects/items" \
              -H "Content-Type: application/json" \
              -d '{
                "project_id": "'$PROJECT_ID'",
                "item_id": "'$ITEM_ID'",
                "field_id": "'$STATUS_FIELD_ID'",
                "value": "'$STATUS_OPTIONS'"
              }' | pretty_json
        fi
    fi
fi

echo ""
echo -e "${YELLOW}Test 5: Get project status with metrics${NC}"
echo "---------------------------------------"
curl -s "$API_BASE/github/projects/$OWNER/$PROJECT_NUMBER/status" | pretty_json

echo ""
echo -e "${YELLOW}Test 6: Create project from template${NC}"
echo "------------------------------------"
TEMPLATE_PROJECT=$(curl -s -X POST "$API_BASE/github/projects" \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "'$OWNER'",
    "title": "Bug Tracking Template Test",
    "body": "Testing project templates",
    "template": "bugs"
  }')

echo "$TEMPLATE_PROJECT" | pretty_json

echo ""
echo -e "${GREEN}All tests completed!${NC}"
echo ""
echo "Note: You may want to manually delete the test projects created during this test."
echo "Project numbers created: #$PROJECT_NUMBER and any template projects"