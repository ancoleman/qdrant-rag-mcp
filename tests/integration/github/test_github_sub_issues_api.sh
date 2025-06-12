#!/bin/bash

# Test script for GitHub Sub-Issues API endpoints
# This script tests the sub-issues functionality added in v0.3.4.post4

# Load .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Set the API base URL
API_BASE="http://localhost:8081"
REPO_OWNER="anthropics"
REPO_NAME="qdrant-rag-mcp"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if server is running
print_status "Checking if HTTP server is running..."
if ! curl -s "${API_BASE}/health" > /dev/null; then
    print_error "HTTP server is not running on ${API_BASE}"
    print_warning "Please start the server with: python src/http_server.py"
    exit 1
fi
print_success "Server is running"

# Test repository context
print_status "Setting repository context to ${REPO_OWNER}/${REPO_NAME}..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/switch_repository" \
    -H "Content-Type: application/json" \
    -d "{\"owner\": \"${REPO_OWNER}\", \"repo\": \"${REPO_NAME}\"}")

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to set repository context"
    echo "$RESPONSE" | jq .
    exit 1
fi
print_success "Repository context set"

# Variables to store created issues
PARENT_ISSUE=""
SUB_ISSUE_1=""
SUB_ISSUE_2=""
SUB_ISSUE_3=""

# Create a parent issue for testing
print_status "Creating parent issue..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/issues" \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Test Parent Issue for Sub-Issues",
        "body": "This is a test parent issue to demonstrate sub-issues functionality.\n\n## Tasks:\n- [ ] Task 1\n- [ ] Task 2\n- [ ] Task 3",
        "labels": ["test", "enhancement"]
    }')

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to create parent issue"
    echo "$RESPONSE" | jq .
else
    PARENT_ISSUE=$(echo "$RESPONSE" | jq -r '.number')
    print_success "Created parent issue #${PARENT_ISSUE}"
fi

if [ -z "$PARENT_ISSUE" ]; then
    print_error "No parent issue number, cannot continue"
    exit 1
fi

# Test 1: Create sub-issues
print_status "Test 1: Creating sub-issues..."

# Create first sub-issue using github_create_sub_issue
print_status "Creating sub-issue 1 using create_sub_issue..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/create_sub_issue" \
    -H "Content-Type: application/json" \
    -d "{
        \"parent_issue_number\": ${PARENT_ISSUE},
        \"title\": \"Sub-task 1: Implement feature A\",
        \"body\": \"First sub-task of parent issue #${PARENT_ISSUE}\"
    }")

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to create sub-issue 1"
    echo "$RESPONSE" | jq .
else
    SUB_ISSUE_1=$(echo "$RESPONSE" | jq -r '.sub_issue.number')
    print_success "Created sub-issue #${SUB_ISSUE_1}"
fi

# Create second sub-issue normally and then link it
print_status "Creating regular issue to link as sub-issue 2..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/issues" \
    -H "Content-Type: application/json" \
    -d '{
        "title": "Sub-task 2: Implement feature B",
        "body": "Second sub-task to be linked",
        "labels": ["test"]
    }')

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to create issue for sub-issue 2"
    echo "$RESPONSE" | jq .
else
    SUB_ISSUE_2=$(echo "$RESPONSE" | jq -r '.number')
    print_success "Created issue #${SUB_ISSUE_2}"
    
    # Link it as a sub-issue
    print_status "Linking issue #${SUB_ISSUE_2} as sub-issue..."
    RESPONSE=$(curl -s -X POST "${API_BASE}/github/add_sub_issue" \
        -H "Content-Type: application/json" \
        -d "{
            \"parent_issue_number\": ${PARENT_ISSUE},
            \"sub_issue_number\": ${SUB_ISSUE_2}
        }")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to link sub-issue 2"
        echo "$RESPONSE" | jq .
    else
        print_success "Linked issue #${SUB_ISSUE_2} as sub-issue"
    fi
fi

# Create third sub-issue
print_status "Creating sub-issue 3..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/create_sub_issue" \
    -H "Content-Type: application/json" \
    -d "{
        \"parent_issue_number\": ${PARENT_ISSUE},
        \"title\": \"Sub-task 3: Write documentation\",
        \"body\": \"Documentation task\"
    }")

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to create sub-issue 3"
    echo "$RESPONSE" | jq .
else
    SUB_ISSUE_3=$(echo "$RESPONSE" | jq -r '.sub_issue.number')
    print_success "Created sub-issue #${SUB_ISSUE_3}"
fi

# Test 2: List sub-issues
print_status "Test 2: Listing sub-issues for parent #${PARENT_ISSUE}..."
RESPONSE=$(curl -s -X POST "${API_BASE}/github/list_sub_issues" \
    -H "Content-Type: application/json" \
    -d "{\"parent_issue_number\": ${PARENT_ISSUE}}")

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to list sub-issues"
    echo "$RESPONSE" | jq .
else
    SUB_COUNT=$(echo "$RESPONSE" | jq -r '.sub_issues_count')
    print_success "Found ${SUB_COUNT} sub-issues"
    echo "$RESPONSE" | jq '.sub_issues'
fi

# Test 3: Reorder sub-issues
if [ -n "$SUB_ISSUE_1" ] && [ -n "$SUB_ISSUE_2" ] && [ -n "$SUB_ISSUE_3" ]; then
    print_status "Test 3: Reordering sub-issues..."
    RESPONSE=$(curl -s -X POST "${API_BASE}/github/reorder_sub_issues" \
        -H "Content-Type: application/json" \
        -d "{
            \"parent_issue_number\": ${PARENT_ISSUE},
            \"sub_issue_numbers\": [${SUB_ISSUE_3}, ${SUB_ISSUE_1}, ${SUB_ISSUE_2}]
        }")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to reorder sub-issues"
        echo "$RESPONSE" | jq .
    else
        print_success "Reordered sub-issues"
    fi
fi

# Test 4: Remove a sub-issue
if [ -n "$SUB_ISSUE_2" ]; then
    print_status "Test 4: Removing sub-issue #${SUB_ISSUE_2}..."
    RESPONSE=$(curl -s -X POST "${API_BASE}/github/remove_sub_issue" \
        -H "Content-Type: application/json" \
        -d "{
            \"parent_issue_number\": ${PARENT_ISSUE},
            \"sub_issue_number\": ${SUB_ISSUE_2}
        }")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to remove sub-issue"
        echo "$RESPONSE" | jq .
    else
        print_success "Removed sub-issue #${SUB_ISSUE_2}"
    fi
fi

# Test 5: Re-parent a sub-issue
if [ -n "$SUB_ISSUE_2" ] && [ -n "$SUB_ISSUE_3" ]; then
    print_status "Test 5: Re-parenting sub-issue #${SUB_ISSUE_2} to #${SUB_ISSUE_3}..."
    RESPONSE=$(curl -s -X POST "${API_BASE}/github/add_sub_issue" \
        -H "Content-Type: application/json" \
        -d "{
            \"parent_issue_number\": ${SUB_ISSUE_3},
            \"sub_issue_number\": ${SUB_ISSUE_2},
            \"replace_parent\": true
        }")
    
    if echo "$RESPONSE" | grep -q "error"; then
        print_error "Failed to re-parent sub-issue"
        echo "$RESPONSE" | jq .
    else
        print_success "Re-parented sub-issue #${SUB_ISSUE_2} to #${SUB_ISSUE_3}"
    fi
fi

# Test 6: Projects V2 Integration
print_status "Test 6: Testing Projects V2 integration..."

# First, list projects
print_status "Listing available projects..."
RESPONSE=$(curl -s "${API_BASE}/github/projects")

if echo "$RESPONSE" | grep -q "error"; then
    print_error "Failed to list projects"
    echo "$RESPONSE" | jq .
else
    PROJECT_COUNT=$(echo "$RESPONSE" | jq -r '.total_count')
    print_success "Found ${PROJECT_COUNT} projects"
    
    if [ "$PROJECT_COUNT" -gt 0 ]; then
        # Get the first project ID
        PROJECT_ID=$(echo "$RESPONSE" | jq -r '.projects[0].id')
        PROJECT_TITLE=$(echo "$RESPONSE" | jq -r '.projects[0].title')
        
        print_status "Adding sub-issues to project '${PROJECT_TITLE}'..."
        RESPONSE=$(curl -s -X POST "${API_BASE}/github/add_sub_issues_to_project" \
            -H "Content-Type: application/json" \
            -d "{
                \"project_id\": \"${PROJECT_ID}\",
                \"parent_issue_number\": ${PARENT_ISSUE}
            }")
        
        if echo "$RESPONSE" | grep -q "error"; then
            print_error "Failed to add sub-issues to project"
            echo "$RESPONSE" | jq .
        else
            ADDED_COUNT=$(echo "$RESPONSE" | jq -r '.added_count')
            print_success "Added ${ADDED_COUNT} sub-issues to project"
            echo "$RESPONSE" | jq '.sub_issues'
        fi
    else
        print_warning "No projects found, skipping project integration test"
    fi
fi

# Summary
echo
print_status "Test Summary:"
print_status "Parent Issue: #${PARENT_ISSUE}"
print_status "Sub-Issues Created: #${SUB_ISSUE_1}, #${SUB_ISSUE_2}, #${SUB_ISSUE_3}"
echo
print_warning "Note: These are test issues. You may want to close them after testing."
print_warning "To close an issue, use: gh issue close <issue-number>"

# Cleanup prompt
echo
read -p "Would you like to close the test issues? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Closing test issues..."
    
    # Close issues using gh CLI if available
    if command -v gh &> /dev/null; then
        for issue in $PARENT_ISSUE $SUB_ISSUE_1 $SUB_ISSUE_2 $SUB_ISSUE_3; do
            if [ -n "$issue" ]; then
                gh issue close "$issue" --repo "${REPO_OWNER}/${REPO_NAME}" 2>/dev/null && \
                    print_success "Closed issue #${issue}" || \
                    print_warning "Could not close issue #${issue}"
            fi
        done
    else
        print_warning "GitHub CLI (gh) not found. Please close issues manually."
    fi
fi

print_success "Sub-issues API tests completed!"