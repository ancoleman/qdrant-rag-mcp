# GitHub Integration Tests

This directory contains integration tests for GitHub-related functionality.

## Test Files

- `test_github_http_api.sh` - Basic GitHub API tests (v0.3.0)
  - Repository operations
  - Issue CRUD operations
  - Issue analysis and fix suggestions

- `test_github_enhanced_issues_api.sh` - Enhanced issue management tests (v0.3.4.post5)
  - Milestone CRUD operations
  - Issue lifecycle (create, update, assign, close)
  - Enhanced filtering (milestone, assignee)
  - Advanced search with GitHub syntax
  - Sort and pagination

- `test_github_projects_http_api.sh` - GitHub Projects V2 tests (v0.3.4.post4)
  - Project CRUD operations
  - Project item management
  - Field management
  - Template creation

- `test_github_projects_smart_add.sh` - Smart project item addition tests
  - Intelligent field assignment
  - Issue metadata analysis
  - Automated field value selection

- `test_github_sub_issues_api.sh` - Sub-issue management tests (v0.3.4.post4)
  - Sub-issue relationships
  - Hierarchical issue management
  - Bulk operations on sub-issues

## Running Tests

From the project root:

```bash
# Run specific test suite
./tests/integration/github/test_github_enhanced_issues_api.sh

# Run all GitHub tests
for test in tests/integration/github/test_*.sh; do
    echo "Running $test..."
    $test
done
```

## Prerequisites

- HTTP server running with GitHub integration enabled
- GitHub token configured in environment
- Test repository access
- jq installed for JSON parsing

## Environment Variables

Required:
- `GITHUB_TOKEN` - GitHub personal access token
- `TEST_REPO_OWNER` - Repository owner for testing (optional, defaults to authenticated user)
- `TEST_REPO_NAME` - Repository name for testing (optional)