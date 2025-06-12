# GitHub Integration Guide - v0.3.0 (Updated for v0.3.4.post5)

This guide covers the GitHub issue resolution capabilities added in v0.3.0, enhanced issue management features (v0.3.4.post5), and usage examples including milestones, advanced filtering, and complete issue lifecycle management.

## Table of Contents

- [🚀 Overview](#-overview)
- [📋 Prerequisites](#-prerequisites)
- [🏗️ Architecture](#️-architecture)
- [🔐 Authentication Setup](#-authentication-setup)
- [🛠️ Configuration](#️-configuration)
- [📚 Usage Examples](#-usage-examples)
- [📋 Sub-Issues Management](#-sub-issues-management-v034post4)
- [📋 Enhanced Issue Management](#-enhanced-issue-management-v034post5)
- [🔍 Analysis Features](#-analysis-features)
- [🚀 Token Optimization](#-token-optimization)
- [🛡️ Safety Features](#️-safety-features)
- [📊 API Reference](#-api-reference)
- [🧪 Testing Workflows](#-testing-workflows)
- [🤖 Real-World Examples](#-real-world-examples)
- [🏆 Best Practices](#-best-practices)
- [❓ Troubleshooting](#-troubleshooting)
- [🚀 Future Enhancements](#-future-enhancements)

## 🚀 Overview

The GitHub integration transforms the Qdrant RAG server into an intelligent issue resolution system that can:

- **Analyze** GitHub issues using RAG search
- **Understand** issue context and extract relevant information
- **Search** your codebase for related code patterns
- **Generate** fix suggestions with confidence scoring
- **Comment** on issues with updates, analysis results, and progress
- **Create** pull requests with automated solutions
- **Orchestrate** end-to-end resolution workflows

## 📋 Prerequisites

### Dependencies

Install the required GitHub dependencies:

```bash
# Install via pip
pip install PyGithub>=2.6.1 GitPython>=3.1.44

# Or via uv (recommended)
uv add PyGithub>=2.6.1 GitPython>=3.1.44
```

The integration gracefully degrades if these dependencies are not installed.

## 🏗️ Architecture

The GitHub integration follows a layered architecture design that separates concerns and provides flexibility:

### Component Overview

```
┌─────────────────────────────────────────────────────┐
│                   MCP Tools Layer                    │
│  (github_* functions in qdrant_mcp_context_aware.py)│
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               Workflows Layer                        │
│         (GitHubWorkflows - workflows.py)             │
│                                                      │
│  • Orchestrates complex multi-step processes        │
│  • Implements business logic & safety checks        │
│  • Manages analysis → fix → PR workflows           │
│  • Handles dry-run mode and feasibility checks     │
└─────────────┬──────────────┬──────────────┬────────┘
              │              │              │
┌─────────────▼──┐  ┌────────▼────────┐  ┌─▼────────────┐
│  IssueAnalyzer │  │  CodeGenerator  │  │ GitHubClient │
│                │  │                 │  │              │
│ • RAG search   │  │ • Fix templates │  │ • API calls  │
│ • Pattern      │  │ • Code patches  │  │ • Auth       │
│   extraction   │  │ • Confidence    │  │ • Rate limit │
│ • Context      │  │   scoring       │  │ • Git ops    │
└────────────────┘  └─────────────────┘  └──────────────┘
```

### Layer Responsibilities

#### 1. **GitHubClient** (Low-level API Operations)
- **Purpose**: Direct interface to GitHub API
- **Key Methods**:
  - `create_issue()` - Create new issues
  - `create_pull_request()` - Basic PR creation (requires existing branch)
  - `create_pull_request_with_changes()` - Full PR workflow with Git operations
  - `add_comment()` - Add comments to issues
- **Features**:
  - Authentication management (PAT and GitHub App)
  - Intelligent rate limiting with separate tracking for core/search APIs
  - Retry logic with exponential backoff
  - User-friendly error messages

#### 2. **GitHubWorkflows** (High-level Business Logic)
- **Purpose**: Orchestrates complex multi-step workflows
- **Key Workflows**:
  - `analyze_issue_workflow()` - RAG-powered issue analysis
  - `suggest_fix_workflow()` - Generate automated fixes
  - `resolve_issue_workflow()` - Complete resolution process
- **Features**:
  - Combines multiple operations into cohesive workflows
  - Safety checks and validation
  - Feasibility assessment
  - Dry-run mode for previewing changes
  - Audit logging for compliance

#### 3. **GitOperations** (Git Repository Management)
- **Purpose**: Handle actual file modifications and Git operations
- **Key Methods**:
  - `prepare_branch()` - Clone repo and create branch
  - `apply_changes()` - Modify files in repository
  - `commit_and_push()` - Commit and push changes
- **Features**:
  - Secure authentication for Git operations
  - Temporary repository management
  - Automatic cleanup

### Example Workflow

When you run `github_resolve_issue(issue_number=123)`, here's what happens:

```python
1. MCP Tool receives the command
   ↓
2. GitHubWorkflows.resolve_issue_workflow() starts
   ↓
3. IssueAnalyzer analyzes the issue using RAG search
   ↓
4. CodeGenerator creates fix suggestions
   ↓
5. Workflows evaluates feasibility and safety
   ↓
6. If approved: GitHubClient.create_pull_request_with_changes()
   ↓
7. GitOperations handles branch/commit/push
   ↓
8. Pull request is created with automated fixes
```

### Design Benefits

1. **Separation of Concerns**: Each layer has a specific responsibility
2. **Flexibility**: Can use components independently or composed
3. **Safety**: Multiple validation layers before making changes
4. **Testability**: Each component can be tested in isolation
5. **Extensibility**: Easy to add new workflows or modify existing ones

### GitHub Authentication

You need GitHub API access via one of these methods:

1. **Personal Access Token** (recommended for individual use)
2. **GitHub App** (recommended for organization/team use)

## 🔐 Authentication Setup

### Option 1: Personal Access Token

#### Step 1: Create a Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token" > "Generate new token (classic)"
3. Configure your token:
   - **Name**: `qdrant-rag-server`
   - **Expiration**: Choose appropriate duration
   - **Scopes**: Select required permissions:
     - `repo` - Full repository access (for private repos)
     - `public_repo` - Public repository access (for public repos only)
     - `issues` - Issue management
     - `pull_requests` - PR creation and management
     - `metadata` - Repository metadata access

#### Step 2: Configure Environment Variables

Add to your `.env` file:

```bash
# GitHub Authentication
GITHUB_TOKEN=ghp_your_token_here

# Optional: Default repository
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=your-repo-name
```

### Option 2: GitHub App (Advanced)

#### Step 1: Create a GitHub App

1. Go to [GitHub Settings > Developer settings > GitHub Apps](https://github.com/settings/apps)
2. Click "New GitHub App"
3. Configure your app:
   - **GitHub App name**: `qdrant-rag-bot`
   - **Homepage URL**: Your organization URL
   - **Webhook**: Disable for now
   - **Permissions**:
     - Repository permissions:
       - `Contents`: Read & Write
       - `Issues`: Read & Write
       - `Pull requests`: Read & Write
       - `Metadata`: Read
     - Account permissions: None needed

#### Step 2: Install and Configure

1. After creation, note the **App ID**
2. Generate and download a **private key**
3. Install the app on your repositories
4. Note the **Installation ID** from the installation URL

#### Step 3: Configure Environment Variables

Add to your `.env` file:

```bash
# GitHub App Authentication
GITHUB_APP_ID=123456
GITHUB_PRIVATE_KEY_PATH=/path/to/your/private-key.pem
GITHUB_INSTALLATION_ID=12345678

# Optional: Default repository
GITHUB_REPO_OWNER=your-org
GITHUB_REPO_NAME=your-repo
```

## 🛠️ Configuration

### Server Configuration

The GitHub integration can be configured via `config/server_config.json`:

```json
{
  "github": {
    "authentication": {
      "token": "${GITHUB_TOKEN:-}",
      "app_id": "${GITHUB_APP_ID:-}",
      "private_key_path": "${GITHUB_PRIVATE_KEY_PATH:-}",
      "installation_id": "${GITHUB_INSTALLATION_ID:-}"
    },
    "api": {
      "base_url": "${GITHUB_API_URL:-https://api.github.com}",
      "timeout": 30,
      "retry_attempts": 3,
      "retry_delay": 1.0,
      "rate_limit_buffer": 100
    },
    "repository": {
      "current_owner": "${GITHUB_REPO_OWNER:-}",
      "current_repo": "${GITHUB_REPO_NAME:-}",
      "auto_index_on_switch": true
    },
    "issues": {
      "default_state": "open",
      "max_fetch_count": 50,
      "analysis": {
        "search_limit": 5,              // Reduced for token optimization (v0.3.4.post1)
        "context_expansion": true,       // Include surrounding code chunks
        "include_dependencies": false,   // Disabled for token optimization (v0.3.4.post1)
        "code_similarity_threshold": 0.7,
        "response_verbosity": "summary", // Return summary not full results (v0.3.4.post1)
        "include_raw_search_results": false,
        "max_relevant_files": 5,
        "truncate_content": true,
        "content_preview_length": 200,
        "progressive_context": {         // New in v0.3.4.post1
          "enabled": true,
          "default_level": "class",      // Default context granularity
          "bug_level": "method",         // Bugs need method-level detail
          "feature_level": "file"        // Features need file-level overview
        }
      }
    },
    "safety": {
      "dry_run_by_default": true,
      "require_confirmation": true,
      "max_files_per_pr": 10,
      "blocked_file_patterns": [
        "*.key", "*.pem", "*.env", "secrets/**", ".github/workflows/**"
      ]
    }
  }
}
```

## 📚 Usage Examples

### MCP Tools via Claude Code

The GitHub integration provides 10 MCP tools accessible through Claude Code's natural language interface:

```
# Authentication and repository management
"List my repositories"
"Switch to repository owner/repo-name"
"Check GitHub health status"

# Issue management
"Show me open issues with bug label"
"Get details for issue #123"
"Create a test issue with title 'Bug in login' and labels 'bug,urgent'"

# Analysis and resolution
"Analyze issue #123 using RAG search"
"Generate fix suggestions for issue #123"
"Resolve issue #123 in dry-run mode"
"Create a pull request for issue fix"
```

### HTTP API Testing

All GitHub functionality is also available via HTTP endpoints for testing and integration:

#### Basic Authentication Test
```bash
# Test GitHub connection
curl http://localhost:8081/github/health

# List repositories
curl http://localhost:8081/github/repositories

# Switch repository context
curl -X POST http://localhost:8081/github/switch_repository \
  -H "Content-Type: application/json" \
  -d '{"owner": "myorg", "repo": "myproject"}'
```

#### Issue Management via HTTP
```bash
# Fetch issues
curl "http://localhost:8081/github/issues?state=open&limit=5"

# Get specific issue
curl http://localhost:8081/github/issues/123

# Create test issue
curl -X POST http://localhost:8081/github/issues \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Issue: Sample bug for testing",
    "body": "This is a test issue for GitHub integration testing.",
    "labels": ["test", "bug"],
    "assignees": ["username"]
  }'

# Add comment to issue
curl -X POST http://localhost:8081/github/issues/123/comment \
  -H "Content-Type: application/json" \
  -d '{
    "issue_number": 123,
    "body": "Thanks for reporting this issue! We are investigating."
  }'
```

#### Analysis and Fix Generation via HTTP
```bash
# Analyze issue with RAG search
curl -X POST http://localhost:8081/github/issues/123/analyze

# Generate fix suggestions
curl -X POST http://localhost:8081/github/issues/123/suggest_fix

# Test resolution workflow (dry run)
curl -X POST http://localhost:8081/github/issues/123/resolve?dry_run=true

# Create pull request
curl -X POST http://localhost:8081/github/pull_requests \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Fix: Resolve authentication bug",
    "body": "This PR fixes the issue described in #123",
    "head": "fix-auth-bug",
    "base": "main"
  }'
```

### Complete Testing Script

Use the provided script for comprehensive HTTP API testing:

```bash
# Make the script executable
chmod +x scripts/test_github_http_api.sh

# Run comprehensive tests
./scripts/test_github_http_api.sh

# Test specific operations
./scripts/test_github_http_api.sh --test-auth
./scripts/test_github_http_api.sh --test-issues  
./scripts/test_github_http_api.sh --test-analysis
```

### Basic MCP Workflow

```bash
# 1. Switch to repository context (MCP)
"Switch to repository myorg/myproject"

# 2. List recent issues (MCP)
"Show me the last 10 open issues"

# 3. Get detailed issue information (MCP)
"Get details for issue #123"

# 4. Analyze issue with RAG search (MCP)
"Analyze issue #123 using RAG search to find related code"

# 5. Generate fix suggestions (MCP)
"Generate fix suggestions for issue #123"

# 6. Test resolution workflow (MCP dry run)
"Resolve issue #123 in dry-run mode to see what would be done"

# 7. Create actual PR when ready (MCP)
"Create a pull request to resolve issue #123"
```

### Repository Management

```bash
# List your repositories
github_list_repositories

# List organization repositories
github_list_repositories owner="myorg"

# Switch context to different repository
github_switch_repository owner="myorg" repo="backend-api"

# Add comment to issue
github_add_comment issue_number=123 body="Investigation complete. Fix incoming!"
```

### Issue Analysis

```bash
# Analyze a bug report
github_analyze_issue issue_number=456

# The analysis includes:
# - Issue classification (bug, feature, performance, etc.)
# - Error pattern extraction
# - Code reference identification
# - RAG search results for related code
# - Confidence scoring
# - Implementation recommendations
```

### Fix Generation

```bash
# Generate automated fix suggestions
github_suggest_fix issue_number=789

# Suggestions include:
# - Code modifications with templates
# - New file recommendations
# - Test suggestions
# - Safety warnings
# - Implementation feasibility assessment
```

### Pull Request Creation

```bash
# Manual PR creation
github_create_pull_request \
  title="Fix authentication bug in login flow" \
  body="This PR fixes the issue described in #123" \
  head="fix-auth-bug" \
  base="main"
```

## 📋 Sub-Issues Management (v0.3.4.post4)

GitHub sub-issues (task lists) enable hierarchical issue organization for complex features and projects.

### Overview

Sub-issues provide:
- **Hierarchical Organization**: Parent issues can have multiple sub-tasks
- **Progress Tracking**: See completion status across all sub-issues
- **Dependency Management**: Understand task relationships
- **Projects V2 Integration**: Bulk-add sub-issues to project boards

### Creating Sub-Issues

#### Method 1: Create and Link New Sub-Issue
```bash
# Via Claude Code (MCP)
"Create a sub-task for issue #123 titled 'Implement user authentication'"

# Via HTTP API
curl -X POST http://localhost:8081/github/create_sub_issue \
  -H "Content-Type: application/json" \
  -d '{
    "parent_issue_number": 123,
    "title": "Implement user authentication",
    "body": "Add OAuth2 authentication flow",
    "labels": ["enhancement", "backend"]
  }'
```

#### Method 2: Link Existing Issues
```bash
# Via Claude Code
"Add issue #456 as a sub-issue of #123"

# Via HTTP API
curl -X POST http://localhost:8081/github/add_sub_issue \
  -H "Content-Type: application/json" \
  -d '{
    "parent_issue_number": 123,
    "sub_issue_number": 456
  }'
```

### Managing Sub-Issues

#### List All Sub-Issues
```bash
# Via Claude Code
"List all sub-issues for issue #123"
"Show me the sub-tasks of #123"

# Via HTTP API
curl -X POST http://localhost:8081/github/list_sub_issues \
  -H "Content-Type: application/json" \
  -d '{"parent_issue_number": 123}'

# Response example:
{
  "parent_issue": 123,
  "sub_issues_count": 3,
  "sub_issues": [
    {
      "number": 456,
      "title": "Implement user authentication",
      "state": "open",
      "assignee": "developer1"
    },
    {
      "number": 457,
      "title": "Add database migrations",
      "state": "closed",
      "assignee": "developer2"
    }
  ]
}
```

#### Remove Sub-Issue Relationship
```bash
# Via Claude Code
"Remove sub-issue #456 from parent #123"

# Via HTTP API
curl -X POST http://localhost:8081/github/remove_sub_issue \
  -H "Content-Type: application/json" \
  -d '{
    "parent_issue_number": 123,
    "sub_issue_number": 456
  }'
```

#### Reorder Sub-Issues
```bash
# Via Claude Code
"Reorder sub-issues for #123: put #458 first, then #456, then #457"

# Via HTTP API
curl -X POST http://localhost:8081/github/reorder_sub_issues \
  -H "Content-Type: application/json" \
  -d '{
    "parent_issue_number": 123,
    "sub_issue_numbers": [458, 456, 457]
  }'
```

### Re-parenting Issues

Move a sub-issue to a different parent:
```bash
# Via Claude Code
"Move sub-issue #456 from parent #123 to parent #789"

# Via HTTP API (with replace_parent=true)
curl -X POST http://localhost:8081/github/add_sub_issue \
  -H "Content-Type: application/json" \
  -d '{
    "parent_issue_number": 789,
    "sub_issue_number": 456,
    "replace_parent": true
  }'
```

### Projects V2 Integration

Automatically add all sub-issues to a project:
```bash
# Via Claude Code
"Add all sub-issues of #123 to project PVT_kwDOAdYevc4A7ABC"

# Via HTTP API
curl -X POST http://localhost:8081/github/add_sub_issues_to_project \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "PVT_kwDOAdYevc4A7ABC",
    "parent_issue_number": 123
  }'

# Response includes:
{
  "added_count": 3,
  "failed_count": 0,
  "sub_issues": [
    {
      "issue_number": 456,
      "item_id": "PVTI_lADOAdYevc4A7ABC",
      "applied_fields": {
        "Status": "In Progress",
        "Priority": "High"
      }
    }
  ]
}
```

### Common Workflows

#### Breaking Down a Feature
```bash
# 1. Create parent issue
"Create issue 'Implement user dashboard' with label 'epic'"

# 2. Create sub-tasks
"Create sub-issue for #100 titled 'Design dashboard UI'"
"Create sub-issue for #100 titled 'Implement API endpoints'"
"Create sub-issue for #100 titled 'Add unit tests'"
"Create sub-issue for #100 titled 'Write documentation'"

# 3. Add all to project board
"Add all sub-issues of #100 to the current sprint project"
```

#### Organizing Existing Issues
```bash
# 1. Create a parent issue for organization
"Create issue 'Q1 Performance Improvements' as an epic"

# 2. Link existing issues as sub-issues
"Add issue #45 as sub-issue of #200"  # Database optimization
"Add issue #67 as sub-issue of #200"  # Cache implementation
"Add issue #89 as sub-issue of #200"  # Query optimization

# 3. Reorder by priority
"Reorder sub-issues for #200: #89, #45, #67"
```

### Best Practices

1. **Use Clear Hierarchies**: Keep parent issues high-level, sub-issues specific
2. **Consistent Labeling**: Sub-issues inherit parent labels by default
3. **Progress Tracking**: Close sub-issues to track parent progress
4. **Avoid Deep Nesting**: GitHub supports one level of sub-issues only
5. **Projects Integration**: Use bulk-add for better project management

### Technical Notes

- **API Implementation**: Uses GitHub REST API with preview headers (PyGithub doesn't support sub-issues)
- **Parameter Naming**: API uses `sub_issue_id` but accepts issue numbers for current repository
- **Limitations**: One level of hierarchy only (no sub-sub-issues)
- **Permissions**: Requires write access to both parent and sub-issues

## 📋 Enhanced Issue Management (v0.3.4.post5)

The v0.3.4.post5 release adds comprehensive issue lifecycle management tools, milestone support, and advanced filtering capabilities for complete release management workflows.

### Issue Lifecycle Management

#### Close Issues with State Reasons
```bash
# Via Claude Code (MCP)
"Close issue #123 as completed"
"Close issue #456 as not planned with comment 'Duplicate of #123'"
"Close issue #789 as duplicate"

# Via HTTP API
curl -X PATCH http://localhost:8081/github/issues/123/close \
  -H "Content-Type: application/json" \
  -d '{
    "reason": "completed",
    "comment": "Fixed in PR #124"
  }'

# State reasons:
# - completed: Issue was resolved
# - not_planned: Won't be implemented
# - duplicate: Duplicate of another issue
```

#### Assign/Unassign Users
```bash
# Via Claude Code
"Assign issue #123 to developer1 and developer2"
"Unassign developer1 from issue #123"
"Assign issue #456 to @me"

# Via HTTP API - Assign users
curl -X POST http://localhost:8081/github/issues/123/assignees \
  -H "Content-Type: application/json" \
  -d '{
    "assignees": ["developer1", "developer2"],
    "operation": "add"
  }'

# Unassign users
curl -X POST http://localhost:8081/github/issues/123/assignees \
  -H "Content-Type: application/json" \
  -d '{
    "assignees": ["developer1"],
    "operation": "remove"
  }'
```

#### Update Issue Properties
```bash
# Via Claude Code
"Update issue #123 title to 'Critical: Authentication bug'"
"Add labels 'priority-high' and 'security' to issue #123"
"Set milestone v0.3.5 for issue #123"

# Via HTTP API - Update multiple properties
curl -X PATCH http://localhost:8081/github/issues/123 \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Critical: Authentication bug",
    "body": "Updated description with more details...",
    "labels": ["bug", "priority-high", "security"],
    "milestone": 1,
    "assignees": ["developer1", "developer2"]
  }'

# Remove milestone
curl -X PATCH http://localhost:8081/github/issues/123 \
  -H "Content-Type: application/json" \
  -d '{"milestone": null}'
```

### Milestone Management

#### List Milestones
```bash
# Via Claude Code
"List all open milestones"
"Show milestones sorted by completeness"
"List all milestones including closed ones"

# Via HTTP API
curl "http://localhost:8081/github/milestones?state=open&sort=due_on&direction=asc"

# Response example:
{
  "milestones": [
    {
      "number": 1,
      "title": "v0.3.5 - Adaptive Search Intelligence",
      "description": "Smart query understanding and optimization",
      "state": "open",
      "due_on": "2025-07-01T00:00:00Z",
      "open_issues": 15,
      "closed_issues": 5,
      "completion_percentage": 25
    }
  ]
}
```

#### Create Milestones
```bash
# Via Claude Code
"Create milestone 'v0.3.5 - Adaptive Search Intelligence' due July 1st"
"Create milestone for Q2 2025 release"

# Via HTTP API
curl -X POST http://localhost:8081/github/milestones \
  -H "Content-Type: application/json" \
  -d '{
    "title": "v0.3.5 - Adaptive Search Intelligence",
    "description": "Implement smart query understanding and dynamic optimization",
    "due_on": "2025-07-01"
  }'
```

#### Update/Close Milestones
```bash
# Via Claude Code
"Update milestone #1 due date to August 1st"
"Close milestone #1 as completed"

# Via HTTP API - Update
curl -X PATCH http://localhost:8081/github/milestones/1 \
  -H "Content-Type: application/json" \
  -d '{
    "title": "v0.3.5 - Extended deadline",
    "due_on": "2025-08-01"
  }'

# Close milestone
curl -X DELETE http://localhost:8081/github/milestones/1
```

### Enhanced Issue Filtering

The `github_fetch_issues` tool now supports advanced filtering:

#### Filter by Milestone
```bash
# Via Claude Code
"Show all issues in milestone v0.3.5"
"List open issues in milestone #1"

# Via HTTP API - By milestone name
curl "http://localhost:8081/github/issues?milestone=v0.3.5&state=open"

# By milestone number
curl "http://localhost:8081/github/issues?milestone=1&state=open"
```

#### Filter by Assignee
```bash
# Via Claude Code
"Show issues assigned to developer1"
"List unassigned issues"
"Show issues assigned to me"

# Via HTTP API - Assigned to specific user
curl "http://localhost:8081/github/issues?assignee=developer1"

# Unassigned issues
curl "http://localhost:8081/github/issues?assignee=none"
```

#### Filter by Date Range
```bash
# Via Claude Code
"Show issues created after January 1st, 2025"
"List issues updated since last week"

# Via HTTP API - ISO date format
curl "http://localhost:8081/github/issues?since=2025-01-01T00:00:00Z"
```

#### Sort and Order
```bash
# Via Claude Code
"Show issues sorted by most recently updated"
"List issues sorted by comment count descending"

# Via HTTP API
curl "http://localhost:8081/github/issues?sort=updated&direction=desc"
curl "http://localhost:8081/github/issues?sort=comments&direction=desc"

# Sort options: created, updated, comments
# Direction: asc, desc
```

#### Combined Filters
```bash
# Via Claude Code
"Show unassigned bugs in milestone v0.3.5 sorted by priority"

# Via HTTP API - Complex query
curl "http://localhost:8081/github/issues?state=open&labels=bug&milestone=v0.3.5&assignee=none&sort=created&direction=desc"
```

### Advanced Search with GitHub Syntax

The `github_search_issues` tool supports full GitHub search syntax:

```bash
# Via Claude Code
"Search for issues: is:open milestone:v0.3.5 label:priority-high"
"Find all unassigned bugs created this month"
"Search for issues mentioning 'performance' in authentication module"

# Via HTTP API
curl -X POST http://localhost:8081/github/issues/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "is:issue is:open milestone:v0.3.5 label:priority-high",
    "sort": "created",
    "order": "desc"
  }'

# Common search patterns:
# - "is:issue is:open no:assignee" - Unassigned open issues
# - "is:issue label:bug created:>2025-01-01" - Recent bugs
# - "is:issue is:open milestone:\"v0.3.5\"" - Issues in specific milestone
# - "is:issue assignee:@me state:open" - My open issues
```

### Complete Workflow Examples

#### Release Management Workflow
```bash
# 1. Create milestone for release
"Create milestone 'v0.3.6 - Query Enhancement' due August 15th"

# 2. List all issues for planning
"Show all open enhancement issues sorted by priority"

# 3. Assign issues to milestone
"Update issue #201 with milestone v0.3.6"
"Update issue #202 with milestone v0.3.6"
"Update issue #203 with milestone v0.3.6"

# 4. Assign developers
"Assign issue #201 to developer1"
"Assign issue #202 to developer2"
"Assign issue #203 to developer1 and developer2"

# 5. Track progress
"Show all issues in milestone v0.3.6"
"List completed issues in milestone v0.3.6"

# 6. Close completed work
"Close issue #201 as completed with comment 'Implemented in PR #250'"
"Close issue #202 as completed"

# 7. Close milestone when done
"Close milestone #2 as completed"
```

#### Team Workload Management
```bash
# 1. View unassigned work
"Show all unassigned open issues"
"List unassigned bugs with high priority"

# 2. Check developer workload
"Show issues assigned to developer1"
"List open issues assigned to developer2 in current milestone"

# 3. Balance assignments
"Unassign developer1 from issue #123"
"Assign issue #123 to developer2"

# 4. Bulk operations
"Search for issues: is:open assignee:developer1 label:low-priority"
"Update issue #456 removing developer1 from assignees"
```

#### Issue Triage Workflow
```bash
# 1. Find untriaged issues
"Search for issues: is:open no:label no:milestone"

# 2. Classify and prioritize
"Update issue #300 with labels 'bug', 'priority-high'"
"Set milestone v0.3.5 for issue #300"
"Assign issue #300 to developer1"

# 3. Handle duplicates
"Close issue #301 as duplicate with comment 'Duplicate of #300'"

# 4. Defer issues
"Close issue #302 as not_planned with comment 'Out of scope for current roadmap'"
```

### API Reference for New Tools

#### Close Issue
**Tool**: `github_close_issue`
**HTTP**: `PATCH /github/issues/{id}/close`

Parameters:
- `issue_number` (required): Issue to close
- `reason` (optional): "completed", "not_planned", "duplicate" (default: "completed")
- `comment` (optional): Comment to add before closing

#### Assign/Unassign Issue
**Tool**: `github_assign_issue`
**HTTP**: `POST /github/issues/{id}/assignees`

Parameters:
- `issue_number` (required): Issue number
- `assignees` (required): List of usernames
- `operation` (optional): "add" or "remove" (default: "add")

#### Update Issue
**Tool**: `github_update_issue`
**HTTP**: `PATCH /github/issues/{id}`

Parameters:
- `issue_number` (required): Issue to update
- `title` (optional): New title
- `body` (optional): New description
- `state` (optional): "open" or "closed"
- `labels` (optional): List of labels
- `assignees` (optional): List of assignees
- `milestone` (optional): Milestone number or null

#### Search Issues
**Tool**: `github_search_issues`
**HTTP**: `POST /github/issues/search`

Parameters:
- `query` (required): GitHub search syntax query
- `sort` (optional): "comments", "created", "updated"
- `order` (optional): "asc" or "desc" (default: "desc")

#### Milestone Operations
**Tools**: `github_list_milestones`, `github_create_milestone`, `github_update_milestone`, `github_close_milestone`
**HTTP**: Various endpoints under `/github/milestones`

### Best Practices for Issue Management

1. **Consistent Labeling**: Use a standard label taxonomy (e.g., type/bug, priority/high)
2. **Milestone Planning**: Create milestones before sprint/release planning
3. **Assignment Balance**: Monitor workload across team members
4. **Regular Triage**: Schedule regular issue triage sessions
5. **Clear Close Reasons**: Always specify why issues are closed
6. **Automation**: Use search queries to find issues needing attention

## 📋 Migration Guide for Enhanced Issue Management

If you're upgrading to v0.3.4.post5, here's how to leverage the new features with your existing issues:

### Creating Milestones for Existing Releases

```bash
# 1. List your releases/versions that need milestones
"Show me all closed issues with label 'release'"

# 2. Create milestones for past releases (documentation purposes)
"Create milestone 'v0.3.0 - GitHub Integration' as already completed"
"Create milestone 'v0.3.4 - GitHub Projects V2'"

# 3. Create milestones for upcoming releases
"Create milestone 'v0.3.5 - Adaptive Search Intelligence' due July 15th"
"Create milestone 'v0.3.6 - Query Enhancement' due August 30th"
```

### Bulk Update Issues with Milestones

```bash
# 1. Find issues for a specific release
"Search for issues: is:issue label:v0.3.5"

# 2. Update each issue with the appropriate milestone
"Update issue #100 with milestone v0.3.5"
"Update issue #101 with milestone v0.3.5"
"Update issue #102 with milestone v0.3.5"

# 3. Verify milestone assignment
"Show all issues in milestone v0.3.5"
```

### Setting Up Release Views

```bash
# 1. Create release dashboard project
"Create project 'Release Dashboard' from roadmap template"

# 2. Add all issues from current milestone
"Show all issues in milestone v0.3.5"
"Add issue #100 to project with smart field assignment"

# 3. Set up milestone-based filters in project
# (Use GitHub UI to create views filtered by milestone)
```

### Organizing Unassigned Work

```bash
# 1. Find all unassigned issues
"Show all unassigned open issues"
"Search for issues: is:open no:assignee no:milestone"

# 2. Triage and assign milestones
"Update issue #200 with milestone v0.3.6 and labels 'enhancement'"
"Assign issue #200 to developer1"

# 3. Defer non-priority items
"Close issue #201 as not_planned with comment 'Deferred to future release'"
```

### Troubleshooting Common Issues

#### "Milestone not found" Error
- **Cause**: Milestone name doesn't match exactly or doesn't exist
- **Solution**: List milestones first to verify exact names
```bash
"List all open milestones"
"Update issue #123 with milestone 'Exact Milestone Name'"
```

#### Bulk Operations Timing Out
- **Cause**: Too many operations in sequence
- **Solution**: Use search to filter first, then update in smaller batches
```bash
"Search for issues: is:open label:bug no:milestone"
# Update 5-10 at a time rather than all at once
```

#### Assignment Conflicts
- **Cause**: Trying to assign users who don't have repository access
- **Solution**: Verify user has access before assigning
```bash
"List repository collaborators"  # Use GitHub UI
"Assign issue #123 to valid-username"
```

#### Search Not Finding Expected Issues
- **Cause**: Incorrect search syntax or filters
- **Solution**: Start with simple queries and add filters gradually
```bash
# Start simple
"Search for issues: is:issue is:open"
# Add filters one at a time
"Search for issues: is:issue is:open label:bug"
"Search for issues: is:issue is:open label:bug milestone:v0.3.5"
```

## 🔍 Analysis Features

### Issue Classification

The system automatically classifies issues into categories:

- **Bug**: Error reports, crashes, unexpected behavior
- **Feature**: New functionality requests
- **Performance**: Speed, memory, optimization issues
- **Documentation**: Docs, guides, examples

### Information Extraction

From issue text, the analyzer extracts:

- **Error messages** and stack traces
- **Code snippets** and references
- **File paths** and function names
- **Feature requirements**
- **Technical keywords**

### RAG Integration

The analyzer performs multiple search strategies:

- **Code search** for function/class references
- **Error search** for similar issues
- **General search** for contextual understanding
- **Dependency analysis** for related components

### Confidence Scoring

Results include confidence metrics:

- **High (>80%)**: Strong matches found, likely similar issue resolved before
- **Medium (50-80%)**: Related patterns identified, moderate confidence
- **Low (<50%)**: Limited matches, requires manual investigation

## 🚀 Token Optimization (v0.3.4.post1)

The GitHub integration implements sophisticated token optimization that reduces token consumption by 70-85% while maintaining full analysis quality. This is achieved through multiple optimization strategies working together.

### Key Optimizations

1. **Query Deduplication & Limiting**
   - Generates up to 17 potential queries from issue content
   - Deduplicates similar queries
   - Limits to maximum 8 unique queries
   - Reduces redundant searches by ~50%

2. **Search Result Limiting**
   - Results per query: 10 → 5 (50% reduction)
   - Max relevant files: 5 (prevents information overload)
   - Content preview: 200 characters (focused excerpts)

3. **Progressive Context**
   - Dynamically adjusts context level based on issue type:
     - **Bugs**: Method-level detail (finest granularity)
     - **Features**: File-level overview (broader context)
     - **Default**: Class-level (balanced approach)
   - Reduces token usage by ~50% through intelligent chunking

4. **Dependency Exclusion**
   - `include_dependencies: false` saves ~20% tokens
   - Focuses on direct matches, not transitive dependencies

5. **Summary Mode**
   - `response_verbosity: "summary"` returns only key findings
   - Excludes raw search results from response
   - Maintains full internal analysis quality

### How the Analyzer Works

The issue analyzer follows a sophisticated multi-step process:

```
1. Extract Information (from issue title/body)
   ├── Title → Primary query
   ├── Errors (up to 3) → Error queries
   ├── Functions (up to 3) → Function queries
   ├── Classes (up to 3) → Class queries
   ├── Features (up to 2) → Feature queries
   └── Keywords (up to 5) → Keyword queries

2. Query Optimization
   ├── Deduplicate queries (remove similar)
   ├── Limit to 8 unique queries
   └── Assign appropriate search type

3. Execute Searches
   ├── Error/Function/Class → search_code
   ├── Features → search_docs
   └── Keywords/Title → search_code (fallback to search_docs)

4. Apply Progressive Context
   ├── Bug issues → Method-level detail
   ├── Feature requests → File-level overview
   └── Other → Class-level balance

5. Generate Summary
   ├── Relevant files (max 5)
   ├── Code patterns found
   ├── Confidence scoring
   └── Actionable recommendations
```

### Real-World Impact

For a typical bug report:
- **Old approach**: ~90,000 tokens (10 queries × 10 results × 750 tokens × 1.2)
- **New approach**: ~15,000 tokens (8 queries × 5 results × 750 tokens × 0.5)
- **Reduction**: 83% ✓

### Configuration Reference

All settings in `server_config.json` under `github.issues.analysis`:

```json
{
  "search_limit": 5,                    // Results per search
  "context_expansion": true,            // Include surrounding chunks
  "include_dependencies": false,        // Exclude dependency expansion
  "response_verbosity": "summary",      // Summary mode
  "progressive_context": {
    "enabled": true,                    // Enable progressive context
    "default_level": "class",           // Default granularity
    "bug_level": "method",              // Bug-specific level
    "feature_level": "file"             // Feature-specific level
  }
}
```

## 🛡️ Safety Features

### Dry Run Mode

All destructive operations default to dry-run mode:

```bash
# Safe - shows what would be done
github_resolve_issue issue_number=123

# Explicit dry run
github_resolve_issue issue_number=123 dry_run=true

# Actual execution (requires explicit confirmation)
github_resolve_issue issue_number=123 dry_run=false
```

### File Protection

Certain file patterns are blocked from modification:

- Secrets and keys (`*.key`, `*.pem`, `*.env`)
- CI/CD workflows (`.github/workflows/**`)
- Security-sensitive directories (`secrets/**`)

### Rate Limiting

Built-in GitHub API rate limit handling:

- Automatic rate limit detection
- Intelligent request spacing
- Retry logic with exponential backoff
- Configurable rate limit buffer

### Audit Logging

All GitHub operations are logged for compliance:

- Repository access
- Issue analysis
- PR creation
- Workflow execution

## 🚨 Troubleshooting

### Authentication Issues

```bash
# Test authentication
github_list_repositories

# Expected response for success:
{
  "repositories": [...],
  "count": 10,
  "owner": "authenticated_user"
}

# Error response for auth failure:
{
  "error": "Failed to list repositories: Bad credentials"
}
```

### Common Issues

#### "GitHub integration not available"
- **Cause**: PyGithub/GitPython not installed
- **Solution**: Install dependencies: `pip install PyGithub GitPython`

#### "No repository context set"
- **Cause**: Haven't switched to a repository
- **Solution**: Run `github_switch_repository` first

#### "Bad credentials"
- **Cause**: Invalid or expired token
- **Solution**: Check token permissions and expiration

#### "Rate limit exceeded"
- **Cause**: Too many API requests
- **Solution**: Wait for rate limit reset or increase buffer

### Debug Mode

Enable detailed logging:

```bash
export QDRANT_LOG_LEVEL=DEBUG
```

## 🔗 API Reference

### Complete Tool Coverage

The GitHub integration provides 29 MCP tools with corresponding HTTP endpoints:

#### Core GitHub Tools (v0.3.0)
| MCP Tool | HTTP Endpoint | Purpose |
|----------|---------------|---------|
| `github_list_repositories` | `GET /github/repositories` | List user/org repositories |
| `github_switch_repository` | `POST /github/switch_repository` | Set repository context |
| `github_fetch_issues` | `GET /github/issues` | Fetch repository issues (enhanced in v0.3.4.post5) |
| `github_get_issue` | `GET /github/issues/{id}` | Get issue details |
| `github_create_issue` | `POST /github/issues` | Create new issues |
| `github_add_comment` | `POST /github/issues/{id}/comment` | Add comments to issues |
| `github_analyze_issue` | `POST /github/issues/{id}/analyze` | RAG-powered analysis |
| `github_suggest_fix` | `POST /github/issues/{id}/suggest_fix` | Generate fix suggestions |
| `github_create_pull_request` | `POST /github/pull_requests` | Create pull requests |
| `github_resolve_issue` | `POST /github/issues/{id}/resolve` | End-to-end resolution |

#### Enhanced Issue Management (v0.3.4.post5)
| MCP Tool | HTTP Endpoint | Purpose |
|----------|---------------|---------|
| `github_close_issue` | `PATCH /github/issues/{id}/close` | Close issues with state reasons |
| `github_assign_issue` | `POST /github/issues/{id}/assignees` | Assign/unassign users |
| `github_update_issue` | `PATCH /github/issues/{id}` | Update any issue property |
| `github_search_issues` | `POST /github/issues/search` | Advanced search with GitHub syntax |
| `github_list_milestones` | `GET /github/milestones` | List repository milestones |
| `github_create_milestone` | `POST /github/milestones` | Create milestones |
| `github_update_milestone` | `PATCH /github/milestones/{number}` | Update milestone properties |
| `github_close_milestone` | `DELETE /github/milestones/{number}` | Close milestones |

#### Projects V2 Management (v0.3.4)
| MCP Tool | HTTP Endpoint | Purpose |
|----------|---------------|---------|
| `github_list_projects` | `GET /github/projects` | List GitHub Projects V2 |
| `github_create_project` | `POST /github/projects` | Create new projects |
| `github_get_project` | `GET /github/projects/{number}` | Get project details |
| `github_add_project_item` | `POST /github/projects/{id}/items` | Add issues/PRs to projects |
| `github_update_project_item` | `PATCH /github/projects/{id}/items/{item_id}` | Update item fields |
| `github_create_project_field` | `POST /github/projects/{id}/fields` | Create custom fields |
| `github_get_project_status` | `GET /github/projects/{id}/status` | Get project metrics |
| `github_delete_project` | `DELETE /github/projects/{id}` | Delete projects |

#### Sub-Issues Management (v0.3.4.post4)
| MCP Tool | HTTP Endpoint | Purpose |
|----------|---------------|---------|
| `github_list_sub_issues` | `POST /github/list_sub_issues` | List sub-issues |
| `github_add_sub_issue` | `POST /github/add_sub_issue` | Add sub-issue relationship |
| `github_remove_sub_issue` | `POST /github/remove_sub_issue` | Remove sub-issue |
| `github_create_sub_issue` | `POST /github/create_sub_issue` | Create and link sub-issue |
| `github_reorder_sub_issues` | `POST /github/reorder_sub_issues` | Reorder sub-issues |
| `github_add_sub_issues_to_project` | `POST /github/add_sub_issues_to_project` | Bulk add to projects |

### Repository Tools

#### `github_list_repositories`
List repositories for a user/organization.

**Parameters:**
- `owner` (optional): Repository owner, defaults to authenticated user

**Returns:**
- `repositories`: List of repository information
- `count`: Number of repositories
- `owner`: Repository owner

#### `github_switch_repository`
Switch to a repository context.

**Parameters:**
- `owner` (required): Repository owner
- `repo` (required): Repository name

**Returns:**
- `repository`: Repository information
- `message`: Success message

### Issue Tools

#### `github_fetch_issues`
Fetch issues from current repository.

**Parameters:**
- `state` (optional): Issue state (open, closed, all), default: "open"
- `labels` (optional): Filter by labels
- `limit` (optional): Maximum number of issues

**Returns:**
- `issues`: List of issue information
- `count`: Number of issues
- `repository`: Repository name

#### `github_get_issue`
Get detailed issue information.

**Parameters:**
- `issue_number` (required): Issue number

**Returns:**
- `issue`: Detailed issue data including comments
- `repository`: Repository name

#### `github_create_issue`
Create a new GitHub issue.

**Parameters:**
- `title` (required): Issue title
- `body` (optional): Issue description/body
- `labels` (optional): List of label names to apply
- `assignees` (optional): List of usernames to assign

**Returns:**
- `issue`: Created issue information
- `repository`: Repository name
- `message`: Success message

#### `github_add_comment`
Add a comment to an existing GitHub issue.

**Parameters:**
- `issue_number` (required): Issue number to comment on
- `body` (required): Comment body text (supports markdown)

**Returns:**
- `comment`: Comment information including ID and URL
- `repository`: Repository name
- `message`: Success message

#### `github_analyze_issue`
Perform RAG-powered issue analysis.

**Parameters:**
- `issue_number` (required): Issue number

**Returns:**
- `analysis`: Complete analysis results
- `workflow_status`: Analysis workflow status
- `recommendations`: Action recommendations

#### `github_suggest_fix`
Generate fix suggestions for an issue.

**Parameters:**
- `issue_number` (required): Issue number

**Returns:**
- `suggestions`: Fix suggestions and code modifications
- `feasibility`: Implementation feasibility assessment
- `confidence_level`: Suggestion confidence

### Workflow Tools

#### `github_resolve_issue`
Complete issue resolution workflow.

**Parameters:**
- `issue_number` (required): Issue number
- `dry_run` (optional): Dry run mode, default: true

**Returns:**
- `workflow_status`: Resolution workflow status
- `suitability`: Auto-resolution suitability
- `pr_preview` (dry run): Preview of what would be created

#### `github_create_pull_request`
Create a pull request.

**Parameters:**
- `title` (required): PR title
- `body` (required): PR description
- `head` (required): Head branch
- `base` (optional): Base branch, default: "main"
- `files` (optional): File references

**Returns:**
- `pull_request`: Created PR information
- `message`: Success message

## 📈 Best Practices

### Repository Setup

1. **Index your repository** before issue analysis for better results
2. **Use descriptive issue titles** for better classification
3. **Include code snippets** in issue descriptions for better analysis
4. **Tag issues appropriately** for filtering

### Workflow Optimization

1. **Start with analysis** before generating fixes
2. **Use dry-run mode** to review suggestions
3. **Test fixes locally** before creating PRs
4. **Review safety warnings** carefully

### Security Considerations

1. **Use least-privilege tokens** with minimal required scopes
2. **Rotate tokens regularly** for security
3. **Monitor audit logs** for unusual activity
4. **Keep private keys secure** for GitHub Apps

### Performance Tips

1. **Limit search results** to avoid token limits
2. **Use specific queries** for better relevance
3. **Enable context expansion** for comprehensive analysis
4. **Monitor rate limits** to avoid throttling

## 🧪 Testing & Development Patterns

### Development Workflow Testing

Create and test issues for development:

```bash
# Via Claude Code (MCP)
"Create a test issue with title 'Testing GitHub integration' and labels 'test'"
"Switch to repository ancoleman/qdrant-rag-mcp"
"Analyze issue #1 to test RAG search functionality"

# Via HTTP API
curl -X POST http://localhost:8081/github/issues \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test: Performance issue in search function",
    "body": "The search function is slow when processing large datasets...",
    "labels": ["performance", "test"]
  }'
```

### MCP Tool Testing Patterns

Test all 9 GitHub MCP tools systematically:

```bash
# 1. Authentication and setup
"Check GitHub health status"
"List my repositories"

# 2. Repository context
"Switch to repository owner/repo-name"

# 3. Issue operations  
"Show me open issues"
"Get details for issue #123"
"Create a test issue with labels 'bug,test'"
"Add a comment to issue #123 saying 'Thanks for reporting this!'"

# 4. Analysis workflow
"Analyze issue #123 using RAG search"
"Generate fix suggestions for issue #123"

# 5. Resolution workflow
"Test resolving issue #123 in dry-run mode"
"Create a pull request to fix issue #123"
```

### HTTP API Integration Testing

Complete HTTP endpoint testing workflow:

```bash
# Start HTTP server
export $(grep -v '^#' .env | xargs)
python src/http_server.py

# Test authentication endpoints
curl http://localhost:8081/github/health
curl http://localhost:8081/github/repositories

# Test issue lifecycle
curl -X POST http://localhost:8081/github/switch_repository \
  -H "Content-Type: application/json" \
  -d '{"owner": "myorg", "repo": "myproject"}'

curl -X POST http://localhost:8081/github/issues \
  -H "Content-Type: application/json" \
  -d '{"title": "Test issue", "body": "Testing HTTP API"}'

curl http://localhost:8081/github/issues/1
curl -X POST http://localhost:8081/github/issues/1/analyze
curl -X POST http://localhost:8081/github/issues/1/suggest_fix
```

### Automated Testing with Scripts

Use the testing script for comprehensive validation:

```bash
# Run all tests
./scripts/test_github_http_api.sh

# Test authentication only
./scripts/test_github_http_api.sh --auth-only

# Test issue operations only  
./scripts/test_github_http_api.sh --issues-only

# Test analysis workflow only
./scripts/test_github_http_api.sh --analysis-only
```

## 🔄 Integration with Existing Workflows

### With Claude Code Natural Language

The GitHub tools integrate seamlessly with Claude Code's natural language interface:

```
# Issue analysis and resolution
"Analyze GitHub issue #123 and suggest a fix"
"Create a PR to resolve the authentication bug in issue #456"
"Show me all open bugs in the authentication module"

# Repository and issue management
"Switch to my project repository and show recent issues"
"Create a test issue for the login bug and analyze it"
"List all performance-related issues and analyze the most critical one"
"Add a comment to issue #123 with the analysis results"

# Workflow automation
"Analyze issue #789, generate fixes, and create a PR in dry-run mode"
"Show me the health status of GitHub integration"
"Comment on issue #456 saying the fix has been deployed to staging"
```

### With Existing RAG Features

GitHub analysis leverages all existing RAG capabilities:

- **Code search** with dependency analysis
- **Documentation search** for related guides
- **Context expansion** for comprehensive understanding
- **Enhanced ranking** for better relevance

### With Auto-indexing

Repository switching can trigger automatic indexing:

```json
{
  "github": {
    "repository": {
      "auto_index_on_switch": true
    }
  }
}
```

## 🎯 Advanced Use Cases

### Automated Issue Triage

```bash
# Fetch all open issues
github_fetch_issues state="open"

# Analyze each for classification and priority
for issue in issues:
    github_analyze_issue issue_number=issue.number
```

### Batch Analysis

```bash
# Analyze multiple related issues
github_analyze_issue issue_number=123  # Main bug
github_analyze_issue issue_number=124  # Related feature
github_analyze_issue issue_number=125  # Performance issue
```

### Cross-Repository Analysis

```bash
# Switch between related repositories
github_switch_repository owner="myorg" repo="frontend"
github_analyze_issue issue_number=123

github_switch_repository owner="myorg" repo="backend"
github_analyze_issue issue_number=456
```

## 📊 Monitoring and Analytics

### Health Monitoring

Check GitHub integration health:

```bash
health_check

# Response includes GitHub status:
{
  "github": {
    "status": "healthy",
    "authenticated_user": "username",
    "rate_limit": {
      "remaining": 4500,
      "limit": 5000
    }
  }
}
```

### Usage Analytics

Monitor GitHub operations via audit logs:

```bash
# View GitHub operation logs
./scripts/qdrant-logs --operation github_analyze_issue
./scripts/qdrant-logs --operation github_create_pull_request
```

## 🚀 Future Enhancements

Planned improvements for future versions:

- **Automated PR merging** with confidence thresholds
- **Multi-repository analysis** for complex issues
- **AI-powered code generation** for fixes
- **Integration with CI/CD** for automated testing
- **Issue prediction** based on code changes
- **Custom workflow templates** for different issue types

---

## 📞 Support

For GitHub integration issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review GitHub API documentation
3. File an issue with detailed error logs
4. Include authentication method and configuration (sanitized)

## 🔗 Related Documentation

- [GitHub Workflow Examples](github-workflow-examples.md) - Real-world usage patterns and issue remediation workflows
- [Complete Setup & Usage Guide](complete-setup-and-usage-guide.md)
- [Enhanced RAG Guide](technical/enhanced-qdrant-rag-guide.md)
- [MCP Scope Configuration](mcp-scope-configuration-guide.md)
- [Development Workflow Guide](development-workflow-guide.md)