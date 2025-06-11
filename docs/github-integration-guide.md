# GitHub Integration Guide - v0.3.0

This guide covers the GitHub issue resolution capabilities added in v0.3.0, including setup, authentication, and usage examples.

## Table of Contents

- [🚀 Overview](#-overview)
- [📋 Prerequisites](#-prerequisites)
- [🏗️ Architecture](#️-architecture)
- [🔐 Authentication Setup](#-authentication-setup)
- [🛠️ Configuration](#️-configuration)
- [📚 Usage Examples](#-usage-examples)
- [📋 Sub-Issues Management](#-sub-issues-management-v034post4)
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

The GitHub integration provides 10 MCP tools with corresponding HTTP endpoints:

| MCP Tool | HTTP Endpoint | Purpose |
|----------|---------------|---------|
| `github_list_repositories` | `GET /github/repositories` | List user/org repositories |
| `github_switch_repository` | `POST /github/switch_repository` | Set repository context |
| `github_fetch_issues` | `GET /github/issues` | Fetch repository issues |
| `github_get_issue` | `GET /github/issues/{id}` | Get issue details |
| `github_create_issue` | `POST /github/issues` | Create new issues |
| `github_add_comment` | `POST /github/issues/{id}/comment` | Add comments to issues |
| `github_analyze_issue` | `POST /github/issues/{id}/analyze` | RAG-powered analysis |
| `github_suggest_fix` | `POST /github/issues/{id}/suggest_fix` | Generate fix suggestions |
| `github_create_pull_request` | `POST /github/pull_requests` | Create pull requests |
| `github_resolve_issue` | `POST /github/issues/{id}/resolve` | End-to-end resolution |

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