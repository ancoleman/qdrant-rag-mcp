# GitHub Integration Guide - v0.3.0

This guide covers the GitHub issue resolution capabilities added in v0.3.0, including setup, authentication, and usage examples.

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
        "search_limit": 10,
        "context_expansion": true,
        "include_dependencies": true,
        "code_similarity_threshold": 0.7
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

- [Complete Setup & Usage Guide](complete-setup-and-usage-guide.md)
- [Enhanced RAG Guide](technical/enhanced-qdrant-rag-guide.md)
- [MCP Scope Configuration](mcp-scope-configuration-guide.md)
- [Development Workflow Guide](development-workflow-guide.md)