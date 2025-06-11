# GitHub Projects V2 Integration Guide - v0.3.4

This guide covers the GitHub Projects V2 capabilities added in v0.3.4, including project management, templates, and RAG-enhanced smart field assignment.

## Table of Contents

- [🚀 Overview](#-overview)
- [📋 Prerequisites](#-prerequisites)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Configuration](#️-configuration)
- [📚 Usage Examples](#-usage-examples)
- [🎯 Smart Field Assignment](#-smart-field-assignment)
- [📊 API Reference](#-api-reference)
- [🧪 Testing](#-testing)
- [🏆 Best Practices](#-best-practices)
- [❓ Troubleshooting](#-troubleshooting)

## 🚀 Overview

The GitHub Projects V2 integration transforms project management with intelligent automation:

- **Create** projects with predefined templates (Roadmap, Bug Tracking, Feature Development)
- **Manage** project items with automatic field assignment
- **Track** project status and statistics
- **Analyze** issues using RAG to determine appropriate field values
- **Automate** tedious project management tasks

### Key Features

1. **Project Templates**: Quick project setup with pre-configured fields
2. **Smart Field Assignment**: RAG-powered analysis automatically sets field values
3. **GraphQL Integration**: Full Projects V2 API support
4. **Field Type Support**: Single-select, text, number, date, and iteration fields
5. **Error Handling**: Comprehensive validation and user-friendly error messages

## 📋 Prerequisites

### Required Scopes

Your GitHub Personal Access Token (PAT) must have:
- `project` scope (for project operations)
- `repo` scope (for repository access)

**Important**: Use a Classic PAT, not Fine-Grained, for full Projects V2 support.

### Dependencies

The Projects V2 integration requires:
```bash
pip install "gql[aiohttp]"
```

This is already included in the standard installation.

## 🏗️ Architecture

### Component Design

```
┌─────────────────────────────────────────────────────┐
│                   MCP Tools Layer                    │
│        (github_create_project, etc.)                 │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│           GitHubProjectsManager                      │
│         (GraphQL Adapter Pattern)                    │
│                                                      │
│  • Project CRUD operations                          │
│  • Field management                                 │
│  • Smart issue analysis                             │
│  • Template application                             │
└─────────────┬──────────────┬──────────────┬────────┘
              │              │              │
      GraphQL Client    RAG Analysis    GitHubClient
         (gql)       (Issue Analysis)   (REST API)
```

### GraphQL vs REST

Projects V2 requires GraphQL API:
- REST API: Issues, PRs, repositories (via PyGithub)
- GraphQL API: Projects V2, fields, views (via gql)

## 🛠️ Configuration

### Authentication Setup

```bash
# Set your GitHub token with project scope
export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx

# Or in .env file
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

### Server Configuration

In `config/server_config.json`:
```json
{
  "github": {
    "authentication": {
      "token": "${GITHUB_TOKEN}"
    },
    "projects": {
      "default_template": "bugs",
      "auto_assign_fields": true,
      "max_items_per_project": 1000
    }
  }
}
```

## 📚 Usage Examples

### Creating Projects

#### Basic Project Creation

```bash
# Via Claude Code (MCP)
"Create a GitHub project called 'Q1 Roadmap' for owner ancoleman"

# Via HTTP API
curl -X POST http://localhost:8081/github/projects \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "ancoleman",
    "title": "Q1 Roadmap"
  }'
```

#### Project with Template

```bash
# Via Claude Code
"Create a bug tracking project called 'Bug Tracker' using the bugs template"

# Via HTTP API
curl -X POST http://localhost:8081/github/projects \
  -H "Content-Type: application/json" \
  -d '{
    "owner": "ancoleman",
    "title": "Bug Tracker",
    "template": "bugs"
  }'
```

### Available Templates

#### 1. Bug Tracking Template (`bugs`)
- **Bug Status**: 🆕 New, 🔍 Triaged, 🔧 In Progress, 👀 In Review, ✅ Fixed, ❌ Won't Fix
- **Severity**: 💥 Critical, 🔴 High, 🟡 Medium, 🟢 Low
- **Component**: 🎯 Core, 🔍 Search, 📚 Indexing, 🌐 API, 🧩 Integration, 📖 Documentation
- **Reproduction Steps**: Text field
- **Fix Version**: Text field

#### 2. Implementation Roadmap Template (`roadmap`)
- **Progress**: 📋 Planned, 🚧 In Progress, ✅ Completed, ❌ Cancelled, ⏸️ On Hold
- **Priority**: 🔥 Critical, ⭐ High, 📌 Medium, 📎 Low
- **Epic**: 🏗️ Foundation, 🚀 Enhancement, 🔌 Integration, ⚡ Optimization, 📚 Documentation
- **Complexity**: 1️⃣ Trivial to 5️⃣ Very Complex
- **Target Version**: Text field
- **Due Date**: Date field

#### 3. Feature Development Template (`features`)
- **Stage**: 💡 Ideation, 📋 Planning, 🛠️ Development, 🧪 Testing, 🚀 Deployment, 📊 Monitoring
- **Effort**: XS (<1 day) to XL (>2 weeks)
- **Impact**: 🌟 High, ➕ Medium, ➖ Low
- **Dependencies**: Text field
- **Target Release**: Text field

### Managing Project Items

#### Adding Issues to Projects

```bash
# Basic add
"Add issue #123 to project PVT_kwHOAdYevc4A7ABC"

# With smart field assignment
"Smart add issue #123 to project PVT_kwHOAdYevc4A7ABC"

# HTTP API
curl -X POST http://localhost:8081/github/projects/items/smart \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "PVT_kwHOAdYevc4A7ABC",
    "issue_number": 123
  }'
```

#### Updating Field Values

```bash
# Via Claude Code
"Update project item PVTI_xxxx field Status to 'In Progress'"

# HTTP API
curl -X POST http://localhost:8081/github/projects/items/update \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "PVT_xxxx",
    "item_id": "PVTI_xxxx",
    "field_updates": {
      "Status": "47fc9ee4",
      "Priority": "high"
    }
  }'
```

### Project Status and Analytics

```bash
# Via Claude Code
"Get status for project #5 owned by ancoleman"

# HTTP API
curl http://localhost:8081/github/projects/ancoleman/5/status

# Response includes:
{
  "project": {
    "id": "PVT_xxxx",
    "title": "Bug Tracker",
    "url": "https://github.com/users/ancoleman/projects/5"
  },
  "statistics": {
    "total_items": 42,
    "issues": {
      "open": 30,
      "closed": 12
    },
    "completion_rate": 28.6
  },
  "fields": [...]
}
```

## 🎯 Smart Field Assignment

The most powerful feature is RAG-enhanced field assignment that analyzes issue content to automatically set appropriate field values.

### How It Works

1. **Issue Analysis**: Examines title, body, and labels
2. **Field Detection**: Identifies relevant project fields
3. **Value Matching**: Uses RAG and keyword analysis to determine values
4. **Automatic Assignment**: Sets fields based on confidence levels

### Analysis Examples

#### Bug Priority Detection
```
Issue: "Critical: Application crashes on startup"
Labels: ["bug", "critical", "blocking"]

Analysis Result:
- Severity: 💥 Critical (high confidence)
- Bug Status: 🆕 New
- Component: 🎯 Core
```

#### Component Identification
```
Issue: "Search results are inconsistent"
Body: "The search functionality returns different results..."

Analysis Result:
- Component: 🔍 Search (high confidence)
- Severity: 🟡 Medium
```

#### Feature Classification
```
Issue: "Add dark mode support"
Labels: ["enhancement", "ui"]

Analysis Result:
- Type: ✨ Enhancement
- Impact: 🌟 High
- Stage: 💡 Ideation
```

### Field Type Support

- **Single Select**: Automatically matches option based on keywords
- **Text**: Extracts relevant content (e.g., reproduction steps)
- **Number**: Parses numeric values from content
- **Date**: Extracts dates in various formats

## 📊 API Reference

### MCP Tools

#### `github_create_project`
Create a new GitHub Project V2.

**Parameters:**
- `owner` (required): Username or organization
- `title` (required): Project title
- `template` (optional): Template name (bugs, roadmap, features)

**Returns:**
- `project`: Created project with ID and URL
- `fields`: List of fields (if template used)

#### `github_add_project_item`
Add an issue or PR to a project.

**Parameters:**
- `project_id` (required): Project ID (PVT_xxxx format)
- `issue_number` or `pr_number` (required): Item to add

**Returns:**
- `item`: Added item with ID

#### `github_update_project_item`
Update field values for a project item.

**Parameters:**
- `project_id` (required): Project ID
- `item_id` (required): Item ID
- `field_updates` (required): Dictionary of field:value pairs

#### `github_get_project_status`
Get project statistics and field information.

**Parameters:**
- `owner` (required): Project owner
- `number` (required): Project number

**Returns:**
- `project`: Project information
- `statistics`: Item counts and completion rate
- `fields`: Available fields with options

#### `github_create_project_view`
Create a filtered view in a project.

**Parameters:**
- `project_id` (required): Project ID
- `name` (required): View name
- `filter` (optional): Filter expression
- `sort` (optional): Sort configuration

#### `github_smart_add_project_item`
Add item with intelligent field assignment.

**Parameters:**
- `project_id` (required): Project ID
- `issue_number` (required): Issue number

**Returns:**
- `item`: Added item
- `applied_fields`: Fields that were set
- `suggestions`: All suggested values

### HTTP Endpoints

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/github/projects` | POST | Create project |
| `/github/projects/{owner}/{number}` | GET | Get project |
| `/github/projects/items` | POST | Add item |
| `/github/projects/items/update` | POST | Update item |
| `/github/projects/items/smart` | POST | Smart add |
| `/github/projects/{owner}/{number}/status` | GET | Get status |
| `/github/projects/views` | POST | Create view |

## 🧪 Testing

### Test Script

Use the provided test script:

```bash
# Run all project tests
./scripts/test_github_projects_http_api.sh

# Test smart add functionality
./scripts/test_github_projects_smart_add.sh
```

### Manual Testing Flow

1. **Create a test project**:
```bash
curl -X POST http://localhost:8081/github/projects \
  -d '{"owner": "ancoleman", "title": "Test Project", "template": "bugs"}'
```

2. **Create test issues**:
```bash
curl -X POST http://localhost:8081/github/issues \
  -d '{"title": "Critical bug", "labels": ["bug", "critical"]}'
```

3. **Smart add to project**:
```bash
curl -X POST http://localhost:8081/github/projects/items/smart \
  -d '{"project_id": "PVT_xxxx", "issue_number": 1}'
```

4. **Check results**:
```bash
curl http://localhost:8081/github/projects/ancoleman/1/status
```

## 🏆 Best Practices

### Project Organization

1. **Use templates** for consistent field structure
2. **Name projects clearly** with purpose and timeframe
3. **Limit items per project** to maintain performance
4. **Archive completed projects** to reduce clutter

### Field Design

1. **Keep field names short** but descriptive
2. **Use emojis** for visual distinction
3. **Limit select options** to 6-8 for usability
4. **Include "Other" option** for flexibility

### Smart Assignment

1. **Use descriptive issue titles** for better analysis
2. **Apply consistent labels** for accurate classification
3. **Include keywords** in issue body for field detection
4. **Review assignments** before finalizing

### Error Handling

1. **Check for existing items** before adding
2. **Validate field IDs** before updates
3. **Handle GraphQL errors** gracefully
4. **Log operations** for debugging

## ❓ Troubleshooting

### Common Issues

#### "Personal access token cannot create projects"
- **Cause**: Token missing `project` scope
- **Solution**: Create Classic PAT with project scope

#### "Field name is reserved"
- **Cause**: Using reserved names like "Status"
- **Solution**: Use alternative names (e.g., "Task Status")

#### "Project not found"
- **Cause**: Wrong owner or project number
- **Solution**: Verify owner name and project exists

#### "Invalid content ID format"
- **Cause**: Wrong issue/PR node ID format
- **Solution**: Ensure IDs start with I_ or PR_

### Debug Mode

Enable debug logging:
```bash
export QDRANT_LOG_LEVEL=DEBUG
```

Check logs for GraphQL queries:
```bash
./scripts/qdrant-logs | grep ">>>"
```

### Rate Limiting

Projects V2 uses GraphQL rate limiting:
- Limit: 5,000 points per hour
- Project creation: ~50 points
- Field updates: ~10 points
- Queries: ~1-5 points

## 🔗 Related Documentation

- [GitHub Integration Guide](github-integration-guide.md) - Issue management features
- [Complete Setup Guide](complete-setup-and-usage-guide.md) - Installation and configuration
- [GitHub Workflow Examples](github-workflow-examples.md) - Real-world patterns
- [Advanced RAG Implementation Roadmap](technical/advanced-rag-implementation-roadmap.md) - Future features

---

## 📞 Support

For GitHub Projects integration issues:

1. Verify token has `project` scope
2. Check GraphQL query logs
3. Review error messages for field names
4. File issue with reproduction steps

## 🚀 Future Enhancements

Planned improvements:
- Project templates marketplace
- Bulk operations support
- Cross-project item linking
- Advanced automation rules
- Project insights and analytics
- AI-powered project planning