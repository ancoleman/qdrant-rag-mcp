# Local Prototype: GitHub Issue Resolution

## Two implementation approaches

Since you already have a Python RAG MCP server, here are your options:

**Option A: Two Separate Servers** 
- Keep your existing Python RAG server
- Add new TypeScript GitHub server
- Claude Code orchestrates both

**Option B: Extended Python Server**
- Add GitHub operations to your existing RAG server
- One server, simpler setup
- Faster to prototype

---

## Option A: Separate Servers

### Your existing Python RAG server
Keep running as-is, no changes needed.

### New GitHub Issues server (TypeScript)

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Octokit } from "@octokit/rest";

const server = new McpServer({
  name: "GitHub Issue Manager",
  version: "1.0.0"
});

const octokit = new Octokit({
  auth: process.env.GITHUB_TOKEN
});

// Fetch repository issues
server.tool("fetch_issues", {
  owner: z.string(),
  repo: z.string(),
  state: z.enum(["open", "closed", "all"]).default("open"),
  labels: z.array(z.string()).optional(),
  limit: z.number().default(10)
}, async ({ owner, repo, state, labels, limit }) => {
  const issues = await octokit.rest.issues.listForRepo({
    owner, repo, state,
    labels: labels?.join(","),
    per_page: limit
  });
  
  return {
    content: [{
      type: "text",
      text: JSON.stringify(issues.data, null, 2)
    }]
  };
});

// Get specific issue
server.tool("get_issue", {
  owner: z.string(),
  repo: z.string(),
  issue_number: z.number()
}, async ({ owner, repo, issue_number }) => {
  const issue = await octokit.rest.issues.get({
    owner, repo, issue_number
  });
  
  return {
    content: [{
      type: "text",
      text: JSON.stringify(issue.data, null, 2)
    }]
  };
});

// Create pull request
server.tool("create_pull_request", {
  owner: z.string(),
  repo: z.string(),
  title: z.string(),
  body: z.string(),
  head: z.string(),
  base: z.string().default("main"),
  files: z.array(z.object({
    path: z.string(),
    content: z.string()
  }))
}, async ({ owner, repo, title, body, head, base, files }) => {
  // Create new branch
  const baseBranch = await octokit.rest.repos.getBranch({
    owner, repo, branch: base
  });
  
  await octokit.rest.git.createRef({
    owner, repo,
    ref: `refs/heads/${head}`,
    sha: baseBranch.data.commit.sha
  });
  
  // Create/update files
  for (const file of files) {
    await octokit.rest.repos.createOrUpdateFileContents({
      owner, repo,
      path: file.path,
      message: `Update ${file.path}`,
      content: Buffer.from(file.content).toString('base64'),
      branch: head
    });
  }
  
  // Create PR
  const pr = await octokit.rest.pulls.create({
    owner, repo, title, body, head, base
  });
  
  return {
    content: [{
      type: "text",
      text: `Created PR #${pr.data.number}: ${pr.data.html_url}`
    }]
  };
});

const transport = new StdioServerTransport();
server.connect(transport);
```

### Configuration for Option A

```json
{
  "mcpServers": {
    "code-rag": {
      "command": "python",
      "args": ["/path/to/your/existing/rag-server.py"],
      "env": {
        "REPOSITORY_PATH": "/path/to/repo",
        "QDRANT_URL": "http://localhost:6333"
      }
    },
    "github-issues": {
      "command": "node",
      "args": ["./github-issue-server.js"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

---

## Option B: Extended Python Server

### Add to your existing Python RAG server

```python
from github import Github
import json
import os
from typing import List

class GitHubService:
    def __init__(self, token: str):
        self.github = Github(token)
    
    def fetch_issues(self, owner: str, repo: str, 
                    state: str = "open", labels: list = None, 
                    limit: int = 10):
        repo_obj = self.github.get_repo(f"{owner}/{repo}")
        issues = repo_obj.get_issues(
            state=state,
            labels=labels or []
        )
        
        issue_list = []
        for i, issue in enumerate(issues):
            if i >= limit:
                break
            issue_list.append({
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "html_url": issue.html_url
            })
        
        return issue_list
    
    def get_issue(self, owner: str, repo: str, issue_number: int):
        repo_obj = self.github.get_repo(f"{owner}/{repo}")
        issue = repo_obj.get_issue(issue_number)
        
        return {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body,
            "state": issue.state,
            "labels": [label.name for label in issue.labels],
            "html_url": issue.html_url
        }
    
    def create_pull_request(self, owner: str, repo: str, 
                          title: str, body: str, head: str, 
                          base: str, files: list):
        repo_obj = self.github.get_repo(f"{owner}/{repo}")
        
        # Create branch
        base_branch = repo_obj.get_branch(base)
        repo_obj.create_git_ref(
            ref=f"refs/heads/{head}",
            sha=base_branch.commit.sha
        )
        
        # Create/update files
        for file_data in files:
            try:
                existing = repo_obj.get_contents(file_data["path"], ref=head)
                repo_obj.update_file(
                    path=file_data["path"],
                    message=f"Update {file_data['path']}",
                    content=file_data["content"],
                    sha=existing.sha,
                    branch=head
                )
            except:
                repo_obj.create_file(
                    path=file_data["path"],
                    message=f"Create {file_data['path']}",
                    content=file_data["content"],
                    branch=head
                )
        
        # Create PR
        pr = repo_obj.create_pull(
            title=title, body=body, head=head, base=base
        )
        
        return {
            "number": pr.number,
            "html_url": pr.html_url,
            "title": pr.title
        }

# Add to your existing server
github_service = GitHubService(os.getenv("GITHUB_TOKEN"))

# Add these tools to your MCP server
@mcp.tool()
def fetch_issues(owner: str, repo: str, state: str = "open", 
                labels: List[str] = None, limit: int = 10) -> str:
    """Fetch issues from GitHub repository"""
    issues = github_service.fetch_issues(owner, repo, state, labels, limit)
    return json.dumps(issues, indent=2)

@mcp.tool()
def get_issue(owner: str, repo: str, issue_number: int) -> str:
    """Get detailed GitHub issue information"""
    issue = github_service.get_issue(owner, repo, issue_number)
    return json.dumps(issue, indent=2)

@mcp.tool()
def create_pull_request(owner: str, repo: str, title: str, 
                       body: str, head: str, base: str = "main", 
                       files: List[dict] = None) -> str:
    """Create pull request with files"""
    pr = github_service.create_pull_request(
        owner, repo, title, body, head, base, files or []
    )
    return json.dumps(pr, indent=2)
```

### Configuration for Option B

```json
{
  "mcpServers": {
    "enhanced-rag": {
      "command": "python",
      "args": ["/path/to/your/enhanced-rag-server.py"],
      "env": {
        "REPOSITORY_PATH": "/path/to/repo",
        "QDRANT_URL": "http://localhost:6333",
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

---

## Getting started

### Prerequisites
```bash
# Install dependencies
pip install PyGithub  # for Option B
npm install @octokit/rest  # for Option A

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Usage with Claude Code
```bash
claude-code
> "Show me all open bugs in myorg/myrepo and find related code"
> "Process issue #123 and create a fix"
```

## Recommendation

**For quick prototyping**: Option B (extend Python server)
**For clean architecture**: Option A (separate servers)

Both give identical Claude Code experience!