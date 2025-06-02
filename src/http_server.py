#!/usr/bin/env python3
"""
HTTP REST API wrapper for the Qdrant MCP RAG Server
This allows testing the RAG server functionality via HTTP requests
"""

import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn

# Import our MCP server
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from qdrant_mcp_context_aware import (
    index_code, index_config, index_documentation, index_directory,
    search, search_code, search_docs, reindex_directory,
    detect_changes, get_file_chunks, get_context, switch_project, health_check,
    # GitHub integration functions (v0.3.0)
    github_list_repositories, github_switch_repository, github_fetch_issues,
    github_get_issue, github_create_issue, github_add_comment, github_analyze_issue, github_suggest_fix,
    github_create_pull_request, github_resolve_issue
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing RAG server...")
    # The MCP functions are already initialized when imported
    print("RAG server initialized!")
    yield
    # Shutdown (if needed)
    print("Shutting down...")

app = FastAPI(
    title="Qdrant RAG Server HTTP API", 
    version="0.3.0",
    lifespan=lifespan
)

class IndexCodeRequest(BaseModel):
    file_path: str

class IndexConfigRequest(BaseModel):
    file_path: str

class IndexDirectoryRequest(BaseModel):
    directory: str
    patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    collections: Optional[List[str]] = None
    cross_project: Optional[bool] = False
    search_mode: Optional[str] = "hybrid"
    include_dependencies: Optional[bool] = False
    include_context: Optional[bool] = True
    context_chunks: Optional[int] = 1

class SearchCodeRequest(BaseModel):
    query: str
    language: Optional[str] = None
    n_results: Optional[int] = 5
    cross_project: Optional[bool] = False
    search_mode: Optional[str] = "hybrid"
    include_dependencies: Optional[bool] = False
    include_context: Optional[bool] = True
    context_chunks: Optional[int] = 1

class SearchConfigRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    file_type: Optional[str] = None
    path: Optional[str] = None
    depth: Optional[int] = None

class IndexDocumentationRequest(BaseModel):
    file_path: str
    force_global: Optional[bool] = False

class SearchDocsRequest(BaseModel):
    query: str
    doc_type: Optional[str] = None
    n_results: Optional[int] = 5
    cross_project: Optional[bool] = False
    search_mode: Optional[str] = "hybrid"
    include_context: Optional[bool] = True
    context_chunks: Optional[int] = 1

class ReindexDirectoryRequest(BaseModel):
    directory: str
    patterns: Optional[List[str]] = None
    recursive: Optional[bool] = True
    force: Optional[bool] = False
    incremental: Optional[bool] = True

class DetectChangesRequest(BaseModel):
    directory: Optional[str] = "."

class GetFileChunksRequest(BaseModel):
    file_path: str
    start_chunk: Optional[int] = 0
    end_chunk: Optional[int] = None

class SwitchProjectRequest(BaseModel):
    project_path: str

# GitHub request models (v0.3.0)
class GitHubListRepositoriesRequest(BaseModel):
    owner: Optional[str] = None

class GitHubSwitchRepositoryRequest(BaseModel):
    owner: str
    repo: str

class GitHubFetchIssuesRequest(BaseModel):
    state: Optional[str] = "open"
    labels: Optional[List[str]] = None
    limit: Optional[int] = None

class GitHubGetIssueRequest(BaseModel):
    issue_number: int

class GitHubCreateIssueRequest(BaseModel):
    title: str
    body: Optional[str] = ""
    labels: Optional[List[str]] = None
    assignees: Optional[List[str]] = None

class GitHubAddCommentRequest(BaseModel):
    issue_number: int
    body: str

class GitHubAnalyzeIssueRequest(BaseModel):
    issue_number: int

class GitHubSuggestFixRequest(BaseModel):
    issue_number: int

class GitHubCreatePullRequestRequest(BaseModel):
    title: str
    body: str
    head: str
    base: Optional[str] = "main"
    files: Optional[List[Dict[str, str]]] = None

class GitHubResolveIssueRequest(BaseModel):
    issue_number: int
    dry_run: Optional[bool] = True

# Startup is now handled by lifespan context manager

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Qdrant RAG Server HTTP API", "status": "running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "server": "qdrant-rag-http-api"}

@app.post("/index_code")
async def index_code_endpoint(request: IndexCodeRequest):
    """Index a code file"""
    try:
        result = index_code(file_path=request.file_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_config")
async def index_config_endpoint(request: IndexConfigRequest):
    """Index a configuration file"""
    try:
        result = index_config(file_path=request.file_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_directory")
async def index_directory_endpoint(request: IndexDirectoryRequest):
    """Index an entire directory"""
    try:
        result = index_directory(
            directory=request.directory,
            patterns=request.patterns,
            recursive=True
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_endpoint(request: SearchRequest):
    """General search across all collections"""
    try:
        result = search(
            query=request.query,
            n_results=request.n_results,
            cross_project=request.cross_project,
            search_mode=request.search_mode,
            include_dependencies=request.include_dependencies,
            include_context=request.include_context,
            context_chunks=request.context_chunks
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_code")
async def search_code_endpoint(request: SearchCodeRequest):
    """Search in code collection with filtering"""
    try:
        result = search_code(
            query=request.query,
            language=request.language,
            n_results=request.n_results,
            cross_project=request.cross_project,
            search_mode=request.search_mode,
            include_dependencies=request.include_dependencies,
            include_context=request.include_context,
            context_chunks=request.context_chunks
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_config")
async def search_config_endpoint(request: SearchConfigRequest):
    """Search in config collection with filtering"""
    try:
        # Note: search_config is not a current MCP tool, using general search
        result = search(
            query=request.query,
            n_results=request.n_results,
            search_mode="hybrid"
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index_documentation")
async def index_documentation_endpoint(request: IndexDocumentationRequest):
    """Index a documentation file"""
    try:
        result = index_documentation(
            file_path=request.file_path,
            force_global=request.force_global
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_docs")
async def search_docs_endpoint(request: SearchDocsRequest):
    """Search in documentation collection"""
    try:
        result = search_docs(
            query=request.query,
            doc_type=request.doc_type,
            n_results=request.n_results,
            cross_project=request.cross_project,
            search_mode=request.search_mode,
            include_context=request.include_context,
            context_chunks=request.context_chunks
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex_directory")
async def reindex_directory_endpoint(request: ReindexDirectoryRequest):
    """Reindex a directory with smart incremental support"""
    try:
        result = reindex_directory(
            directory=request.directory,
            patterns=request.patterns,
            recursive=request.recursive,
            force=request.force,
            incremental=request.incremental
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_changes")
async def detect_changes_endpoint(request: DetectChangesRequest):
    """Detect changes in directory compared to indexed state"""
    try:
        result = detect_changes(directory=request.directory)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_file_chunks")
async def get_file_chunks_endpoint(request: GetFileChunksRequest):
    """Get chunks for a specific file"""
    try:
        result = get_file_chunks(
            file_path=request.file_path,
            start_chunk=request.start_chunk,
            end_chunk=request.end_chunk
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_context")
async def get_context_endpoint():
    """Get current project context information"""
    try:
        result = get_context()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_project")
async def switch_project_endpoint(request: SwitchProjectRequest):
    """Switch to a different project context"""
    try:
        result = switch_project(project_path=request.project_path)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health_check")
async def health_check_endpoint():
    """Detailed health check of all services"""
    try:
        result = health_check()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def get_collections():
    """Get information about Qdrant collections"""
    try:
        # Use the get_context function to get collection info
        context = get_context()
        if "error" in context:
            raise HTTPException(status_code=500, detail=context["error"])
        
        # Extract collections from context or use direct qdrant client
        from qdrant_mcp_context_aware import get_qdrant_client
        qdrant_client = get_qdrant_client()
        collections = qdrant_client.get_collections()
        return {"collections": [c.name for c in collections.collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# GitHub Integration Endpoints (v0.3.0)

@app.get("/github/repositories")
async def github_list_repositories_endpoint(owner: Optional[str] = None):
    """List GitHub repositories for a user/organization"""
    try:
        result = github_list_repositories(owner=owner)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/switch_repository")
async def github_switch_repository_endpoint(request: GitHubSwitchRepositoryRequest):
    """Switch to a different GitHub repository context"""
    try:
        result = github_switch_repository(owner=request.owner, repo=request.repo)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/github/issues")
async def github_fetch_issues_endpoint(
    state: Optional[str] = "open",
    labels: Optional[str] = None,
    limit: Optional[int] = None
):
    """Fetch GitHub issues from current repository"""
    try:
        # Parse labels from comma-separated string
        labels_list = labels.split(",") if labels else None
        
        result = github_fetch_issues(
            state=state,
            labels=labels_list,
            limit=limit
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/github/issues/{issue_number}")
async def github_get_issue_endpoint(issue_number: int):
    """Get detailed information about a specific GitHub issue"""
    try:
        result = github_get_issue(issue_number=issue_number)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/issues")
async def github_create_issue_endpoint(request: GitHubCreateIssueRequest):
    """Create a new GitHub issue"""
    try:
        result = github_create_issue(
            title=request.title,
            body=request.body,
            labels=request.labels,
            assignees=request.assignees
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/issues/{issue_number}/comment")
async def github_add_comment_endpoint(issue_number: int, request: GitHubAddCommentRequest):
    """Add a comment to an existing GitHub issue"""
    try:
        result = github_add_comment(
            issue_number=issue_number,
            body=request.body
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/issues/{issue_number}/analyze")
async def github_analyze_issue_endpoint(issue_number: int):
    """Perform comprehensive analysis of a GitHub issue using RAG search"""
    try:
        result = github_analyze_issue(issue_number=issue_number)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/issues/{issue_number}/suggest_fix")
async def github_suggest_fix_endpoint(issue_number: int):
    """Generate fix suggestions for a GitHub issue using RAG analysis"""
    try:
        result = github_suggest_fix(issue_number=issue_number)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/pull_requests")
async def github_create_pull_request_endpoint(request: GitHubCreatePullRequestRequest):
    """Create a GitHub pull request"""
    try:
        result = github_create_pull_request(
            title=request.title,
            body=request.body,
            head=request.head,
            base=request.base,
            files=request.files
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/issues/{issue_number}/resolve")
async def github_resolve_issue_endpoint(
    issue_number: int,
    dry_run: Optional[bool] = True
):
    """Attempt to resolve a GitHub issue with automated analysis and PR creation"""
    try:
        result = github_resolve_issue(
            issue_number=issue_number,
            dry_run=dry_run
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/github/health")
async def github_health_endpoint():
    """Check GitHub integration health and authentication status"""
    try:
        # Get overall health check which includes GitHub status
        result = health_check()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Extract GitHub-specific health information
        github_health = result.get("services", {}).get("github", {
            "status": "not_configured",
            "message": "GitHub integration not configured or available"
        })
        
        return {
            "github": github_health,
            "timestamp": result.get("timestamp")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)