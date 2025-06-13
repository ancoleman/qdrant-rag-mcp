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

# Load environment variables and set HuggingFace cache BEFORE imports
from pathlib import Path
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ensure HuggingFace uses our custom cache directory
if 'SENTENCE_TRANSFORMERS_HOME' in os.environ:
    cache_dir = os.path.expanduser(os.environ['SENTENCE_TRANSFORMERS_HOME'])
    if not os.environ.get('HF_HOME'):
        os.environ['HF_HOME'] = cache_dir
    if not os.environ.get('HF_HUB_CACHE'):
        os.environ['HF_HUB_CACHE'] = cache_dir
from qdrant_mcp_context_aware import (
    index_code, index_config, index_documentation, index_directory,
    search, search_code, search_config, search_docs, reindex_directory,
    detect_changes, get_file_chunks, get_context, switch_project, health_check,
    # GitHub integration functions (v0.3.0)
    github_list_repositories, github_switch_repository, github_fetch_issues,
    github_get_issue, github_create_issue, github_add_comment, github_analyze_issue, github_suggest_fix,
    github_create_pull_request, github_resolve_issue,
    # GitHub Projects V2 functions (v0.3.4)
    github_create_project, github_get_project, github_add_project_item,
    github_update_project_item, github_create_project_field, github_get_project_status,
    github_smart_add_project_item,
    # GitHub Sub-Issues functions (v0.3.4.post4)
    github_list_sub_issues, github_add_sub_issue, github_remove_sub_issue,
    github_create_sub_issue, github_reorder_sub_issues, github_add_sub_issues_to_project,
    # Enhanced GitHub Issue Management functions (v0.3.4.post5)
    github_close_issue, github_assign_issue, github_update_issue, github_search_issues,
    github_list_milestones, github_create_milestone, github_update_milestone, github_close_milestone
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
    version="0.3.4.post6",
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
    # Progressive context parameters (v0.3.2)
    context_level: Optional[str] = "auto"
    progressive_mode: Optional[bool] = None
    include_expansion_options: Optional[bool] = True
    semantic_cache: Optional[bool] = True

class SearchCodeRequest(BaseModel):
    query: str
    language: Optional[str] = None
    n_results: Optional[int] = 5
    cross_project: Optional[bool] = False
    search_mode: Optional[str] = "hybrid"
    include_dependencies: Optional[bool] = False
    include_context: Optional[bool] = True
    context_chunks: Optional[int] = 1
    # Progressive context parameters (v0.3.2)
    context_level: Optional[str] = "auto"
    progressive_mode: Optional[bool] = None
    include_expansion_options: Optional[bool] = True
    semantic_cache: Optional[bool] = True

class SearchConfigRequest(BaseModel):
    query: str
    file_type: Optional[str] = None
    n_results: Optional[int] = 5
    cross_project: Optional[bool] = False
    search_mode: Optional[str] = "hybrid"
    include_context: Optional[bool] = True
    context_chunks: Optional[int] = 1
    # Progressive context parameters (v0.3.2)
    context_level: Optional[str] = "auto"
    progressive_mode: Optional[bool] = None
    include_expansion_options: Optional[bool] = True
    semantic_cache: Optional[bool] = True

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
    # Progressive context parameters (v0.3.2)
    context_level: Optional[str] = "auto"
    progressive_mode: Optional[bool] = None
    include_expansion_options: Optional[bool] = True
    semantic_cache: Optional[bool] = True

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
    milestone: Optional[str] = None
    assignee: Optional[str] = None
    since: Optional[str] = None
    sort: str = "created"
    direction: str = "desc"
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

# GitHub Projects V2 request models (v0.3.4)
class GitHubCreateProjectRequest(BaseModel):
    owner: str
    title: str
    body: Optional[str] = None
    template: Optional[str] = None

class GitHubGetProjectRequest(BaseModel):
    owner: str
    number: int

class GitHubAddProjectItemRequest(BaseModel):
    project_id: str
    issue_number: Optional[int] = None
    pr_number: Optional[int] = None

class GitHubUpdateProjectItemRequest(BaseModel):
    project_id: str
    item_id: str
    field_id: str
    value: Any

class GitHubCreateProjectFieldRequest(BaseModel):
    project_id: str
    name: str
    data_type: str
    options: Optional[List[Dict[str, str]]] = None

class GitHubGetProjectStatusRequest(BaseModel):
    owner: str
    number: int

# GitHub Sub-Issues request models (v0.3.4.post4)
class GitHubListSubIssuesRequest(BaseModel):
    parent_issue_number: int

class GitHubAddSubIssueRequest(BaseModel):
    parent_issue_number: int
    sub_issue_number: int
    replace_parent: Optional[bool] = False

class GitHubRemoveSubIssueRequest(BaseModel):
    parent_issue_number: int
    sub_issue_number: int

class GitHubCreateSubIssueRequest(BaseModel):
    parent_issue_number: int
    title: str
    body: Optional[str] = ""
    labels: Optional[List[str]] = None

class GitHubReorderSubIssuesRequest(BaseModel):
    parent_issue_number: int
    sub_issue_numbers: List[int]

class GitHubAddSubIssuesToProjectRequest(BaseModel):
    project_id: str
    parent_issue_number: int

# Enhanced GitHub Issue Management Request Models (v0.3.4.post5)

class GitHubCloseIssueRequest(BaseModel):
    issue_number: int
    reason: str = "completed"
    comment: Optional[str] = None

class GitHubAssignIssueRequest(BaseModel):
    issue_number: int
    assignees: List[str]
    operation: str = "add"

class GitHubUpdateIssueRequest(BaseModel):
    issue_number: int
    title: Optional[str] = None
    body: Optional[str] = None
    labels: Optional[List[str]] = None
    milestone: Optional[int] = None
    assignees: Optional[List[str]] = None
    state: Optional[str] = None

class GitHubSearchIssuesRequest(BaseModel):
    query: str
    sort: Optional[str] = None
    order: str = "desc"
    limit: Optional[int] = None

class GitHubListMilestonesRequest(BaseModel):
    state: str = "open"
    sort: str = "due_on"
    direction: str = "asc"

class GitHubCreateMilestoneRequest(BaseModel):
    title: str
    description: Optional[str] = None
    due_on: Optional[str] = None

class GitHubUpdateMilestoneRequest(BaseModel):
    number: int
    title: Optional[str] = None
    description: Optional[str] = None
    due_on: Optional[str] = None
    state: Optional[str] = None

class GitHubCloseMilestoneRequest(BaseModel):
    number: int

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
            context_chunks=request.context_chunks,
            # Progressive context parameters
            context_level=request.context_level,
            progressive_mode=request.progressive_mode,
            include_expansion_options=request.include_expansion_options,
            semantic_cache=request.semantic_cache
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
            context_chunks=request.context_chunks,
            # Progressive context parameters
            context_level=request.context_level,
            progressive_mode=request.progressive_mode,
            include_expansion_options=request.include_expansion_options,
            semantic_cache=request.semantic_cache
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
        result = search_config(
            query=request.query,
            file_type=request.file_type,
            n_results=request.n_results,
            cross_project=request.cross_project,
            search_mode=request.search_mode,
            include_context=request.include_context,
            context_chunks=request.context_chunks,
            context_level=request.context_level,
            progressive_mode=request.progressive_mode,
            include_expansion_options=request.include_expansion_options,
            semantic_cache=request.semantic_cache
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
            context_chunks=request.context_chunks,
            # Progressive context parameters
            context_level=request.context_level,
            progressive_mode=request.progressive_mode,
            include_expansion_options=request.include_expansion_options,
            semantic_cache=request.semantic_cache
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
    milestone: Optional[str] = None,
    assignee: Optional[str] = None,
    since: Optional[str] = None,
    sort: str = "created",
    direction: str = "desc",
    limit: Optional[int] = None
):
    """Fetch GitHub issues from current repository with enhanced filtering"""
    try:
        # Parse labels from comma-separated string
        labels_list = labels.split(",") if labels else None
        
        result = github_fetch_issues(
            state=state,
            labels=labels_list,
            milestone=milestone,
            assignee=assignee,
            since=since,
            sort=sort,
            direction=direction,
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

# GitHub Projects V2 Endpoints (v0.3.4)

@app.get("/github/projects")
async def github_list_projects_endpoint(owner: Optional[str] = None, limit: int = 20):
    """List GitHub Projects V2 for a user or organization"""
    try:
        from qdrant_mcp_context_aware import github_list_projects
        
        result = github_list_projects(owner=owner, limit=limit)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/projects")
async def github_create_project_endpoint(request: GitHubCreateProjectRequest):
    """Create a new GitHub Project V2"""
    try:
        # Instead of calling the MCP tool, let's call the projects_manager directly
        from qdrant_mcp_context_aware import get_github_instances, GITHUB_AVAILABLE, PROJECTS_AVAILABLE
        
        if not GITHUB_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub integration not available")
        
        if not PROJECTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub Projects integration not available")
        
        github_client, _, _, _, projects_manager = get_github_instances()
        
        if not projects_manager:
            raise HTTPException(status_code=503, detail="Projects manager not available")
        
        # If template is specified, use the template function
        if request.template:
            project = await projects_manager.create_project_from_template(
                owner=request.owner,
                title=request.title,
                template=request.template,
                body=request.body
            )
        else:
            # Regular project creation - call async method directly
            project = await projects_manager.create_project(
                owner=request.owner,
                title=request.title,
                body=request.body
            )
        
        response = {
            "success": True,
            "project": {
                "id": project["id"],
                "number": project["number"],
                "title": project["title"],
                "description": None,
                "url": project["url"],
                "owner": request.owner,
                "created_at": project["createdAt"]
            }
        }
        
        # Add a note if description was requested
        if request.body:
            response["note"] = "GitHub Projects V2 API doesn't support descriptions at creation time. Consider adding a custom field after creation."
            
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/github/projects/{owner}/{number}")
async def github_get_project_endpoint(owner: str, number: int):
    """Get GitHub Project V2 details"""
    try:
        from qdrant_mcp_context_aware import get_github_instances, GITHUB_AVAILABLE, PROJECTS_AVAILABLE
        
        if not GITHUB_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub integration not available")
        
        if not PROJECTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub Projects integration not available")
        
        github_client, _, _, _, projects_manager = get_github_instances()
        
        if not projects_manager:
            raise HTTPException(status_code=503, detail="Projects manager not available")
        
        # Call async method directly
        project = await projects_manager.get_project(owner, number)
        
        return {
            "success": True,
            "project": project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/projects/items")
async def github_add_project_item_endpoint(request: GitHubAddProjectItemRequest):
    """Add an issue or PR to a GitHub Project"""
    try:
        # The MCP tool takes either issue_number or pr_number, not both
        # If pr_number is provided, use it as issue_number (GitHub treats them the same)
        issue_num = request.issue_number if request.issue_number else request.pr_number
        
        if not issue_num:
            raise HTTPException(status_code=400, detail="Either issue_number or pr_number is required")
            
        result = github_add_project_item(
            project_id=request.project_id,
            issue_number=issue_num
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/projects/items/smart")
async def github_smart_add_project_item_endpoint(request: GitHubAddProjectItemRequest):
    """Add an issue to a project with intelligent field assignment"""
    try:
        # The MCP tool takes either issue_number or pr_number, not both
        issue_num = request.issue_number if request.issue_number else request.pr_number
        
        if not issue_num:
            raise HTTPException(status_code=400, detail="Either issue_number or pr_number is required")
            
        result = github_smart_add_project_item(
            project_id=request.project_id,
            issue_number=issue_num
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/github/projects/items")
async def github_update_project_item_endpoint(request: GitHubUpdateProjectItemRequest):
    """Update a field value for a project item"""
    try:
        result = github_update_project_item(
            project_id=request.project_id,
            item_id=request.item_id,
            field_id=request.field_id,
            value=request.value
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/github/projects/fields")
async def github_create_project_field_endpoint(request: GitHubCreateProjectFieldRequest):
    """Create a custom field in a GitHub Project"""
    try:
        result = github_create_project_field(
            project_id=request.project_id,
            name=request.name,
            data_type=request.data_type,
            options=request.options
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/github/projects/{project_id}")
async def github_delete_project_endpoint(project_id: str):
    """Delete a GitHub Project V2"""
    try:
        from qdrant_mcp_context_aware import github_delete_project
        
        result = github_delete_project(project_id=project_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/github/projects/{owner}/{number}/status")
async def github_get_project_status_endpoint(owner: str, number: int):
    """Get GitHub Project V2 status with metrics"""
    try:
        # First get the project to retrieve its ID
        from qdrant_mcp_context_aware import get_github_instances, GITHUB_AVAILABLE, PROJECTS_AVAILABLE
        
        if not GITHUB_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub integration not available")
        
        if not PROJECTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="GitHub Projects integration not available")
        
        github_client, _, _, _, projects_manager = get_github_instances()
        
        if not projects_manager:
            raise HTTPException(status_code=503, detail="Projects manager not available")
        
        # Get project details first to get the ID
        project = await projects_manager.get_project(owner, number)
        project_id = project["id"]
        
        # Now call the status function with the project ID
        result = github_get_project_status(project_id=project_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# GitHub Sub-Issues endpoints (v0.3.4.post4)
@app.post("/github/list_sub_issues")
async def github_list_sub_issues_endpoint(request: GitHubListSubIssuesRequest):
    """List all sub-issues for a parent issue"""
    try:
        result = github_list_sub_issues(parent_issue_number=request.parent_issue_number)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/add_sub_issue")
async def github_add_sub_issue_endpoint(request: GitHubAddSubIssueRequest):
    """Add a sub-issue relationship to a parent issue"""
    try:
        result = github_add_sub_issue(
            parent_issue_number=request.parent_issue_number,
            sub_issue_number=request.sub_issue_number,
            replace_parent=request.replace_parent
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/remove_sub_issue")
async def github_remove_sub_issue_endpoint(request: GitHubRemoveSubIssueRequest):
    """Remove a sub-issue relationship from a parent issue"""
    try:
        result = github_remove_sub_issue(
            parent_issue_number=request.parent_issue_number,
            sub_issue_number=request.sub_issue_number
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/create_sub_issue")
async def github_create_sub_issue_endpoint(request: GitHubCreateSubIssueRequest):
    """Create a new issue and immediately add it as a sub-issue"""
    try:
        result = github_create_sub_issue(
            parent_issue_number=request.parent_issue_number,
            title=request.title,
            body=request.body,
            labels=request.labels
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/reorder_sub_issues")
async def github_reorder_sub_issues_endpoint(request: GitHubReorderSubIssuesRequest):
    """Reorder sub-issues within a parent issue"""
    try:
        result = github_reorder_sub_issues(
            parent_issue_number=request.parent_issue_number,
            sub_issue_numbers=request.sub_issue_numbers
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/add_sub_issues_to_project")
async def github_add_sub_issues_to_project_endpoint(request: GitHubAddSubIssuesToProjectRequest):
    """Add all sub-issues of a parent issue to a GitHub Project V2"""
    try:
        result = github_add_sub_issues_to_project(
            project_id=request.project_id,
            parent_issue_number=request.parent_issue_number
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Enhanced GitHub Issue Management Endpoints (v0.3.4.post5)

@app.patch("/github/issues/{issue_number}/close")
async def github_close_issue_endpoint(issue_number: int, request: GitHubCloseIssueRequest):
    """Close a GitHub issue with state reason"""
    try:
        result = github_close_issue(
            issue_number=issue_number,
            reason=request.reason,
            comment=request.comment
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/issues/{issue_number}/assignees")
async def github_assign_issue_endpoint(issue_number: int, request: GitHubAssignIssueRequest):
    """Assign or unassign users to/from a GitHub issue"""
    try:
        result = github_assign_issue(
            issue_number=issue_number,
            assignees=request.assignees,
            operation=request.operation
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/github/issues/{issue_number}")
async def github_update_issue_endpoint(issue_number: int, request: GitHubUpdateIssueRequest):
    """Update issue properties"""
    try:
        result = github_update_issue(
            issue_number=issue_number,
            title=request.title,
            body=request.body,
            labels=request.labels,
            milestone=request.milestone,
            assignees=request.assignees,
            state=request.state
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/issues/search")
async def github_search_issues_endpoint(request: GitHubSearchIssuesRequest):
    """Search issues using GitHub's search API"""
    try:
        result = github_search_issues(
            query=request.query,
            sort=request.sort,
            order=request.order,
            limit=request.limit
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Milestone Management Endpoints

@app.get("/github/milestones")
async def github_list_milestones_endpoint(
    state: str = "open",
    sort: str = "due_on",
    direction: str = "asc"
):
    """List repository milestones"""
    try:
        result = github_list_milestones(
            state=state,
            sort=sort,
            direction=direction
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/github/milestones")
async def github_create_milestone_endpoint(request: GitHubCreateMilestoneRequest):
    """Create a new milestone"""
    try:
        result = github_create_milestone(
            title=request.title,
            description=request.description,
            due_on=request.due_on
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/github/milestones/{number}")
async def github_update_milestone_endpoint(number: int, request: GitHubUpdateMilestoneRequest):
    """Update milestone properties"""
    try:
        result = github_update_milestone(
            number=number,
            title=request.title,
            description=request.description,
            due_on=request.due_on,
            state=request.state
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/github/milestones/{number}")
async def github_close_milestone_endpoint(number: int):
    """Close a milestone"""
    try:
        result = github_close_milestone(number=number)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Apple Silicon optimization endpoints
@app.get("/apple_silicon_status")
async def apple_silicon_status_endpoint():
    """Get Apple Silicon optimization status and memory information"""
    try:
        # Import the functions from the MCP server
        from qdrant_mcp_context_aware import get_apple_silicon_status
        result = get_apple_silicon_status()
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class CleanupRequest(BaseModel):
    level: str = "standard"


@app.post("/apple_silicon_cleanup")
async def apple_silicon_cleanup_endpoint(request: CleanupRequest):
    """Manually trigger Apple Silicon memory cleanup"""
    try:
        # Import the function from the MCP server
        from qdrant_mcp_context_aware import trigger_apple_silicon_cleanup
        result = trigger_apple_silicon_cleanup(level=request.level)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)