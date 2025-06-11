# src/github/__init__.py
"""
GitHub Integration Module for Qdrant MCP RAG Server

Provides GitHub API integration for issue analysis and resolution workflows.
"""

from .client import GitHubClient, get_github_client
from .issue_analyzer import IssueAnalyzer, get_issue_analyzer
from .code_generator import CodeGenerator, get_code_generator
from .workflows import GitHubWorkflows, get_github_workflows

# Optional Git operations support
try:
    from .git_operations import GitOperations, get_git_operations
    GIT_OPS_AVAILABLE = True
except ImportError:
    GIT_OPS_AVAILABLE = False
    GitOperations = None
    get_git_operations = None

# Optional GitHub Projects V2 support
try:
    from .projects_manager import GitHubProjectsManager, get_projects_manager
    PROJECTS_AVAILABLE = True
except ImportError:
    PROJECTS_AVAILABLE = False
    GitHubProjectsManager = None
    get_projects_manager = None

__all__ = [
    "GitHubClient",
    "get_github_client",
    "IssueAnalyzer", 
    "get_issue_analyzer",
    "CodeGenerator",
    "get_code_generator",
    "GitHubWorkflows",
    "get_github_workflows",
    "GitOperations",
    "get_git_operations",
    "GIT_OPS_AVAILABLE",
    "GitHubProjectsManager",
    "get_projects_manager", 
    "PROJECTS_AVAILABLE"
]