"""
GitHub API Client for Qdrant MCP RAG Server

Provides authenticated GitHub API access with error handling and rate limiting.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

try:
    from github import Github, Auth
    from github.GithubException import RateLimitExceededException
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

try:
    from ..config import get_config
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import get_config

logger = logging.getLogger(__name__)


class GitHubAuthError(Exception):
    """Raised when GitHub authentication fails."""
    pass


class GitHubRateLimitError(Exception):
    """Raised when GitHub rate limit is exceeded."""
    pass


class GitHubClient:
    """
    GitHub API client with authentication and error handling.
    
    Supports both personal access tokens and GitHub App authentication.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GitHub client.
        
        Args:
            config: Optional configuration override
        """
        if not GITHUB_AVAILABLE:
            raise ImportError("PyGithub not available. Install with: pip install PyGithub")
            
        self.config = config or get_config().get("github", {})
        self._github = None
        self._current_repo = None
        self._rate_limit_reset = None
        
        # Initialize authentication
        self._init_auth()
        
    def _init_auth(self):
        """Initialize GitHub authentication."""
        auth_config = self.config.get("authentication", {})
        
        # Try personal access token first
        token = auth_config.get("token")
        if token:
            try:
                auth = Auth.Token(token)
                self._github = Github(auth=auth, **self._get_api_config())
                # Test authentication
                user = self._github.get_user()
                logger.info(f"Authenticated as GitHub user: {user.login}")
                return
            except Exception as e:
                logger.error(f"Token authentication failed: {e}")
                raise GitHubAuthError(f"Token authentication failed: {e}")
        
        # Try GitHub App authentication
        app_id = auth_config.get("app_id")
        private_key_path = auth_config.get("private_key_path")
        installation_id = auth_config.get("installation_id")
        
        if app_id and private_key_path and installation_id:
            try:
                if Path(private_key_path).exists():
                    with open(private_key_path, 'r') as f:
                        private_key = f.read()
                    
                    auth = Auth.AppAuth(app_id, private_key)
                    app_github = Github(auth=auth)
                    installation = app_github.get_app().get_installation(installation_id)
                    auth = Auth.AppInstallationAuth(installation)
                    self._github = Github(auth=auth, **self._get_api_config())
                    logger.info(f"Authenticated as GitHub App installation: {installation_id}")
                    return
                else:
                    logger.error(f"Private key file not found: {private_key_path}")
            except Exception as e:
                logger.error(f"App authentication failed: {e}")
        
        # No valid authentication found
        raise GitHubAuthError("No valid GitHub authentication configured. Set GITHUB_TOKEN or configure GitHub App.")
    
    def _get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for GitHub client."""
        api_config = self.config.get("api", {})
        config = {}
        
        base_url = api_config.get("base_url")
        # Skip base_url if it contains unresolved environment variable syntax
        if base_url and not base_url.startswith("${"):
            config["base_url"] = base_url
        if api_config.get("timeout"):
            config["timeout"] = api_config["timeout"]
            
        return config
    
    def _handle_rate_limit(self):
        """Handle GitHub rate limiting."""
        if not self._github:
            return
            
        try:
            rate_limit = self._github.get_rate_limit()
            remaining = rate_limit.core.remaining
            reset_time = rate_limit.core.reset
            
            # Check if we're close to rate limit
            buffer = self.config.get("api", {}).get("rate_limit_buffer", 100)
            if remaining < buffer:
                sleep_time = (reset_time - datetime.now()).total_seconds() + 10
                if sleep_time > 0:
                    logger.warning(f"Rate limit nearly exceeded. Sleeping for {sleep_time} seconds.")
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.warning(f"Failed to check rate limit: {e}")
    
    def _retry_request(self, func, *args, **kwargs):
        """Retry a request with exponential backoff."""
        retry_config = self.config.get("api", {})
        max_attempts = retry_config.get("retry_attempts", 3)
        base_delay = retry_config.get("retry_delay", 1.0)
        
        for attempt in range(max_attempts):
            try:
                self._handle_rate_limit()
                return func(*args, **kwargs)
            except RateLimitExceededException as e:
                if attempt == max_attempts - 1:
                    raise GitHubRateLimitError(f"Rate limit exceeded after {max_attempts} attempts")
                
                # Extract reset time from exception
                reset_time = datetime.fromtimestamp(e.headers.get('X-RateLimit-Reset', time.time() + 3600))
                sleep_time = (reset_time - datetime.now()).total_seconds() + 10
                logger.warning(f"Rate limit exceeded. Sleeping for {sleep_time} seconds.")
                time.sleep(sleep_time)
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                    
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {delay} seconds.")
                time.sleep(delay)
    
    def set_repository(self, owner: str, repo: str):
        """
        Set the current repository context.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            Repository object
        """
        try:
            repository = self._retry_request(self._github.get_repo, f"{owner}/{repo}")
            self._current_repo = repository
            logger.info(f"Set current repository: {owner}/{repo}")
            return repository
        except Exception as e:
            raise GitHubAuthError(f"Failed to access repository {owner}/{repo}: {e}")
    
    def get_current_repository(self):
        """Get the current repository context."""
        return self._current_repo
    
    def list_repositories(self, owner: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List repositories for a user/organization.
        
        Args:
            owner: Repository owner (defaults to authenticated user)
            
        Returns:
            List of repository information
        """
        try:
            if owner:
                user = self._retry_request(self._github.get_user, owner)
                repos = self._retry_request(user.get_repos)
            else:
                repos = self._retry_request(self._github.get_user().get_repos)
            
            result = []
            for repo in repos:
                result.append({
                    "name": repo.name,
                    "full_name": repo.full_name,
                    "description": repo.description,
                    "private": repo.private,
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                    "clone_url": repo.clone_url,
                    "ssh_url": repo.ssh_url
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            raise
    
    def get_issues(self, state: str = "open", labels: Optional[List[str]] = None, 
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get issues from current repository.
        
        Args:
            state: Issue state (open, closed, all)
            labels: Filter by labels
            limit: Maximum number of issues
            
        Returns:
            List of issue information
        """
        if not self._current_repo:
            raise ValueError("No repository set. Call set_repository() first.")
        
        try:
            issues = self._retry_request(
                self._current_repo.get_issues,
                state=state,
                labels=labels or []
            )
            
            result = []
            count = 0
            max_count = limit or self.config.get("issues", {}).get("max_fetch_count", 50)
            
            for issue in issues:
                if count >= max_count:
                    break
                    
                # Skip pull requests (they appear as issues in GitHub API)
                if issue.pull_request:
                    continue
                
                result.append({
                    "number": issue.number,
                    "title": issue.title,
                    "body": issue.body,
                    "state": issue.state,
                    "labels": [label.name for label in issue.labels],
                    "assignees": [assignee.login for assignee in issue.assignees],
                    "author": issue.user.login if issue.user else None,
                    "created_at": issue.created_at.isoformat() if issue.created_at else None,
                    "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                    "comments_count": issue.comments,
                    "url": issue.html_url
                })
                count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get issues: {e}")
            raise
    
    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific issue.
        
        Args:
            issue_number: Issue number
            
        Returns:
            Detailed issue information
        """
        if not self._current_repo:
            raise ValueError("No repository set. Call set_repository() first.")
        
        try:
            issue = self._retry_request(self._current_repo.get_issue, issue_number)
            
            # Get comments
            comments = []
            for comment in issue.get_comments():
                comments.append({
                    "id": comment.id,
                    "body": comment.body,
                    "author": comment.user.login if comment.user else None,
                    "created_at": comment.created_at.isoformat() if comment.created_at else None,
                    "updated_at": comment.updated_at.isoformat() if comment.updated_at else None
                })
            
            return {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
                "author": issue.user.login if issue.user else None,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                "comments_count": issue.comments,
                "comments": comments,
                "url": issue.html_url,
                "milestone": issue.milestone.title if issue.milestone else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get issue {issue_number}: {e}")
            raise
    
    def create_issue(self, title: str, body: str = "", labels: Optional[List[str]] = None,
                    assignees: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create a new GitHub issue.
        
        Args:
            title: Issue title
            body: Issue description/body
            labels: List of label names to apply
            assignees: List of usernames to assign
            
        Returns:
            Created issue information
        """
        if not self._current_repo:
            raise ValueError("No repository set. Call set_repository() first.")
        
        try:
            # Create issue
            issue = self._retry_request(
                self._current_repo.create_issue,
                title=title,
                body=body or "",
                labels=labels or [],
                assignees=assignees or []
            )
            
            return {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body,
                "state": issue.state,
                "labels": [label.name for label in issue.labels],
                "assignees": [assignee.login for assignee in issue.assignees],
                "author": issue.user.login if issue.user else None,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                "comments_count": issue.comments,
                "url": issue.html_url
            }
            
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            raise
    
    def add_comment(self, issue_number: int, body: str) -> Dict[str, Any]:
        """
        Add a comment to an existing issue.
        
        Args:
            issue_number: Issue number to comment on
            body: Comment body text
            
        Returns:
            Comment information
        """
        if not self._current_repo:
            raise ValueError("No repository set. Call set_repository() first.")
        
        try:
            issue = self._retry_request(self._current_repo.get_issue, issue_number)
            comment = self._retry_request(issue.create_comment, body)
            
            return {
                "id": comment.id,
                "body": comment.body,
                "author": comment.user.login if comment.user else None,
                "created_at": comment.created_at.isoformat() if comment.created_at else None,
                "updated_at": comment.updated_at.isoformat() if comment.updated_at else None,
                "issue_number": issue_number,
                "url": comment.html_url
            }
            
        except Exception as e:
            logger.error(f"Failed to add comment to issue {issue_number}: {e}")
            raise
    
    def create_pull_request(self, title: str, body: str, head: str, base: str = "main",
                           files: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Create a pull request.
        
        Args:
            title: PR title
            body: PR description
            head: Head branch
            base: Base branch (default: main)
            files: List of files to include (for reference only - actual files must be committed)
            
        Returns:
            Pull request information
        """
        if not self._current_repo:
            raise ValueError("No repository set. Call set_repository() first.")
        
        try:
            # Create PR
            pr = self._retry_request(
                self._current_repo.create_pull,
                title=title,
                body=body,
                head=head,
                base=base,
                draft=self.config.get("pull_requests", {}).get("draft_by_default", True)
            )
            
            return {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                "state": pr.state,
                "head": pr.head.ref,
                "base": pr.base.ref,
                "author": pr.user.login if pr.user else None,
                "created_at": pr.created_at.isoformat() if pr.created_at else None,
                "url": pr.html_url,
                "draft": pr.draft
            }
            
        except Exception as e:
            logger.error(f"Failed to create pull request: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check GitHub API connectivity and rate limits.
        
        Returns:
            Health check information
        """
        try:
            if not self._github:
                return {
                    "status": "error",
                    "message": "GitHub client not initialized"
                }
            
            # Test API connectivity
            user = self._retry_request(self._github.get_user)
            rate_limit = self._retry_request(self._github.get_rate_limit)
            
            return {
                "status": "healthy",
                "authenticated_user": user.login,
                "rate_limit": {
                    "core": {
                        "remaining": rate_limit.core.remaining,
                        "limit": rate_limit.core.limit,
                        "reset": rate_limit.core.reset.isoformat()
                    },
                    "search": {
                        "remaining": rate_limit.search.remaining,
                        "limit": rate_limit.search.limit,
                        "reset": rate_limit.search.reset.isoformat()
                    }
                },
                "current_repository": self._current_repo.full_name if self._current_repo else None
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e)
            }


# Global client instance
_github_client = None


def get_github_client() -> GitHubClient:
    """Get or create global GitHub client instance."""
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    return _github_client