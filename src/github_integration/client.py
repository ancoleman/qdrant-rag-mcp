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
        """Handle GitHub rate limiting with intelligent backoff."""
        if not self._github:
            return
            
        try:
            rate_limit = self._github.get_rate_limit()
            core_remaining = rate_limit.core.remaining
            core_limit = rate_limit.core.limit
            reset_time = rate_limit.core.reset
            
            # Also check search rate limit for issue analysis
            search_remaining = rate_limit.search.remaining
            search_limit = rate_limit.search.limit
            
            # Log current rate limit status
            logger.debug(f"GitHub rate limits - Core: {core_remaining}/{core_limit}, Search: {search_remaining}/{search_limit}")
            
            # Check if we're close to rate limit
            buffer = self.config.get("api", {}).get("rate_limit_buffer", 100)
            
            # Check core API limit
            if core_remaining < buffer:
                sleep_time = (reset_time - datetime.now(reset_time.tzinfo)).total_seconds() + 10
                if sleep_time > 0:
                    logger.warning(
                        f"GitHub core API rate limit nearly exceeded ({core_remaining}/{core_limit} remaining). "
                        f"Waiting {sleep_time:.0f} seconds until reset at {reset_time.strftime('%H:%M:%S %Z')}."
                    )
                    time.sleep(sleep_time)
            
            # Check search API limit (used for issue analysis)
            if search_remaining < 5:  # More aggressive for search API
                search_reset = rate_limit.search.reset
                sleep_time = (search_reset - datetime.now(search_reset.tzinfo)).total_seconds() + 10
                if sleep_time > 0:
                    logger.warning(
                        f"GitHub search API rate limit nearly exceeded ({search_remaining}/{search_limit} remaining). "
                        f"Waiting {sleep_time:.0f} seconds."
                    )
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
            raise ValueError(
                "No GitHub repository context set. Please use 'github_switch_repository' "
                "to set the repository context first. Example: github_switch_repository(owner='myorg', repo='myproject')"
            )
        
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
    
    def create_pull_request_with_changes(self, title: str, body: str, branch_name: str,
                                       files: List[Dict[str, str]], base: str = "main",
                                       commit_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a pull request with file changes using GitOperations.
        
        This method handles the complete workflow:
        1. Creates a new branch
        2. Applies file changes
        3. Commits the changes
        4. Pushes the branch
        5. Creates a pull request
        
        Args:
            title: PR title
            body: PR description
            branch_name: Name for the new branch
            files: List of file changes with 'path' and 'content'
            base: Base branch (default: main)
            commit_message: Custom commit message (defaults to PR title)
            
        Returns:
            Pull request information with git operation details
        """
        if not self._current_repo:
            raise ValueError(
                "No GitHub repository context set. Please use 'github_switch_repository' "
                "to set the repository context first."
            )
            
        # Import GitOperations here to avoid circular imports
        try:
            from .git_operations import get_git_operations, GIT_AVAILABLE
        except ImportError:
            GIT_AVAILABLE = False
            
        if not GIT_AVAILABLE:
            return {
                "error": "Git operations not available",
                "message": "GitPython is required for file modifications. Install with: pip install GitPython",
                "fallback": "Create branch and commit changes manually, then use create_pull_request()"
            }
            
        git_ops = None
        try:
            # Get repository full name
            repo_name = self._current_repo.full_name
            
            # Initialize GitOperations
            git_ops = get_git_operations(self)
            
            # Prepare branch
            logger.info(f"Creating branch {branch_name} from {base}")
            repo_path = git_ops.prepare_branch(repo_name, branch_name, base)
            
            # Apply changes
            logger.info(f"Applying {len(files)} file changes")
            modified_files = git_ops.apply_changes(repo_path, files)
            
            if not modified_files:
                return {
                    "error": "No files were modified",
                    "message": "Check file paths and content in the files parameter"
                }
            
            # Commit and push
            commit_msg = commit_message or f"{title}\n\n{body}"
            logger.info(f"Committing and pushing changes")
            commit_result = git_ops.commit_and_push(repo_path, branch_name, commit_msg)
            
            if commit_result.get("status") == "no_changes":
                return {
                    "error": "No changes to commit",
                    "message": "Files were not modified or changes were already committed"
                }
            
            # Create pull request
            logger.info(f"Creating pull request from {branch_name} to {base}")
            pr = self.create_pull_request(
                title=title,
                body=body,
                head=branch_name,
                base=base
            )
            
            # Add git operation details
            pr["git_operations"] = {
                "branch_created": branch_name,
                "files_modified": modified_files,
                "commit_sha": commit_result.get("commit_sha"),
                "status": "success"
            }
            
            return pr
            
        except Exception as e:
            logger.error(f"Failed to create pull request with changes: {e}")
            error_msg = str(e)
            
            # Provide helpful error messages
            if "404" in error_msg:
                raise ValueError(
                    f"Repository not found or branch '{base}' doesn't exist. "
                    f"Repository: {self._current_repo.full_name}"
                )
            elif "already exists" in error_msg:
                raise ValueError(
                    f"Branch '{branch_name}' already exists. Please use a different branch name."
                )
            elif "permission" in error_msg.lower():
                raise GitHubAuthError(
                    "Insufficient permissions. Ensure your token has 'repo' scope for private repos "
                    "or 'public_repo' scope for public repos."
                )
            else:
                raise
                
        finally:
            # Cleanup temporary repository
            if git_ops:
                git_ops.cleanup(branch_name)
    
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
    
    # Sub-issues REST API support
    def _get_rest_headers(self) -> Dict[str, str]:
        """Get headers for REST API requests including auth and preview headers."""
        # Get the token from PyGithub's auth
        token = None
        if hasattr(self._github, '_Github__requester') and hasattr(self._github._Github__requester, '_Requester__authorizationHeader'):
            auth_header = self._github._Github__requester._Requester__authorizationHeader
            if auth_header and auth_header.startswith('token '):
                token = auth_header[6:]  # Remove 'token ' prefix
            elif auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        if not token:
            # Try to get from config
            token = self.config.get("auth", {}).get("token") or os.getenv("GITHUB_TOKEN")
        
        if not token:
            raise GitHubAuthError("Cannot find GitHub token for REST API calls")
        
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def _make_rest_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make a REST API request to GitHub."""
        import requests
        
        base_url = "https://api.github.com"
        if hasattr(self._github, '_Github__requester') and hasattr(self._github._Github__requester, '_Requester__hostname'):
            base_url = f"https://{self._github._Github__requester._Requester__hostname}"
        
        url = f"{base_url}{endpoint}"
        headers = self._get_rest_headers()
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers)
            elif method == "PATCH":
                response = requests.patch(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Return empty dict for 204 No Content
            if response.status_code == 204:
                return {"success": True}
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_msg = f"GitHub API error: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if 'message' in error_data:
                        error_msg += f" - {error_data['message']}"
                except:
                    error_msg += f" - {e.response.text}"
                raise Exception(error_msg)
            raise Exception(f"Request failed: {str(e)}")
    
    def add_sub_issue(self, parent_issue_number: int, sub_issue_id: int, replace_parent: bool = False) -> Dict[str, Any]:
        """
        Add a sub-issue relationship to a parent issue.
        
        Args:
            parent_issue_number: The parent issue number
            sub_issue_id: The sub-issue number (for same repo) or ID (for cross-repo)
            replace_parent: Whether to replace the current parent (re-parenting)
            
        Returns:
            Dict containing the operation result
        """
        if not self._current_repo:
            raise ValueError("No repository set. Use set_repository() first.")
        
        # If sub_issue_id looks like an issue number (small number), get the actual ID
        if sub_issue_id < 1000000:  # Likely an issue number, not an ID
            try:
                sub_issue = self._current_repo.get_issue(sub_issue_id)
                actual_sub_issue_id = sub_issue.id
            except Exception:
                # If we can't get the issue, assume it's already an ID
                actual_sub_issue_id = sub_issue_id
        else:
            actual_sub_issue_id = sub_issue_id
        
        endpoint = f"/repos/{self._current_repo.owner.login}/{self._current_repo.name}/issues/{parent_issue_number}/sub_issues"
        data = {
            "sub_issue_id": actual_sub_issue_id,
            "replace_parent": replace_parent
        }
        
        return self._retry_request(self._make_rest_request, "POST", endpoint, data)
    
    def list_sub_issues(self, parent_issue_number: int) -> List[Dict[str, Any]]:
        """
        List all sub-issues for a parent issue.
        
        Args:
            parent_issue_number: The parent issue number
            
        Returns:
            List of sub-issue data
        """
        if not self._current_repo:
            raise ValueError("No repository set. Use set_repository() first.")
        
        endpoint = f"/repos/{self._current_repo.owner.login}/{self._current_repo.name}/issues/{parent_issue_number}/sub_issues"
        
        result = self._retry_request(self._make_rest_request, "GET", endpoint)
        return result if isinstance(result, list) else []
    
    def remove_sub_issue(self, parent_issue_number: int, sub_issue_id: int) -> Dict[str, Any]:
        """
        Remove a sub-issue relationship from a parent issue.
        
        Args:
            parent_issue_number: The parent issue number
            sub_issue_id: The sub-issue number (for same repo) or ID (for cross-repo)
            
        Returns:
            Dict containing the operation result
        """
        if not self._current_repo:
            raise ValueError("No repository set. Use set_repository() first.")
        
        # If sub_issue_id looks like an issue number (small number), get the actual ID
        if sub_issue_id < 1000000:  # Likely an issue number, not an ID
            try:
                sub_issue = self._current_repo.get_issue(sub_issue_id)
                actual_sub_issue_id = sub_issue.id
            except Exception:
                # If we can't get the issue, assume it's already an ID
                actual_sub_issue_id = sub_issue_id
        else:
            actual_sub_issue_id = sub_issue_id
        
        # The DELETE endpoint uses query parameter for sub_issue_id
        endpoint = f"/repos/{self._current_repo.owner.login}/{self._current_repo.name}/issues/{parent_issue_number}/sub_issue?sub_issue_id={actual_sub_issue_id}"
        
        return self._retry_request(self._make_rest_request, "DELETE", endpoint)
    
    def reorder_sub_issues(self, parent_issue_number: int, sub_issue_ids: List[int]) -> Dict[str, Any]:
        """
        Reorder sub-issues within a parent issue.
        
        Args:
            parent_issue_number: The parent issue number
            sub_issue_ids: Ordered list of sub-issue numbers (for same repo) or IDs
            
        Returns:
            Dict containing the operation result
        """
        if not self._current_repo:
            raise ValueError("No repository set. Use set_repository() first.")
        
        # Convert issue numbers to IDs if needed
        actual_sub_issue_ids = []
        for sub_id in sub_issue_ids:
            if sub_id < 1000000:  # Likely an issue number, not an ID
                try:
                    sub_issue = self._current_repo.get_issue(sub_id)
                    actual_sub_issue_ids.append(sub_issue.id)
                except Exception:
                    # If we can't get the issue, assume it's already an ID
                    actual_sub_issue_ids.append(sub_id)
            else:
                actual_sub_issue_ids.append(sub_id)
        
        endpoint = f"/repos/{self._current_repo.owner.login}/{self._current_repo.name}/issues/{parent_issue_number}/sub_issues/priority"
        data = {
            "sub_issue_ids": actual_sub_issue_ids
        }
        
        return self._retry_request(self._make_rest_request, "PATCH", endpoint, data)


# Global client instance
_github_client = None


def get_github_client() -> GitHubClient:
    """Get or create global GitHub client instance."""
    global _github_client
    if _github_client is None:
        _github_client = GitHubClient()
    return _github_client