"""
Git operations support for creating branches and committing changes.

This module provides Git functionality for the GitHub integration,
enabling automated branch creation and file modifications for PRs.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import tempfile
import shutil

try:
    import git
    from git import Repo, GitCommandError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    

logger = logging.getLogger(__name__)


class GitOperationsError(Exception):
    """Raised when Git operations fail."""
    pass


class GitOperations:
    """
    Handle Git operations for creating branches and committing changes.
    
    This class provides methods to:
    - Clone repositories
    - Create branches
    - Modify files
    - Commit changes
    - Push branches
    """
    
    def __init__(self, github_client):
        """
        Initialize GitOperations.
        
        Args:
            github_client: GitHubClient instance for repository access
        """
        if not GIT_AVAILABLE:
            raise ImportError("GitPython not available. Install with: pip install GitPython")
            
        self.github_client = github_client
        self._temp_repos = {}  # Track temporary repo clones
        
    def prepare_branch(self, repo_name: str, branch_name: str, base_branch: str = "main") -> str:
        """
        Prepare a new branch for changes.
        
        Args:
            repo_name: Full repository name (owner/repo)
            branch_name: Name for the new branch
            base_branch: Base branch to create from
            
        Returns:
            Path to the cloned repository
        """
        try:
            # Create temporary directory for repo
            temp_dir = tempfile.mkdtemp(prefix=f"github_pr_{branch_name}_")
            logger.info(f"Cloning repository {repo_name} to {temp_dir}")
            
            # Get repository URL with authentication
            repo_url = self._get_authenticated_url(repo_name)
            
            # Clone the repository
            repo = Repo.clone_from(repo_url, temp_dir, branch=base_branch)
            
            # Create and checkout new branch
            new_branch = repo.create_head(branch_name)
            new_branch.checkout()
            
            # Store reference for cleanup
            self._temp_repos[branch_name] = temp_dir
            
            logger.info(f"Created branch {branch_name} from {base_branch}")
            return temp_dir
            
        except GitCommandError as e:
            raise GitOperationsError(f"Failed to prepare branch: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error preparing branch: {e}")
            raise
            
    def apply_changes(self, repo_path: str, files: List[Dict[str, str]]) -> List[str]:
        """
        Apply file changes to the repository.
        
        Args:
            repo_path: Path to the repository
            files: List of file changes with 'path' and 'content'
            
        Returns:
            List of modified file paths
        """
        modified_files = []
        
        try:
            repo = Repo(repo_path)
            
            for file_info in files:
                file_path = file_info.get("path")
                content = file_info.get("content")
                
                if not file_path or content is None:
                    logger.warning(f"Skipping invalid file entry: {file_info}")
                    continue
                    
                # Create full path
                full_path = Path(repo_path) / file_path
                
                # Ensure parent directory exists
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                # Stage the file
                repo.index.add([file_path])
                modified_files.append(file_path)
                
                logger.info(f"Modified file: {file_path}")
                
            return modified_files
            
        except Exception as e:
            logger.error(f"Failed to apply changes: {e}")
            raise GitOperationsError(f"Failed to apply changes: {str(e)}")
            
    def commit_and_push(self, repo_path: str, branch_name: str, commit_message: str) -> Dict[str, Any]:
        """
        Commit changes and push the branch.
        
        Args:
            repo_path: Path to the repository
            branch_name: Branch name to push
            commit_message: Commit message
            
        Returns:
            Commit information
        """
        try:
            repo = Repo(repo_path)
            
            # Check if there are changes to commit
            if not repo.is_dirty() and not repo.untracked_files:
                logger.warning("No changes to commit")
                return {
                    "status": "no_changes",
                    "message": "No changes to commit"
                }
            
            # Commit changes
            commit = repo.index.commit(commit_message)
            logger.info(f"Created commit: {commit.hexsha[:8]}")
            
            # Push the branch
            origin = repo.remote("origin")
            push_info = origin.push(branch_name)
            
            if push_info:
                push_result = push_info[0]
                if push_result.flags & push_result.ERROR:
                    raise GitOperationsError(f"Push failed: {push_result.summary}")
                    
            logger.info(f"Pushed branch {branch_name}")
            
            return {
                "status": "success",
                "commit_sha": commit.hexsha,
                "branch": branch_name,
                "message": commit_message
            }
            
        except GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            raise GitOperationsError(f"Git operation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during commit/push: {e}")
            raise
            
    def cleanup(self, branch_name: str):
        """
        Clean up temporary repository clone.
        
        Args:
            branch_name: Branch name to clean up
        """
        if branch_name in self._temp_repos:
            temp_dir = self._temp_repos[branch_name]
            try:
                shutil.rmtree(temp_dir)
                del self._temp_repos[branch_name]
                logger.info(f"Cleaned up temporary repo for branch {branch_name}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_dir}: {e}")
                
    def cleanup_all(self):
        """Clean up all temporary repositories."""
        for branch_name in list(self._temp_repos.keys()):
            self.cleanup(branch_name)
            
    def _get_authenticated_url(self, repo_name: str) -> str:
        """
        Get authenticated repository URL.
        
        Args:
            repo_name: Full repository name (owner/repo)
            
        Returns:
            Authenticated repository URL
        """
        # Get authentication token
        auth_config = self.github_client.config.get("authentication", {})
        token = auth_config.get("token")
        
        if not token:
            raise GitOperationsError("No GitHub token available for authentication")
            
        # Build authenticated URL
        return f"https://{token}@github.com/{repo_name}.git"
        
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_all()


def get_git_operations(github_client) -> GitOperations:
    """
    Get GitOperations instance.
    
    Args:
        github_client: GitHubClient instance
        
    Returns:
        GitOperations instance
    """
    return GitOperations(github_client)