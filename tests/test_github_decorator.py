#!/usr/bin/env python3
"""
Test script for the GitHub operation decorator.

This script tests that the decorator properly:
1. Handles prerequisites validation
2. Provides consistent error handling
3. Adds proper logging
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.decorators import github_operation
from typing import Dict, Any, Optional

# Mock objects for testing
class MockGitHubClient:
    def list_repositories(self, owner):
        return [
            {"name": "test-repo-1", "owner": owner or "test-user"},
            {"name": "test-repo-2", "owner": owner or "test-user"}
        ]

class MockLogger:
    def __init__(self):
        self.logs = []
    
    def info(self, msg, extra=None):
        self.logs.append({"level": "info", "msg": msg, "extra": extra})
    
    def error(self, msg, extra=None):
        self.logs.append({"level": "error", "msg": msg, "extra": extra})
    
    def debug(self, msg, extra=None):
        self.logs.append({"level": "debug", "msg": msg, "extra": extra})


# Test function using the decorator
@github_operation("list test repositories")
def test_list_repos(instances, owner: Optional[str] = None) -> Dict[str, Any]:
    """Test function that lists repositories."""
    github_client, _, _, _, _ = instances
    repositories = github_client.list_repositories(owner)
    
    return {
        "repositories": repositories,
        "count": len(repositories),
        "owner": owner or "test-user"
    }


def test_success_case():
    """Test successful operation."""
    print("Testing successful operation...")
    
    # Mock the prerequisites validation
    import sys
    
    # Create mock instances
    mock_client = MockGitHubClient()
    mock_logger = MockLogger()
    
    # We need to mock at the module level where it's imported
    import qdrant_mcp_context_aware
    
    # Store originals
    original_validate = qdrant_mcp_context_aware.validate_github_prerequisites
    original_logger = qdrant_mcp_context_aware.console_logger
    
    try:
        # Mock the validation function
        def mock_validate(require_repo=False, require_projects=False):
            return None, (mock_client, None, None, None, None)
        
        # Replace with mocks in the actual module
        qdrant_mcp_context_aware.validate_github_prerequisites = mock_validate
        qdrant_mcp_context_aware.console_logger = mock_logger
        
        # Call the decorated function
        result = test_list_repos(owner="test-owner")
        
        # Verify results
        assert result["count"] == 2, f"Expected 2 repositories, got {result['count']}"
        assert result["owner"] == "test-owner", f"Expected owner 'test-owner', got {result['owner']}"
        assert len(result["repositories"]) == 2, "Expected 2 repositories in list"
        
        # Check logging
        info_logs = [log for log in mock_logger.logs if log["level"] == "info"]
        assert len(info_logs) > 0, "Expected info log for successful operation"
        assert "Successfully completed" in info_logs[0]["msg"], "Expected success message in log"
        
        print("✅ Success case passed!")
        
    finally:
        # Restore originals
        qdrant_mcp_context_aware.validate_github_prerequisites = original_validate
        qdrant_mcp_context_aware.console_logger = original_logger


def test_error_case():
    """Test error handling."""
    print("\nTesting error handling...")
    
    import qdrant_mcp_context_aware
    
    mock_logger = MockLogger()
    
    # Store originals
    original_validate = qdrant_mcp_context_aware.validate_github_prerequisites
    original_logger = qdrant_mcp_context_aware.console_logger
    
    try:
        # Mock validation to return an error
        def mock_validate_error(require_repo=False, require_projects=False):
            return {"error": "GitHub not configured"}, None
        
        # Replace with mocks
        qdrant_mcp_context_aware.validate_github_prerequisites = mock_validate_error
        qdrant_mcp_context_aware.console_logger = mock_logger
        
        # Call should return the error
        result = test_list_repos()
        
        # Verify error response
        assert "error" in result, "Expected error in result"
        assert result["error"] == "GitHub not configured", f"Unexpected error: {result['error']}"
        
        print("✅ Error case passed!")
        
    finally:
        # Restore originals
        qdrant_mcp_context_aware.validate_github_prerequisites = original_validate
        qdrant_mcp_context_aware.console_logger = original_logger


def test_exception_handling():
    """Test exception handling."""
    print("\nTesting exception handling...")
    
    import qdrant_mcp_context_aware
    
    mock_logger = MockLogger()
    
    # Create a function that raises an exception
    @github_operation("failing operation")
    def failing_function(instances):
        raise ValueError("Test exception")
    
    # Store originals
    original_validate = qdrant_mcp_context_aware.validate_github_prerequisites
    original_logger = qdrant_mcp_context_aware.console_logger
    
    try:
        # Mock successful validation but function throws
        def mock_validate(require_repo=False, require_projects=False):
            return None, (None, None, None, None, None)
        
        qdrant_mcp_context_aware.validate_github_prerequisites = mock_validate
        qdrant_mcp_context_aware.console_logger = mock_logger
        
        # Call should catch exception and return error
        result = failing_function()
        
        # Verify error response
        assert "error" in result, "Expected error in result"
        assert "Failed to failing operation" in result["error"], f"Unexpected error message: {result['error']}"
        assert result.get("error_code") == "GITHUB_OPERATION_ERROR", "Expected error code"
        
        # Check error logging
        error_logs = [log for log in mock_logger.logs if log["level"] == "error"]
        assert len(error_logs) > 0, "Expected error log for exception"
        
        print("✅ Exception handling passed!")
        
    finally:
        qdrant_mcp_context_aware.validate_github_prerequisites = original_validate
        qdrant_mcp_context_aware.console_logger = original_logger


if __name__ == "__main__":
    print("Testing GitHub operation decorator...\n")
    
    try:
        test_success_case()
        test_error_case()
        test_exception_handling()
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)