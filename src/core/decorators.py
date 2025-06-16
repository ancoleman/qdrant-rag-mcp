"""
Decorators for common patterns in the Qdrant RAG MCP Server.

This module provides decorators to eliminate code duplication,
particularly for GitHub operations that follow similar patterns.
"""

import functools
from typing import Callable, Any, Dict, Tuple, Optional
import logging

# Import from parent directory - we'll need to fix this import path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# We'll need to import these from the main module
# For now, we'll define the structure we expect

logger = logging.getLogger(__name__)


# Store instances in a context variable to avoid changing function signatures
_github_instances = None

def get_github_instances():
    """
    Get the current GitHub instances from the decorator context.
    
    This should be called within a function decorated with @github_operation.
    
    Returns:
        Tuple of (github_client, issue_analyzer, code_generator, workflows, projects_manager)
        
    Raises:
        RuntimeError: If called outside of a @github_operation decorated function
    """
    if _github_instances is None:
        raise RuntimeError("get_github_instances() called outside of @github_operation context")
    return _github_instances




def github_operation(
    operation_name: str, 
    require_repo: bool = False, 
    require_projects: bool = False
) -> Callable:
    """
    Decorator for GitHub operations that handles common patterns.
    
    This decorator standardizes:
    - Prerequisites validation
    - Error handling with parameter context
    - Logging with consistent format
    - Return value structure
    
    IMPORTANT: This decorator preserves the original function signature
    to maintain MCP compatibility. Use get_github_instances() within
    the decorated function to access the instances.
    
    Args:
        operation_name: Human-readable operation name for logging
        require_repo: Whether the operation requires a repository to be set
        require_projects: Whether the operation requires projects manager
        
    Returns:
        Decorated function that handles common GitHub operation patterns
        
    Example:
        @github_operation("switch repository")
        def github_switch_repository(owner: str, repo: str):
            github_client, _, _, _, _ = get_github_instances()
            repository = github_client.set_repository(owner, repo)
            return {"repository": repository}
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Dict[str, Any]:
            global _github_instances
            
            try:
                # Import here to avoid circular imports
                try:
                    from ..qdrant_mcp_context_aware import (
                        validate_github_prerequisites,
                        console_logger
                    )
                except ImportError:
                    # Fallback for testing or different import contexts
                    import sys
                    import os
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from qdrant_mcp_context_aware import (
                        validate_github_prerequisites,
                        console_logger
                    )
                
                # Validate prerequisites
                error, instances = validate_github_prerequisites(
                    require_repo=require_repo,
                    require_projects=require_projects
                )
                
                if error:
                    return error
                
                # Store instances in context variable
                _github_instances = instances
                
                try:
                    # Call the actual function with its original signature
                    result = func(*args, **kwargs)
                    
                    # Log success
                    console_logger.info(
                        f"Successfully completed {operation_name}",
                        extra={
                            "operation": func.__name__,
                            "status": "success"
                        }
                    )
                    
                    return result
                    
                finally:
                    # Clean up context
                    _github_instances = None
                
            except Exception as e:
                # Import here to avoid circular imports
                try:
                    from ..qdrant_mcp_context_aware import console_logger
                except ImportError:
                    # Fallback for testing or different import contexts
                    import sys
                    import os
                    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from qdrant_mcp_context_aware import console_logger
                
                # Extract context information from args and kwargs
                import inspect
                import re
                import json
                
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Parse error details for better error codes and messages
                error_str = str(e)
                error_code = "GITHUB_OPERATION_ERROR"
                error_details = error_str
                clean_error_msg = f"Failed to {operation_name}"
                
                # Try to extract HTTP status code
                status_match = re.search(r'\b(\d{3})\b', error_str)
                if status_match:
                    status_code = status_match.group(1)
                    error_code = f"GITHUB_ERROR_{status_code}"
                    
                    # Provide cleaner error messages based on status code
                    if status_code == "404":
                        clean_error_msg = f"Resource not found during {operation_name}"
                    elif status_code == "401":
                        clean_error_msg = f"Authentication failed during {operation_name}"
                    elif status_code == "403":
                        clean_error_msg = f"Access forbidden during {operation_name}"
                    elif status_code == "422":
                        clean_error_msg = f"Invalid request during {operation_name}"
                    elif status_code == "429":
                        clean_error_msg = f"Rate limit exceeded during {operation_name}"
                    else:
                        clean_error_msg = f"HTTP {status_code} error during {operation_name}"
                
                # Try to extract JSON error message if present
                try:
                    # First try to parse the entire error as JSON (for GraphQL responses)
                    if error_str.strip().startswith('{'):
                        error_json = json.loads(error_str)
                        # Handle GraphQL error format
                        if 'errors' in error_json and isinstance(error_json['errors'], list):
                            # GraphQL errors array
                            first_error = error_json['errors'][0]
                            error_details = first_error.get('message', error_str)
                            # Check for GraphQL error codes
                            if 'extensions' in first_error and 'code' in first_error['extensions']:
                                error_code = f"GRAPHQL_{first_error['extensions']['code']}"
                        elif 'message' in error_json:
                            error_details = error_json['message']
                    else:
                        # Try to extract JSON from within the string
                        json_match = re.search(r'\{[^}]+\}', error_str)
                        if json_match:
                            error_json = json.loads(json_match.group(0))
                            if 'message' in error_json:
                                error_details = error_json['message']
                except:
                    pass  # Keep original error details if JSON parsing fails
                
                # Extract specific error patterns
                if "rate limit" in error_str.lower():
                    error_code = "GITHUB_RATE_LIMIT"
                    clean_error_msg = f"GitHub API rate limit exceeded during {operation_name}"
                elif "bad credentials" in error_str.lower():
                    error_code = "GITHUB_AUTH_ERROR"
                    clean_error_msg = f"Invalid GitHub credentials during {operation_name}"
                
                console_logger.error(f"{clean_error_msg}: {error_details}", extra={
                    "operation": func.__name__,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_code": error_code,
                    "status": "error",
                    "context": bound_args.arguments
                })
                
                return {
                    "error": clean_error_msg,
                    "error_code": error_code,
                    "details": error_details,
                    "context": {param: str(value) for param, value in bound_args.arguments.items() if value is not None}
                }
                
        # Add metadata to the wrapper for introspection
        wrapper._github_operation = True
        wrapper._operation_name = operation_name
        wrapper._require_repo = require_repo
        wrapper._require_projects = require_projects
        
        return wrapper
    return decorator


def http_endpoint_handler(operation_name: str) -> Callable:
    """
    Decorator for HTTP endpoints that standardizes response handling.
    
    This decorator:
    - Converts synchronous operations to async
    - Standardizes error responses
    - Adds consistent logging
    
    Args:
        operation_name: Name of the operation for logging
        
    Returns:
        Decorated async function with standard error handling
        
    Example:
        @http_endpoint_handler("index documentation")
        async def index_documentation_endpoint(request: IndexDocumentationRequest):
            result = index_documentation(request.file_path)
            return result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                from fastapi.responses import JSONResponse
                from fastapi import HTTPException
                import asyncio
                from concurrent.futures import ThreadPoolExecutor
                
                # Get the event loop
                loop = asyncio.get_event_loop()
                
                # Execute the function (handling both sync and async)
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    # Run sync function in executor
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor,
                            lambda: func(*args, **kwargs)
                        )
                
                # If result is already a Response, return it
                if hasattr(result, 'status_code'):
                    return result
                    
                # Otherwise wrap in JSONResponse
                return JSONResponse(content=result)
                
            except HTTPException as e:
                # Re-raise HTTP exceptions as-is
                raise e
            except Exception as e:
                logger.error(f"Error in {operation_name}: {str(e)}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": str(e),
                        "type": type(e).__name__,
                        "operation": operation_name
                    }
                )
                
        return wrapper
    return decorator


def with_logging(operation_type: str = "operation") -> Callable:
    """
    Decorator that adds standardized logging to any function.
    
    Args:
        operation_type: Type of operation for logging context
        
    Returns:
        Decorated function with automatic logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            start_time = time.time()
            
            try:
                # Import here to avoid circular imports
                from ..qdrant_mcp_context_aware import console_logger
                
                # Log start
                console_logger.debug(
                    f"Starting {operation_type}: {func.__name__}",
                    extra={"operation": func.__name__, "type": operation_type}
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Log success
                duration_ms = (time.time() - start_time) * 1000
                console_logger.info(
                    f"Completed {operation_type}: {func.__name__}",
                    extra={
                        "operation": func.__name__,
                        "type": operation_type,
                        "duration_ms": duration_ms,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                # Log error
                duration_ms = (time.time() - start_time) * 1000
                console_logger.error(
                    f"Failed {operation_type}: {func.__name__}",
                    extra={
                        "operation": func.__name__,
                        "type": operation_type,
                        "duration_ms": duration_ms,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "status": "error"
                    }
                )
                raise
                
        return wrapper
    return decorator