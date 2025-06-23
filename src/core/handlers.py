"""
Base HTTP endpoint handler with consistent error handling, logging, and async support.

This module provides a base class for all HTTP endpoints to ensure consistency
across the API, reduce code duplication, and improve maintainability.
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Union, TypeVar, cast
from functools import wraps
from contextlib import asynccontextmanager

from fastapi import HTTPException
from pydantic import BaseModel

from ..utils.logging_config import get_logger

# Type variable for generic return types
T = TypeVar('T')


class AsyncEndpointHandler:
    """
    Base handler for all HTTP endpoints with consistent error handling.
    
    Features:
    - Automatic error handling and HTTP status codes
    - Request/response logging with timing
    - Async/sync function handling
    - Parameter sanitization
    - Memory pressure checking (optional)
    """
    
    def __init__(self, operation_name: str, log_params: bool = True):
        """
        Initialize the endpoint handler.
        
        Args:
            operation_name: Name of the operation for logging
            log_params: Whether to log request parameters (disable for sensitive data)
        """
        self.operation_name = operation_name
        self.log_params = log_params
        self.logger = get_logger()
    
    async def handle(
        self,
        request_handler: Callable[..., Union[T, Dict[str, Any]]],
        request: Optional[BaseModel] = None,
        **kwargs
    ) -> T:
        """
        Main handler with error handling, logging, and metrics.
        
        Args:
            request_handler: The function to execute
            request: Optional Pydantic request model
            **kwargs: Additional arguments to pass to the handler
            
        Returns:
            The result from the request handler
            
        Raises:
            HTTPException: For client errors (400) or server errors (500)
        """
        start_time = time.time()
        
        # Prepare parameters
        if request:
            # Convert Pydantic model to dict and merge with kwargs
            params = {**request.dict(exclude_unset=True), **kwargs}
        else:
            params = kwargs
        
        try:
            # Log request
            log_extra = {
                "operation": f"http_{self.operation_name}",
                "endpoint_type": "http",
            }
            
            if self.log_params:
                log_extra["params"] = self._sanitize_params(params)
            
            self.logger.info(
                f"HTTP {self.operation_name} started",
                extra=log_extra
            )
            
            # Execute handler
            result = await self._execute(request_handler, **params)
            
            # Check for errors in result
            if isinstance(result, dict) and "error" in result:
                # This is an application-level error
                error_detail = result.get("error", "Unknown error")
                self.logger.warning(
                    f"HTTP {self.operation_name} returned error",
                    extra={
                        "operation": f"http_{self.operation_name}",
                        "error": error_detail,
                        "duration": time.time() - start_time,
                        "status": "client_error"
                    }
                )
                raise HTTPException(status_code=400, detail=error_detail)
            
            # Log success
            duration = time.time() - start_time
            self.logger.info(
                f"HTTP {self.operation_name} completed",
                extra={
                    "operation": f"http_{self.operation_name}",
                    "duration": duration,
                    "status": "success"
                }
            )
            
            return cast(T, result)
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
            
        except Exception as e:
            # Log unexpected errors
            duration = time.time() - start_time
            self.logger.error(
                f"HTTP {self.operation_name} failed with unexpected error",
                extra={
                    "operation": f"http_{self.operation_name}",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "duration": duration,
                    "status": "server_error"
                },
                exc_info=True
            )
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _execute(
        self,
        handler: Callable[..., Union[T, Dict[str, Any]]],
        **kwargs
    ) -> Union[T, Dict[str, Any]]:
        """
        Execute handler, handling sync/async appropriately.
        
        Args:
            handler: The function to execute
            **kwargs: Arguments to pass to the handler
            
        Returns:
            The result from the handler
        """
        if asyncio.iscoroutinefunction(handler):
            # Handler is async, await it directly
            return await handler(**kwargs)
        else:
            # Handler is sync, run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, handler, **kwargs)
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for logging (remove sensitive data).
        
        Args:
            params: Parameters to sanitize
            
        Returns:
            Sanitized parameters safe for logging
        """
        # Define sensitive parameter names
        sensitive_keys = {
            'password', 'token', 'secret', 'api_key', 'auth',
            'authorization', 'credentials'
        }
        
        sanitized = {}
        for key, value in params.items():
            # Check if key contains sensitive words
            is_sensitive = any(
                sensitive in key.lower()
                for sensitive in sensitive_keys
            )
            
            if is_sensitive:
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                sanitized[key] = self._sanitize_params(value)
            elif isinstance(value, list) and len(value) > 10:
                # Truncate long lists
                sanitized[key] = f"[{len(value)} items]"
            elif isinstance(value, str) and len(value) > 200:
                # Truncate long strings
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        
        return sanitized


def endpoint_handler(
    operation_name: str,
    log_params: bool = True
) -> Callable[[Callable], Callable]:
    """
    Decorator for creating endpoint handlers with consistent behavior.
    
    Usage:
        @app.post("/index_code")
        @endpoint_handler("index_code")
        async def index_code_endpoint(request: IndexCodeRequest):
            return index_code(file_path=request.file_path)
    
    Args:
        operation_name: Name of the operation for logging
        log_params: Whether to log request parameters
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        handler = AsyncEndpointHandler(operation_name, log_params)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args if present
            request = None
            func_kwargs = kwargs
            
            # Check if first arg is a Pydantic model
            if args and hasattr(args[0], 'dict'):
                request = args[0]
                func_args = args[1:]
            else:
                func_args = args
            
            # Create a lambda that calls the original function
            if func_args:
                # If there are positional args, include them
                request_handler = lambda **kw: func(*func_args, **kw)
            else:
                request_handler = func
            
            return await handler.handle(
                request_handler,
                request=request,
                **func_kwargs
            )
        
        return wrapper
    return decorator


class BatchEndpointHandler(AsyncEndpointHandler):
    """
    Extended handler for batch operations with progress tracking.
    
    Useful for operations that process multiple items and need
    to report progress.
    """
    
    def __init__(self, operation_name: str, log_params: bool = True):
        super().__init__(operation_name, log_params)
        self.progress_callback = None
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """Set a callback for progress updates (current, total)."""
        self.progress_callback = callback
    
    async def handle_batch(
        self,
        items: list,
        item_handler: Callable,
        batch_size: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Handle a batch of items with progress tracking.
        
        Args:
            items: List of items to process
            item_handler: Function to process each item
            batch_size: Number of items to process concurrently
            **kwargs: Additional arguments for the handler
            
        Returns:
            Dict with results and any errors
        """
        results = []
        errors = []
        total = len(items)
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self._process_item(item, item_handler, **kwargs)
                for item in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Separate results and errors
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    errors.append({
                        "item": batch[idx],
                        "error": str(result)
                    })
                else:
                    results.append(result)
            
            # Report progress
            processed = min(i + batch_size, total)
            if self.progress_callback:
                self.progress_callback(processed, total)
            
            self.logger.info(
                f"Batch progress: {processed}/{total}",
                extra={
                    "operation": f"batch_{self.operation_name}",
                    "progress": processed,
                    "total": total,
                    "errors_so_far": len(errors)
                }
            )
        
        return {
            "total": total,
            "processed": len(results),
            "errors": errors,
            "results": results
        }
    
    async def _process_item(self, item: Any, handler: Callable, **kwargs):
        """Process a single item with error handling."""
        try:
            return await self._execute(handler, item=item, **kwargs)
        except Exception as e:
            self.logger.warning(
                f"Failed to process item in batch",
                extra={
                    "operation": f"batch_{self.operation_name}",
                    "error": str(e),
                    "item": str(item)[:100]  # Truncate for logging
                }
            )
            raise


# Utility context managers for common patterns

@asynccontextmanager
async def handle_github_auth():
    """
    Context manager for handling GitHub authentication errors.
    
    Usage:
        async with handle_github_auth():
            result = github_operation()
    """
    try:
        yield
    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str:
            raise HTTPException(
                status_code=401,
                detail="GitHub authentication failed. Please check your GITHUB_TOKEN."
            )
        elif "403" in error_str or "forbidden" in error_str:
            raise HTTPException(
                status_code=403,
                detail="GitHub API rate limit exceeded or insufficient permissions."
            )
        elif "404" in error_str or "not found" in error_str:
            raise HTTPException(
                status_code=404,
                detail="GitHub resource not found. Please check repository and issue numbers."
            )
        else:
            # Re-raise other exceptions
            raise


@asynccontextmanager
async def handle_qdrant_errors():
    """
    Context manager for handling Qdrant-specific errors.
    
    Usage:
        async with handle_qdrant_errors():
            result = qdrant_operation()
    """
    try:
        yield
    except Exception as e:
        error_str = str(e).lower()
        if "connection" in error_str or "connect" in error_str:
            raise HTTPException(
                status_code=503,
                detail="Qdrant service unavailable. Please check if Qdrant is running."
            )
        elif "not found" in error_str or "does not exist" in error_str:
            raise HTTPException(
                status_code=404,
                detail="Collection or document not found in Qdrant."
            )
        else:
            # Re-raise other exceptions
            raise