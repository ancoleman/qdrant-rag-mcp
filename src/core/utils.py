"""
Core utility abstractions for reducing code duplication.

This module provides utility classes and functions that centralize common
patterns used throughout the codebase.
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from contextlib import contextmanager
from functools import lru_cache

from ..utils.logging_config import get_logger
from ..utils.memory_manager import get_memory_manager, UnifiedMemoryManager
from ..utils.specialized_embeddings import SpecializedEmbeddingManager


# Project markers for determining project boundaries
PROJECT_MARKERS = ['.git', 'pyproject.toml', 'package.json', '.project-root']


class OperationLogger:
    """
    Consistent logging for all operations with timing and context.
    
    Usage:
        logger = OperationLogger("search")
        with logger.log_operation("search_code", query="factorial"):
            # Operation code here
            pass
    """
    
    def __init__(self, operation_type: str):
        """
        Initialize operation logger.
        
        Args:
            operation_type: Type of operation (e.g., "search", "index", "github")
        """
        self.logger = get_logger()
        self.operation_type = operation_type
    
    @contextmanager
    def log_operation(self, name: str, **context):
        """
        Context manager for operation logging with timing.
        
        Args:
            name: Name of the specific operation
            **context: Additional context to log
            
        Yields:
            Dict to store operation results
        """
        operation_id = f"{self.operation_type}_{name}"
        start_time = time.time()
        result = {"status": "success"}
        
        # Log start
        self.logger.info(
            f"{operation_id} started",
            extra={
                "operation": operation_id,
                "type": self.operation_type,
                "context": context
            }
        )
        
        try:
            yield result
            
            # Log success
            duration = time.time() - start_time
            self.logger.info(
                f"{operation_id} completed",
                extra={
                    "operation": operation_id,
                    "type": self.operation_type,
                    "duration": duration,
                    "status": "success",
                    "result": result.get("summary", {})
                }
            )
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            result["status"] = "error"
            result["error"] = str(e)
            
            self.logger.error(
                f"{operation_id} failed",
                extra={
                    "operation": operation_id,
                    "type": self.operation_type,
                    "duration": duration,
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise


class MemoryManagementMixin:
    """
    Mixin for classes that need memory management.
    
    Usage:
        class MyClass(MemoryManagementMixin):
            def my_method(self):
                if self.check_memory_pressure():
                    self.cleanup_memory()
    """
    
    _memory_manager: Optional[UnifiedMemoryManager] = None
    
    @property
    def memory_manager(self) -> UnifiedMemoryManager:
        """Get or create memory manager instance."""
        if not self._memory_manager:
            self._memory_manager = get_memory_manager()
        return self._memory_manager
    
    def check_memory_pressure(self, threshold: float = 0.8) -> bool:
        """
        Check if memory usage is above threshold.
        
        Args:
            threshold: Memory usage threshold (0.0 to 1.0)
            
        Returns:
            True if memory pressure is high
        """
        try:
            usage = self.memory_manager.get_memory_usage()
            return usage.get("percentage", 0) / 100 > threshold
        except Exception:
            # If we can't check memory, assume it's OK
            return False
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """
        Trigger memory cleanup.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Cleanup results
        """
        return self.memory_manager.cleanup(aggressive=aggressive)
    
    @contextmanager
    def memory_managed_operation(self, operation_name: str):
        """
        Context manager for operations that need memory management.
        
        Args:
            operation_name: Name of the operation for logging
        """
        # Check memory before
        if self.check_memory_pressure(threshold=0.7):
            self.cleanup_memory()
        
        try:
            yield
        finally:
            # Check memory after
            if self.check_memory_pressure(threshold=0.8):
                self.cleanup_memory(aggressive=True)


class ModelLoadingStrategy:
    """
    Centralized model loading with fallbacks and caching.
    
    Usage:
        loader = ModelLoadingStrategy()
        model = loader.load_model("nomic-ai/CodeRankEmbed", "code")
    """
    
    def __init__(self):
        """Initialize model loading strategy."""
        self.logger = get_logger()
        self._embeddings_manager = None
        self._fallback_models = {
            "code": ["microsoft/codebert-base", "sentence-transformers/all-MiniLM-L6-v2"],
            "config": ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
            "documentation": ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/all-MiniLM-L6-v2"],
            "general": ["sentence-transformers/all-MiniLM-L6-v2"]
        }
    
    @property
    def embeddings_manager(self) -> SpecializedEmbeddingManager:
        """Get or create embeddings manager."""
        if not self._embeddings_manager:
            self._embeddings_manager = SpecializedEmbeddingManager()
        return self._embeddings_manager
    
    def load_model(
        self,
        model_name: str,
        content_type: str = "general",
        use_fallback: bool = True
    ) -> Optional[Any]:
        """
        Load model with automatic fallback.
        
        Args:
            model_name: Name of the model to load
            content_type: Type of content (code, config, documentation, general)
            use_fallback: Whether to try fallback models on failure
            
        Returns:
            Loaded model or None if all attempts fail
        """
        models_to_try = [model_name]
        
        if use_fallback and content_type in self._fallback_models:
            models_to_try.extend(self._fallback_models[content_type])
        
        for model in models_to_try:
            try:
                self.logger.info(
                    f"Loading model: {model}",
                    extra={
                        "model": model,
                        "content_type": content_type,
                        "operation": "model_load"
                    }
                )
                
                # Try to load through embeddings manager
                # This is a simplified version - actual implementation would
                # depend on the embeddings manager interface
                return self.embeddings_manager.get_model(content_type)
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to load model {model}: {str(e)}",
                    extra={
                        "model": model,
                        "content_type": content_type,
                        "error": str(e),
                        "operation": "model_load_error"
                    }
                )
                
                if model == models_to_try[-1]:
                    # This was the last model, raise the error
                    raise
        
        return None
    
    def get_model_info(self, content_type: str) -> Dict[str, Any]:
        """
        Get information about the model for a content type.
        
        Args:
            content_type: Type of content
            
        Returns:
            Model information dict
        """
        try:
            config = self.embeddings_manager.get_model_config(content_type)
            return {
                "model_name": config.get("model_name"),
                "dimension": config.get("dimension"),
                "requires_query_prefix": config.get("requires_query_prefix", False),
                "query_prefix": config.get("query_prefix")
            }
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            return {}


class CollectionNameGenerator:
    """
    Centralized collection name generation with project awareness.
    
    This replaces the scattered collection naming logic throughout the codebase.
    """
    
    def __init__(self):
        """Initialize collection name generator."""
        self.logger = get_logger()
        self._current_project = None
    
    @property
    def current_project(self) -> Optional[Dict[str, Any]]:
        """Get current project context."""
        if not self._current_project:
            from ..qdrant_mcp_context_aware import get_current_project
            self._current_project = get_current_project()
        return self._current_project
    
    def get_collection_name(
        self,
        file_path: Optional[str] = None,
        file_type: str = "code",
        force_global: bool = False
    ) -> str:
        """
        Get collection name for a file, respecting project boundaries.
        
        Args:
            file_path: Path to the file (optional)
            file_type: Type of file (code, config, documentation)
            force_global: Force use of global collection
            
        Returns:
            Collection name
        """
        # Check force_global flag
        if force_global:
            return f"global_{file_type}"
        
        # If no file path, use current project
        if not file_path:
            if self.current_project:
                return f"{self.current_project['collection_prefix']}_{file_type}"
            else:
                return f"global_{file_type}"
        
        # Check if file is in current project
        path = Path(file_path).resolve()
        
        if self.current_project:
            project_root = Path(self.current_project["root"])
            try:
                # Check if file is within project boundaries
                path.relative_to(project_root)
                return f"{self.current_project['collection_prefix']}_{file_type}"
            except ValueError:
                # File is outside current project
                pass
        
        # Find project for this file
        for parent in path.parents:
            for marker in PROJECT_MARKERS:
                if (parent / marker).exists():
                    project_name = self._sanitize_project_name(parent.name)
                    return f"project_{project_name}_{file_type}"
        
        # No project found - use global collection
        return f"global_{file_type}"
    
    def _sanitize_project_name(self, name: str) -> str:
        """
        Sanitize project name for use in collection names.
        
        Args:
            name: Raw project name
            
        Returns:
            Sanitized name
        """
        # Replace spaces and hyphens with underscores
        sanitized = name.replace(" ", "_").replace("-", "_")
        
        # Remove any non-alphanumeric characters except underscores
        sanitized = "".join(c for c in sanitized if c.isalnum() or c == "_")
        
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"project_{sanitized}"
        
        # Truncate if too long
        if len(sanitized) > 50:
            # Use hash for uniqueness
            hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
            sanitized = f"{sanitized[:40]}_{hash_suffix}"
        
        return sanitized or "unnamed"
    
    def list_project_collections(
        self,
        project_name: Optional[str] = None
    ) -> List[str]:
        """
        List all collections for a project.
        
        Args:
            project_name: Project name (uses current if not specified)
            
        Returns:
            List of collection names
        """
        if not project_name and self.current_project:
            prefix = self.current_project['collection_prefix']
        elif project_name:
            prefix = f"project_{self._sanitize_project_name(project_name)}"
        else:
            return []
        
        # Standard collection types
        collection_types = ['code', 'config', 'documentation']
        
        return [f"{prefix}_{ctype}" for ctype in collection_types]


# Singleton instances
_collection_name_generator = None
_operation_logger_cache = {}


def get_collection_name(
    file_path: Optional[str] = None,
    file_type: str = "code",
    force_global: bool = False
) -> str:
    """
    Get collection name for a file (singleton helper).
    
    Args:
        file_path: Path to the file (optional)
        file_type: Type of file (code, config, documentation)
        force_global: Force use of global collection
        
    Returns:
        Collection name
    """
    global _collection_name_generator
    if not _collection_name_generator:
        _collection_name_generator = CollectionNameGenerator()
    
    return _collection_name_generator.get_collection_name(
        file_path, file_type, force_global
    )


def get_operation_logger(operation_type: str) -> OperationLogger:
    """
    Get or create operation logger (cached).
    
    Args:
        operation_type: Type of operation
        
    Returns:
        Operation logger instance
    """
    global _operation_logger_cache
    if operation_type not in _operation_logger_cache:
        _operation_logger_cache[operation_type] = OperationLogger(operation_type)
    
    return _operation_logger_cache[operation_type]


# Export key utilities
__all__ = [
    'OperationLogger',
    'MemoryManagementMixin',
    'ModelLoadingStrategy',
    'CollectionNameGenerator',
    'get_collection_name',
    'get_operation_logger',
    'PROJECT_MARKERS'
]