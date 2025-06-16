"""
Core utilities and abstractions for the Qdrant RAG MCP Server.

This module provides shared functionality to reduce code duplication
and improve maintainability across the codebase.
"""

from .decorators import github_operation, get_github_instances

__all__ = ['github_operation', 'get_github_instances']