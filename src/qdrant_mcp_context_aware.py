#!/usr/bin/env python3
"""
Qdrant RAG MCP Server - Context-Aware Version
Automatically scopes to current project unless explicitly overridden
Supports optional file watching for auto-reindexing
"""
import os
import sys
import hashlib
import threading
import time
import argparse
import fnmatch
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import version
try:
    from . import __version__
except ImportError:
    __version__ = "0.3.2"  # Fallback version

# Load environment variables from the MCP server directory
from dotenv import load_dotenv
mcp_server_dir = Path(__file__).parent.parent
env_path = mcp_server_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# MCP imports
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("qdrant-rag-context")

# Import project-aware logging
from utils.logging import get_project_logger, get_error_logger, log_operation
# Import hybrid search
from utils.hybrid_search import get_hybrid_searcher
# Import enhanced ranker
from utils.enhanced_ranker import get_enhanced_ranker
# Import file hash utilities
from utils.file_hash import calculate_file_hash, get_file_info
# Import configuration
from config import get_config
# Import progressive context
from utils.progressive_context import (
    get_progressive_manager,
    get_query_classifier,
    ProgressiveResult
)
# Import context tracking
from utils.context_tracking import SessionContextTracker, SessionStore, check_context_usage, get_context_indicator

# Configure basic console logging for startup messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_logger = logging.getLogger(__name__)

# Try to import watchdog for file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    console_logger.info("Watchdog not installed. File watching disabled. Install with: pip install watchdog")

# Custom error types for better error handling
class QdrantConnectionError(Exception):
    """Raised when connection to Qdrant fails."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details

class IndexingError(Exception):
    """Raised when file indexing fails."""
    def __init__(self, message: str, file_path: str, details: Optional[str] = None):
        super().__init__(message)
        self.file_path = file_path
        self.details = details

class SearchError(Exception):
    """Raised when search operation fails."""
    def __init__(self, message: str, query: str, details: Optional[str] = None):
        super().__init__(message)
        self.query = query
        self.details = details

# Global variables for lazy initialization
_qdrant_client = None
_embedding_model = None
_code_indexer = None
_config_indexer = None
_documentation_indexer = None
_current_project = None  # Cache current project detection
_context_tracker = None  # Context tracking instance
_session_store = None    # Session persistence

# Configuration
PROJECT_MARKERS = ['.git', 'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'pom.xml', '.project']
EXCLUDE_PATTERNS = [
    "**/.git/**", "**/__pycache__/**", "**/node_modules/**", 
    "**/.venv/**", "**/venv/**", "**/env/**", "**/.env/**",
    "**/build/**", "**/dist/**", "**/.cache/**", "**/tmp/**"
]

# File watcher implementation
if WATCHDOG_AVAILABLE:
    class RagFileWatcher(FileSystemEventHandler):
        """Watches for file changes and triggers reindexing"""
        
        def __init__(self, debounce_seconds=3.0):
            self.debounce_seconds = debounce_seconds
            self.pending_files: Set[str] = set()
            self.lock = threading.Lock()
            self.timer = None
            self.file_patterns = {
                '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
                '.json', '.yaml', '.yml', '.xml', '.toml', '.ini',
                '.md', '.sh', '.dockerfile', '.gitignore', '.dockerignore'
            }
            self.exclude_dirs = {
                '.git', '__pycache__', 'node_modules', '.venv', 'venv',
                'data', 'logs', 'dist', 'build', '.pytest_cache', '.cache'
            }
            
        def should_index_file(self, path: str) -> bool:
            """Check if file should be indexed"""
            path_obj = Path(path)
            
            # Check if in excluded directory
            for parent in path_obj.parents:
                if parent.name in self.exclude_dirs:
                    return False
                    
            # Check file extension
            return path_obj.suffix.lower() in self.file_patterns
            
        def on_modified(self, event):
            if not event.is_directory and self.should_index_file(event.src_path):
                self._add_pending_file(event.src_path)
                
        def on_created(self, event):
            if not event.is_directory and self.should_index_file(event.src_path):
                self._add_pending_file(event.src_path)
                
        def _add_pending_file(self, file_path: str):
            """Add file to pending list and schedule reindex"""
            with self.lock:
                self.pending_files.add(file_path)
                
            # Cancel existing timer
            if self.timer:
                self.timer.cancel()
                
            # Schedule new reindex
            self.timer = threading.Timer(self.debounce_seconds, self._reindex_pending)
            self.timer.start()
            
        def _reindex_pending(self):
            """Reindex pending files"""
            with self.lock:
                files = list(self.pending_files)
                self.pending_files.clear()
                
            if not files:
                return
                
            get_logger().info(f"🔄 Auto-reindexing {len(files)} changed files...")
            
            success = 0
            for file_path in files:
                try:
                    # Determine file type and index
                    ext = Path(file_path).suffix.lower()
                    if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                        result = index_config(str(file_path))
                    else:
                        result = index_code(str(file_path))
                        
                    if "error" not in result:
                        success += 1
                    else:
                        get_logger().warning(f"Failed to index {file_path}: {result['error']}")
                except Exception as e:
                    get_logger().error(f"Error indexing {file_path}: {e}")
                    
            get_logger().info(f"✅ Auto-indexed {success}/{len(files)} files")

def get_mcp_client_context() -> Optional[str]:
    """Get the client's working directory from MCP context if available."""
    # Check if MCP provides client context
    # This is a placeholder - actual implementation depends on MCP protocol support
    # For now, return None to use fallback methods
    return None

def get_current_project(client_directory: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Detect current project based on working directory.
    
    Args:
        client_directory: Optional explicit directory to use instead of auto-detection
    """
    global _current_project
    
    # Determine which directory to use
    if client_directory:
        # Explicit directory provided (e.g., from index_directory)
        cwd = Path(client_directory).resolve()
    else:
        # Try to get client's actual working directory
        # Option 1: Check for MCP client context
        client_cwd = get_mcp_client_context()
        
        # Option 2: Check environment variable
        if not client_cwd:
            client_cwd = os.environ.get('MCP_CLIENT_CWD')
        
        # Option 3: Fall back to server's cwd (with warning)
        if client_cwd:
            cwd = Path(client_cwd).resolve()
            console_logger.debug(f"Using client working directory: {cwd}")
        else:
            cwd = Path.cwd()
            console_logger.warning(f"Using MCP server's working directory ({cwd}) - may not match client's actual location. "
                         "Set MCP_CLIENT_CWD environment variable or use absolute paths.")
    
    # Check if we've already detected this project
    if _current_project and Path(_current_project["root"]) == cwd:
        return _current_project
    
    # Look for project markers
    for parent in [cwd] + list(cwd.parents):
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                _current_project = {
                    "name": project_name,
                    "root": str(parent),
                    "collection_prefix": f"project_{project_name}"
                }
                console_logger.info(f"Detected project: {project_name} at {parent}")
                return _current_project
    
    # No project found - use directory name as fallback
    _current_project = {
        "name": cwd.name,
        "root": str(cwd),
        "collection_prefix": f"dir_{cwd.name.replace(' ', '_').replace('-', '_')}"
    }
    console_logger.info(f"No project markers found, using directory: {cwd.name}")
    return _current_project

def get_logger():
    """Get a logger instance with current project context."""
    project = get_current_project()
    if project:
        # Add project path to the context
        project_context = {
            "name": project["name"],
            "path": project["root"]
        }
        return get_project_logger(project_context)
    return get_project_logger()

def get_context_tracker():
    """Get or initialize the context tracking system."""
    global _context_tracker, _session_store
    
    if _context_tracker is None:
        # Initialize context tracker
        _context_tracker = SessionContextTracker()
        
        # Initialize session store
        config = get_config()
        context_config = config.get("context_tracking", {})
        base_dir = Path(context_config.get("session_dir", "~/.mcp-servers/qdrant-rag")).expanduser()
        _session_store = SessionStore(base_dir)
        
        # Set current project in tracker
        project = get_current_project()
        if project:
            _context_tracker.set_current_project(project)
        
        console_logger.info(f"Context tracking initialized with session ID: {_context_tracker.session_id}")
    
    return _context_tracker

def get_collection_name(file_path: str, file_type: str = "code") -> str:
    """Get collection name for a file, respecting project boundaries"""
    path = Path(file_path).resolve()
    
    # Check if file is in current project
    current_project = get_current_project()
    if current_project:
        project_root = Path(current_project["root"])
        try:
            # Check if file is within project boundaries
            path.relative_to(project_root)
            return f"{current_project['collection_prefix']}_{file_type}"
        except ValueError:
            # File is outside current project
            pass
    
    # File is outside current project - find its project
    for parent in path.parents:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                return f"project_{project_name}_{file_type}"
    
    # No project found - use global collection
    return f"global_{file_type}"

def retry_operation(func, max_attempts=3, delay=1.0):
    """Simple retry logic for operations."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                continue
            raise
    if last_error:
        raise last_error

def _truncate_content(content: str, max_length: int = 1500) -> str:
    """Truncate content to prevent token limit issues"""
    if len(content) <= max_length:
        return content
    # Truncate and add indicator
    return content[:max_length] + "\n... (truncated)"

def _expand_search_context(results: List[Dict[str, Any]], qdrant_client, search_collections: List[str], context_chunks: int = 1) -> List[Dict[str, Any]]:
    """Expand search results with surrounding chunks for better context"""
    logger = get_logger()
    expanded_results = []
    seen_chunks = set()  # Track which chunks we've already added
    
    # Limit context chunks to reasonable amount
    context_chunks = min(context_chunks, 3)
    
    for result in results:
        # Create a group of related chunks
        chunk_group = {
            "primary_chunk": result,
            "context_before": [],
            "context_after": [],
            "expanded_content": ""
        }
        
        file_path = result.get("file_path", "")
        chunk_index = result.get("chunk_index", 0)
        collection = result.get("collection", "")
        
        if not file_path or collection not in search_collections:
            # Can't expand, just include the original
            expanded_results.append(result)
            continue
        
        # Track this chunk
        chunk_key = f"{file_path}:{chunk_index}"
        seen_chunks.add(chunk_key)
        
        # Fetch surrounding chunks
        try:
            # Get chunks before
            for i in range(1, context_chunks + 1):
                target_index = chunk_index - i
                if target_index >= 0:
                    before_filter = {
                        "must": [
                            {"key": "file_path", "match": {"value": file_path}},
                            {"key": "chunk_index", "match": {"value": target_index}}
                        ]
                    }
                    
                    before_results = qdrant_client.search(
                        collection_name=collection,
                        query_vector=[0.0] * 384,  # Dummy vector for filter-only search
                        query_filter=before_filter,
                        limit=1
                    )
                    
                    if before_results:
                        before_chunk = before_results[0].payload
                        chunk_group["context_before"].insert(0, before_chunk)
                        seen_chunks.add(f"{file_path}:{target_index}")
            
            # Get chunks after
            for i in range(1, context_chunks + 1):
                target_index = chunk_index + i
                after_filter = {
                    "must": [
                        {"key": "file_path", "match": {"value": file_path}},
                        {"key": "chunk_index", "match": {"value": target_index}}
                    ]
                }
                
                after_results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=[0.0] * 384,  # Dummy vector for filter-only search
                    query_filter=after_filter,
                    limit=1
                )
                
                if after_results:
                    after_chunk = after_results[0].payload
                    chunk_group["context_after"].append(after_chunk)
                    seen_chunks.add(f"{file_path}:{target_index}")
            
            # Build expanded content with clear markers
            expanded_parts = []
            
            # Add before context
            if chunk_group["context_before"]:
                expanded_parts.append("=== Context Before ===")
                for ctx in chunk_group["context_before"]:
                    lines = f"[Lines {ctx.get('line_start', '?')}-{ctx.get('line_end', '?')}]"
                    expanded_parts.append(f"{lines}\n{ctx.get('content', '')}")
                expanded_parts.append("")
            
            # Add main chunk with highlighting
            expanded_parts.append("=== Matched Section ===")
            main_lines = f"[Lines {result.get('line_start', '?')}-{result.get('line_end', '?')}]"
            expanded_parts.append(f"{main_lines}\n{result.get('content', '')}")
            expanded_parts.append("")
            
            # Add after context
            if chunk_group["context_after"]:
                expanded_parts.append("=== Context After ===")
                for ctx in chunk_group["context_after"]:
                    lines = f"[Lines {ctx.get('line_start', '?')}-{ctx.get('line_end', '?')}]"
                    expanded_parts.append(f"{lines}\n{ctx.get('content', '')}")
            
            # Update result with expanded content
            expanded_result = result.copy()
            expanded_result["expanded_content"] = "\n".join(expanded_parts)
            expanded_result["has_context"] = True
            expanded_result["context_chunks_before"] = len(chunk_group["context_before"])
            expanded_result["context_chunks_after"] = len(chunk_group["context_after"])
            
            # Calculate total line range
            all_chunks = chunk_group["context_before"] + [result] + chunk_group["context_after"]
            if all_chunks:
                expanded_result["total_line_range"] = {
                    "start": min(c.get("line_start", float('inf')) for c in all_chunks if c.get("line_start")),
                    "end": max(c.get("line_end", 0) for c in all_chunks if c.get("line_end"))
                }
            
            expanded_results.append(expanded_result)
            
        except Exception as e:
            logger.debug(f"Failed to expand context for chunk: {e}")
            # On error, just include the original result
            expanded_results.append(result)
    
    return expanded_results

def ensure_collection(collection_name: str):
    """Ensure a collection exists with retry logic"""
    client = get_qdrant_client()
    
    from qdrant_client.http.models import Distance, VectorParams
    
    def check_and_create():
        existing = [c.name for c in client.get_collections().collections]
        
        if collection_name not in existing:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
    
    # Use retry logic for collection operations
    retry_operation(check_and_create)

def get_qdrant_client():
    """Get or create Qdrant client with connection validation"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        try:
            _qdrant_client = QdrantClient(host=host, port=port)
            # Test connection
            retry_operation(lambda: _qdrant_client.get_collections())
        except Exception as e:
            raise QdrantConnectionError(
                f"Failed to connect to Qdrant at {host}:{port}",
                details=str(e)
            )
    
    return _qdrant_client

def get_embedding_model():
    """Get or create embedding model"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        cache_dir = os.path.expanduser(
            os.getenv("SENTENCE_TRANSFORMERS_HOME", str(mcp_server_dir / "data" / "models"))
        )
        
        _embedding_model = SentenceTransformer(model_name, cache_folder=cache_dir)
    
    return _embedding_model

def get_code_indexer():
    """Get or create code indexer"""
    global _code_indexer
    if _code_indexer is None:
        from indexers import CodeIndexer
        _code_indexer = CodeIndexer()
    return _code_indexer

def get_config_indexer():
    """Get or create config indexer"""
    global _config_indexer
    if _config_indexer is None:
        from indexers import ConfigIndexer
        _config_indexer = ConfigIndexer()
    return _config_indexer

def get_documentation_indexer():
    """Get or create documentation indexer"""
    global _documentation_indexer
    if _documentation_indexer is None:
        from indexers import DocumentationIndexer
        config = get_config()
        
        # Get documentation-specific chunk settings
        chunk_size = config.get("indexing.documentation_chunk_size", 2000)
        chunk_overlap = config.get("indexing.documentation_chunk_overlap", 400)
        
        _documentation_indexer = DocumentationIndexer(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    return _documentation_indexer

def clear_project_collections() -> Dict[str, Any]:
    """
    Clear all collections for the current project.
    Returns info about what was cleared.
    """
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found", "cleared": []}
    
    client = get_qdrant_client()
    cleared = []
    errors = []
    
    # Get all collections
    existing_collections = [c.name for c in client.get_collections().collections]
    
    # Find project collections
    for collection_type in ['code', 'config', 'documentation']:
        collection_name = f"{current_project['collection_prefix']}_{collection_type}"
        
        if collection_name in existing_collections:
            try:
                # Delete the collection
                client.delete_collection(collection_name)
                cleared.append(collection_name)
                
                # Clear BM25 index
                hybrid_searcher = get_hybrid_searcher()
                hybrid_searcher.bm25_manager.clear_index(collection_name)
                
                get_logger().info(f"Cleared collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to clear {collection_name}: {str(e)}"
                errors.append(error_msg)
                get_logger().error(error_msg)
    
    return {
        "project": current_project['name'],
        "cleared_collections": cleared,
        "errors": errors if errors else None
    }

@mcp.tool()
def get_context() -> Dict[str, Any]:
    """Get current project context"""
    current_project = get_current_project()
    if current_project:
        # Get statistics for current project
        client = get_qdrant_client()
        collections = [c.name for c in client.get_collections().collections]
        project_collections = [c for c in collections if c.startswith(current_project["collection_prefix"])]
        
        total_indexed = 0
        for collection in project_collections:
            try:
                info = client.get_collection(collection)
                total_indexed += info.points_count
            except:
                pass
        
        return {
            "current_project": current_project,
            "collections": project_collections,
            "total_indexed": total_indexed,
            "working_directory": str(Path.cwd())
        }
    else:
        return {
            "current_project": None,
            "working_directory": str(Path.cwd()),
            "message": "No project detected in current directory"
        }

@mcp.tool()
def get_file_chunks(file_path: str, start_chunk: int = 0, end_chunk: Optional[int] = None) -> Dict[str, Any]:
    """
    Get all chunks for a specific file or a range of chunks
    
    Args:
        file_path: Path to the file
        start_chunk: Starting chunk index (default: 0)
        end_chunk: Ending chunk index (inclusive, default: all chunks)
    
    Returns:
        File chunks with full content
    """
    try:
        # Resolve to absolute path
        abs_path = Path(file_path).resolve()
        
        # Get services
        qdrant_client = get_qdrant_client()
        
        # Determine file type and collection based on file extension
        suffix = abs_path.suffix.lower()
        
        # Determine file type
        if suffix in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.env']:
            file_type = "config"
        elif suffix in ['.md', '.markdown', '.rst', '.txt', '.mdx']:
            file_type = "documentation"
        else:
            file_type = "code"
        
        # Get collection name based on file type
        collection_name = get_collection_name(str(abs_path), file_type)
        
        # Build filter for the file
        filter_conditions = {
            "must": [
                {"key": "file_path", "match": {"value": str(abs_path)}}
            ]
        }
        
        # If specific chunk range requested
        if end_chunk is not None:
            filter_conditions["must"].append({
                "key": "chunk_index", 
                "range": {
                    "gte": start_chunk,
                    "lte": end_chunk
                }
            })
        elif start_chunk > 0:
            filter_conditions["must"].append({
                "key": "chunk_index",
                "range": {"gte": start_chunk}
            })
        
        # Fetch all matching chunks
        chunks = []
        offset = None
        limit = 100  # Fetch in batches
        
        while True:
            results, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_conditions,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            chunks.extend(results)
            
            if next_offset is None:
                break
            offset = next_offset
        
        # Sort by chunk index
        chunks.sort(key=lambda x: x.payload.get("chunk_index", 0))
        
        # Format results
        formatted_chunks = []
        for chunk in chunks:
            payload = chunk.payload
            formatted_chunks.append({
                "chunk_index": payload.get("chunk_index", 0),
                "line_start": payload.get("line_start", 0),
                "line_end": payload.get("line_end", 0),
                "content": payload.get("content", ""),
                "chunk_type": payload.get("chunk_type", "general"),
                "metadata": {
                    k: v for k, v in payload.items() 
                    if k not in ["content", "file_path", "chunk_index", "line_start", "line_end"]
                }
            })
        
        # Build full content
        if formatted_chunks:
            full_content = "\n".join([
                f"=== Chunk {c['chunk_index']} [Lines {c['line_start']}-{c['line_end']}] ===\n{c['content']}"
                for c in formatted_chunks
            ])
        else:
            full_content = ""
        
        return {
            "file_path": str(abs_path),
            "file_type": file_type,
            "total_chunks": len(formatted_chunks),
            "chunks": formatted_chunks,
            "full_content": full_content,
            "collection": collection_name
        }
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

def delete_file_chunks(file_path: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete all chunks for a specific file from a collection
    
    Args:
        file_path: Path to the file whose chunks should be deleted
        collection_name: Optional collection name. If not provided, will determine from file type
    
    Returns:
        Dict with deletion results or error
    """
    logger = get_logger()
    
    try:
        # Input validation
        if not file_path or not isinstance(file_path, str):
            return {
                "error": "Invalid file path",
                "error_code": "INVALID_INPUT",
                "details": "File path must be a non-empty string"
            }
        
        # Resolve to absolute path
        abs_path = Path(file_path).resolve()
        
        # Get services
        qdrant_client = get_qdrant_client()
        
        # Determine collection if not provided
        if collection_name is None:
            collection_name = get_collection_name(str(abs_path), "code")
        
        # Check if collection exists
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if collection_name not in collections:
                return {
                    "error": f"Collection '{collection_name}' does not exist",
                    "file_path": str(abs_path),
                    "deleted_points": 0
                }
        except Exception:
            return {
                "error": f"Could not access collection '{collection_name}'",
                "file_path": str(abs_path),
                "deleted_points": 0
            }
        
        # Create filter for the specific file
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        filter_condition = Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))]
        )
        
        # Count existing points before deletion
        count_response = qdrant_client.count(
            collection_name=collection_name,
            count_filter=filter_condition,
            exact=True
        )
        points_before = count_response.count
        
        # Delete points matching the filter
        delete_response = qdrant_client.delete(
            collection_name=collection_name,
            points_selector=filter_condition
        )
        
        console_logger.info(f"Deleted {points_before} chunks for file {file_path}", extra={
            "operation": "delete_file_chunks",
            "file_path": str(abs_path),
            "collection": collection_name,
            "deleted_points": points_before,
            "status": "success"
        })
        
        return {
            "file_path": str(abs_path),
            "collection": collection_name,
            "deleted_points": points_before,
            "operation_id": delete_response.operation_id if hasattr(delete_response, 'operation_id') else None
        }
        
    except Exception as e:
        console_logger.error(f"Failed to delete file chunks for {file_path}: {str(e)}", extra={
            "operation": "delete_file_chunks",
            "file_path": file_path,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "file_path": file_path}

@mcp.tool()
def detect_changes(directory: str = ".") -> Dict[str, Any]:
    """
    Compare current filesystem state with indexed state in Qdrant.
    
    Args:
        directory: Directory to scan for changes (default: current directory)
    
    Returns:
        Dict with lists of added/modified/unchanged/deleted files
    """
    logger = get_logger()
    
    try:
        import os
        from pathlib import Path
        
        # Input validation
        if not directory or not isinstance(directory, str):
            return {
                "error": "Invalid directory path",
                "error_code": "INVALID_INPUT",
                "details": "Directory must be a non-empty string"
            }
        
        # Resolve to absolute path
        abs_directory = Path(directory).resolve()
        if not abs_directory.exists():
            return {
                "error": f"Directory does not exist: {abs_directory}",
                "error_code": "DIRECTORY_NOT_FOUND"
            }
        
        if not abs_directory.is_dir():
            return {
                "error": f"Path is not a directory: {abs_directory}",
                "error_code": "NOT_A_DIRECTORY"
            }
        
        console_logger.info(f"Detecting changes in directory: {abs_directory}", extra={
            "operation": "detect_changes",
            "directory": str(abs_directory)
        })
        
        # Get services
        qdrant_client = get_qdrant_client()
        
        # Get current project context to determine which collections to check
        current_project = get_current_project()
        if current_project:
            collection_prefix = current_project['collection_prefix']
        else:
            collection_prefix = "global_"
        
        # Get all collections for this project
        all_collections = [c.name for c in qdrant_client.get_collections().collections]
        project_collections = [c for c in all_collections if c.startswith(collection_prefix)]
        
        # Track indexed files with their hashes
        indexed_files = {}  # file_path -> hash
        
        # Query each collection for indexed files
        
        for collection in project_collections:
            try:
                # Scroll through ALL points to get file metadata
                # Need to paginate through all results, not just first batch
                offset = None
                while True:
                    scroll_result = qdrant_client.scroll(
                        collection_name=collection,
                        limit=1000,  # Smaller batch size for better memory usage
                        offset=offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points, next_offset = scroll_result
                    
                    if not points:
                        break
                    
                    for point in points:
                        payload = point.payload
                        if 'file_path' in payload:
                            file_path = payload['file_path']
                            file_hash = payload.get('file_hash', '')
                            
                            # Only consider files within the target directory
                            try:
                                file_abs_path = Path(file_path).resolve()
                                if file_abs_path.is_relative_to(abs_directory):
                                    # Use absolute path as key to avoid duplicates
                                    abs_path_str = str(file_abs_path)
                                    
                                    # Only track files with the lowest chunk_index to avoid duplicates
                                    chunk_index = payload.get('chunk_index', 0)
                                    if chunk_index == 0:
                                        indexed_files[abs_path_str] = file_hash
                            except (ValueError, OSError):
                                continue  # Skip invalid paths
                    
                    # Move to next page
                    offset = next_offset
                    if offset is None:
                        break
                            
            except Exception as e:
                logger.warning(f"Failed to scan collection {collection}: {str(e)}")
                continue
        
        # Define safe file hash function
        def get_file_hash_safe(file_path: Path) -> str:
            """Get file hash safely, returning empty string on error"""
            try:
                return calculate_file_hash(str(file_path))
            except Exception:
                return ""
        
        # Scan current filesystem
        current_files = {}  # file_path -> hash
        
        # Define patterns to include (similar to indexing logic)
        include_patterns = [
            "*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.cpp", "*.c", "*.h",
            "*.go", "*.rs", "*.php", "*.rb", "*.swift", "*.kt", "*.scala", "*.sh",
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", "*.conf",
            "*.xml", "*.env", "*.properties", "*.config",
            "*.md", "*.markdown", "*.rst", "*.txt", "*.mdx"
        ]
        
        # Load exclude patterns from .ragignore if available
        exclude_dirs, exclude_patterns = load_ragignore_patterns(abs_directory)
        
        def should_include_file(file_path: Path) -> bool:
            """Check if file should be included based on patterns"""
            # Check if file matches include patterns
            file_matches = any(file_path.match(pattern) for pattern in include_patterns)
            if not file_matches:
                return False
            
            # Check if file is in excluded directory
            path_str = str(file_path)
            for exclude_dir in exclude_dirs:
                if f"/{exclude_dir}/" in path_str or path_str.startswith(f"{exclude_dir}/"):
                    return False
            
            # Check if filename matches exclusion patterns
            file_name = file_path.name
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(file_name, pattern):
                    return False
            
            return True
        
        # Walk through directory
        for root, dirs, files in os.walk(abs_directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                if should_include_file(file_path):
                    file_hash = get_file_hash_safe(file_path)
                    if file_hash:  # Only include files we can hash
                        # Use absolute path to match indexed files
                        abs_path_str = str(file_path.resolve())
                        current_files[abs_path_str] = file_hash
        
        # Compare states
        indexed_file_set = set(indexed_files.keys())
        current_file_set = set(current_files.keys())
        
        # Categorize files
        added_files = []
        modified_files = []
        unchanged_files = []
        deleted_files = []
        
        # Files in current but not indexed = added
        for file_path in current_file_set - indexed_file_set:
            added_files.append(file_path)
        
        # Files in indexed but not current = deleted  
        for file_path in indexed_file_set - current_file_set:
            deleted_files.append(file_path)
        
        # Files in both = check if modified
        for file_path in current_file_set & indexed_file_set:
            current_hash = current_files[file_path]
            indexed_hash = indexed_files[file_path]
            
            if current_hash == indexed_hash:
                unchanged_files.append(file_path)
            else:
                modified_files.append(file_path)
        
        result = {
            "directory": str(abs_directory),
            "added": sorted(added_files),
            "modified": sorted(modified_files),
            "unchanged": sorted(unchanged_files),
            "deleted": sorted(deleted_files),
            "summary": {
                "total_indexed": len(indexed_files),
                "total_current": len(current_files),
                "added_count": len(added_files),
                "modified_count": len(modified_files),
                "unchanged_count": len(unchanged_files),
                "deleted_count": len(deleted_files)
            }
        }
        
        # Add debug logging
        if len(added_files) > 50:  # Suspicious number of "new" files
            logger.warning(f"Large number of files marked as added: {len(added_files)}. "
                         f"This might indicate an issue with change detection. "
                         f"Total indexed: {len(indexed_files)}, Total current: {len(current_files)}")
        
        console_logger.info(f"Change detection complete", extra={
            "operation": "detect_changes",
            "directory": str(abs_directory),
            "added": len(added_files),
            "modified": len(modified_files),
            "unchanged": len(unchanged_files),
            "deleted": len(deleted_files),
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        console_logger.error(f"Failed to detect changes in {directory}: {str(e)}", extra={
            "operation": "detect_changes",
            "directory": directory,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "directory": directory}

@mcp.tool()
def index_code(file_path: str, force_global: bool = False) -> Dict[str, Any]:
    """
    Index a source code file
    
    Args:
        file_path: Path to the file to index
        force_global: If True, index to global collection instead of project
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Input validation
        if not file_path or not isinstance(file_path, str):
            return {
                "error": "Invalid file path",
                "error_code": "INVALID_INPUT",
                "details": "File path must be a non-empty string"
            }
        
        # Security check - prevent path traversal
        if "../" in file_path or file_path.startswith("/etc/") or file_path.startswith("/sys/"):
            return {
                "error": "Access denied",
                "error_code": "SECURITY_VIOLATION",
                "details": "Path traversal or system file access attempted"
            }
        
        from qdrant_client.http.models import PointStruct
        
        console_logger.info(f"Starting index_code for {file_path}", extra={
            "operation": "index_code",
            "file_path": file_path,
            "force_global": force_global
        })
        
        # Resolve to absolute path
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Get services
        code_indexer = get_code_indexer()
        embedding_model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
        # Get file modification time and hash
        try:
            mod_time = abs_path.stat().st_mtime
            file_hash = calculate_file_hash(str(abs_path))
        except:
            mod_time = time.time()  # Use current time as fallback
            file_hash = None  # No hash if file reading fails
        
        # Determine collection
        if force_global:
            collection_name = "global_code"
        else:
            collection_name = get_collection_name(str(abs_path), "code")
        ensure_collection(collection_name)
        
        # Index file
        chunks = code_indexer.index_file(str(abs_path))
        
        # Convert to points
        points = []
        current_project = get_current_project()
        
        # Calculate relative path if in current project
        display_path = str(abs_path)
        if current_project:
            try:
                rel_path = abs_path.relative_to(Path(current_project["root"]))
                display_path = str(rel_path)
            except ValueError:
                pass
        
        for chunk in chunks:
            embedding = embedding_model.encode(chunk.content).tolist()
            
            # Generate unique chunk ID
            chunk_id = hashlib.md5(f"{abs_path}_{chunk.chunk_index}".encode()).hexdigest()
            
            # Build payload with hierarchical metadata
            payload = {
                "file_path": str(abs_path),
                "display_path": display_path,
                "chunk_index": chunk.chunk_index,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "language": chunk.metadata.get("language", ""),
                "content": chunk.content,
                "chunk_type": chunk.metadata.get("chunk_type", "general"),
                "project": collection_name.rsplit('_', 1)[0],
                "modified_at": mod_time,
                "file_hash": file_hash
            }
            
            # Add hierarchical metadata if available
            if "hierarchy" in chunk.metadata:
                payload["hierarchy"] = chunk.metadata["hierarchy"]
            if "name" in chunk.metadata:
                payload["name"] = chunk.metadata["name"]
            
            # Add any additional metadata
            for key in ["async", "decorators", "args", "returns", "is_method", "bases", 
                       "method_count", "import_count", "modules", "has_methods", "dependencies"]:
                if key in chunk.metadata:
                    payload[key] = chunk.metadata[key]
            
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Store in Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Update BM25 index
            hybrid_searcher = get_hybrid_searcher()
            documents = []
            for chunk in chunks:
                doc = {
                    "id": f"{str(abs_path)}_{chunk.chunk_index}",
                    "content": chunk.content,
                    "file_path": str(abs_path),
                    "chunk_index": chunk.chunk_index,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "language": chunk.metadata.get("language", ""),
                    "chunk_type": chunk.metadata.get("chunk_type", "general")
                }
                documents.append(doc)
            
            # Get all documents in collection for BM25 update
            # Note: In production, we'd want incremental updates
            all_docs = []
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            for point in scroll_result[0]:
                doc = {
                    "id": f"{point.payload['file_path']}_{point.payload['chunk_index']}",
                    "content": point.payload.get("content", ""),
                    "file_path": point.payload.get("file_path", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "line_start": point.payload.get("line_start", 0),
                    "line_end": point.payload.get("line_end", 0),
                    "language": point.payload.get("language", ""),
                    "chunk_type": point.payload.get("chunk_type", "general")
                }
                all_docs.append(doc)
                
            hybrid_searcher.bm25_manager.update_index(collection_name, all_docs)
        
        duration_ms = (time.time() - start_time) * 1000
        result = {
            "indexed": len(chunks),
            "file_path": display_path,
            "collection": collection_name,
            "language": chunks[0].metadata.get("language", "unknown") if chunks else "unknown",
            "project_context": current_project["name"] if current_project else "global"
        }
        
        console_logger.info(f"Completed index_code for {display_path}", extra={
            "operation": "index_code",
            "file_path": display_path,
            "duration_ms": duration_ms,
            "chunks_indexed": len(chunks),
            "collection": collection_name,
            "status": "success"
        })
        
        return result
        
    except QdrantConnectionError as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Connection error during index_code for {file_path}: {str(e)}", extra={
            "operation": "index_code",
            "file_path": file_path,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": "QdrantConnectionError",
            "status": "error"
        })
        return {
            "error": "Database connection failed",
            "error_code": "DB_CONNECTION_ERROR",
            "details": e.details,
            "file_path": file_path
        }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        
        # Provide user-friendly error messages
        if "not found" in error_msg.lower():
            user_error = "File not found"
            error_code = "FILE_NOT_FOUND"
        elif "permission" in error_msg.lower():
            user_error = "Permission denied"
            error_code = "PERMISSION_DENIED"
        else:
            user_error = "Failed to index file"
            error_code = "INDEX_ERROR"
        
        console_logger.error(f"Failed index_code for {file_path}: {error_msg}", extra={
            "operation": "index_code",
            "file_path": file_path,
            "duration_ms": duration_ms,
            "error": error_msg,
            "error_type": type(e).__name__,
            "status": "error"
        })
        
        return {
            "error": user_error,
            "error_code": error_code,
            "details": error_msg,
            "file_path": file_path
        }

@mcp.tool()
def index_documentation(file_path: str, force_global: bool = False) -> Dict[str, Any]:
    """
    Index a documentation file (markdown, rst, etc.)
    
    Args:
        file_path: Path to the documentation file to index
        force_global: If True, index to global collection instead of project
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Input validation
        if not file_path or not isinstance(file_path, str):
            return {
                "error": "Invalid file path",
                "error_code": "INVALID_INPUT",
                "details": "File path must be a non-empty string"
            }
        
        # Security check - prevent path traversal
        if "../" in file_path or file_path.startswith("/etc/") or file_path.startswith("/sys/"):
            return {
                "error": "Access denied",
                "error_code": "SECURITY_VIOLATION",
                "details": "Path traversal or system file access attempted"
            }
        
        from qdrant_client.http.models import PointStruct
        
        abs_path = Path(file_path).resolve()
        
        # Check if file exists
        if not abs_path.exists():
            return {
                "error": "File not found",
                "error_code": "FILE_NOT_FOUND",
                "details": f"File {file_path} does not exist"
            }
        
        # Get documentation indexer
        doc_indexer = get_documentation_indexer()
        
        # Check if file is supported
        if not doc_indexer.is_supported(str(abs_path)):
            return {
                "error": "Unsupported file type",
                "error_code": "UNSUPPORTED_FILE_TYPE",
                "details": f"File type not supported for documentation indexing: {abs_path.suffix}"
            }
        
        # Get file hash
        try:
            file_hash = calculate_file_hash(str(abs_path))
        except:
            file_hash = None  # No hash if file reading fails
        
        # Determine collection
        if force_global:
            collection_name = "global_documentation"
        else:
            collection_name = get_collection_name(str(abs_path), "documentation")
        
        # Ensure collection exists
        ensure_collection(collection_name)
        
        # Index the file
        chunks = doc_indexer.index_file(str(abs_path))
        
        if not chunks:
            return {
                "error": "No content extracted",
                "error_code": "NO_CONTENT",
                "details": "No documentation chunks could be extracted from the file"
            }
        
        # Get embedding model and Qdrant client
        model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
        # Prepare points for Qdrant
        points = []
        for chunk in chunks:
            # Create unique ID for chunk
            chunk_id = hashlib.md5(
                f"{str(abs_path)}_{chunk['metadata']['chunk_index']}".encode()
            ).hexdigest()
            
            # Generate embedding
            embedding = model.encode(chunk['content']).tolist()
            
            # Prepare payload
            payload = {
                "file_path": str(abs_path),
                "file_name": abs_path.name,
                "content": chunk['content'],
                "doc_type": chunk['metadata'].get('doc_type', 'markdown'),
                "chunk_index": chunk['metadata']['chunk_index'],
                "chunk_type": chunk['metadata'].get('chunk_type', 'section'),
                "title": chunk['metadata'].get('title', ''),
                "heading": chunk['metadata'].get('heading', ''),
                "heading_hierarchy": chunk['metadata'].get('heading_hierarchy', []),
                "heading_level": chunk['metadata'].get('heading_level', 0),
                "has_code_blocks": chunk['metadata'].get('has_code_blocks', False),
                "code_languages": chunk['metadata'].get('code_languages', []),
                "internal_links": chunk['metadata'].get('internal_links', []),
                "external_links": chunk['metadata'].get('external_links', []),
                "modified_at": chunk['metadata'].get('modified_at', ''),
                "file_hash": file_hash,
                "collection": collection_name
            }
            
            # Add frontmatter if present
            if 'frontmatter' in chunk['metadata'] and chunk['metadata']['frontmatter']:
                payload['frontmatter'] = chunk['metadata']['frontmatter']
            
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload=payload
            )
            points.append(point)
        
        # Store in Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Update BM25 index for hybrid search
            hybrid_searcher = get_hybrid_searcher()
            documents = []
            for chunk in chunks:
                doc = {
                    "id": f"{str(abs_path)}_{chunk['metadata']['chunk_index']}",
                    "content": chunk['content'],
                    "file_path": str(abs_path),
                    "chunk_index": chunk['metadata']['chunk_index'],
                    "doc_type": chunk['metadata'].get('doc_type', 'markdown'),
                    "heading": chunk['metadata'].get('heading', ''),
                    "chunk_type": chunk['metadata'].get('chunk_type', 'section')
                }
                documents.append(doc)
            
            # Update BM25 index
            hybrid_searcher = get_hybrid_searcher()
            hybrid_searcher.bm25_manager.update_index(collection_name, documents)
        
        duration_ms = (time.time() - start_time) * 1000
        
        console_logger.info(f"Successfully indexed documentation {file_path}", extra={
            "operation": "index_documentation",
            "file_path": str(abs_path),
            "chunks": len(chunks),
            "collection": collection_name,
            "duration_ms": duration_ms,
            "status": "success"
        })
        
        return {
            "indexed": len(chunks),
            "file_path": str(abs_path),
            "file_name": abs_path.name,
            "doc_type": chunks[0]['metadata'].get('doc_type', 'markdown') if chunks else 'markdown',
            "title": chunks[0]['metadata'].get('title', '') if chunks else '',
            "collection": collection_name,
            "has_code_blocks": any(c['metadata'].get('has_code_blocks', False) for c in chunks),
            "code_languages": list(set(lang for c in chunks for lang in c['metadata'].get('code_languages', []))),
            "message": f"Successfully indexed {len(chunks)} documentation chunks"
        }
        
    except QdrantConnectionError as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Qdrant connection failed for {file_path}", extra={
            "operation": "index_documentation", 
            "file_path": file_path,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": "QdrantConnectionError",
            "status": "error"
        })
        return {
            "error": "Database connection failed",
            "error_code": "CONNECTION_ERROR",
            "details": e.details,
            "file_path": file_path
        }
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        error_msg = str(e)
        
        # Provide user-friendly error messages
        if "not found" in error_msg.lower():
            user_error = "File not found"
            error_code = "FILE_NOT_FOUND"
        elif "permission" in error_msg.lower():
            user_error = "Permission denied"
            error_code = "PERMISSION_DENIED"
        else:
            user_error = "Failed to index documentation"
            error_code = "INDEX_ERROR"
        
        console_logger.error(f"Failed index_documentation for {file_path}: {error_msg}", extra={
            "operation": "index_documentation",
            "file_path": file_path,
            "duration_ms": duration_ms,
            "error": error_msg,
            "error_type": type(e).__name__,
            "status": "error"
        })
        
        return {
            "error": user_error,
            "error_code": error_code,
            "details": error_msg,
            "file_path": file_path
        }

def load_ragignore_patterns(directory: Path) -> Tuple[Set[str], Set[str]]:
    """
    Load ignore patterns from .ragignore file in the directory or its parents.
    
    Returns:
        Tuple of (exclude_dirs, exclude_patterns)
    """
    exclude_dirs = set()
    exclude_patterns = set()
    
    # Default patterns (fallback if no .ragignore found)
    default_exclude_dirs = {
        'node_modules', '__pycache__', '.git', '.venv', 'venv', 
        'env', '.env', 'dist', 'build', 'target', '.pytest_cache',
        '.mypy_cache', '.coverage', 'htmlcov', '.tox', 'data',
        'logs', 'tmp', 'temp', '.idea', '.vscode', '.vs',
        'qdrant_storage', 'models', '.cache'
    }
    
    default_exclude_patterns = {
        '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so', '*.dylib',
        '*.dll', '*.class', '*.log', '*.lock', '*.swp', '*.swo',
        '*.bak', '*.tmp', '*.temp', '*.old', '*.orig', '*.rej',
        '.env*', '*.sqlite', '*.db', '*.pid'
    }
    
    # Look for .ragignore file in directory and parent directories
    ragignore_path = None
    for parent in [directory] + list(directory.parents):
        potential_path = parent / '.ragignore'
        if potential_path.exists():
            ragignore_path = potential_path
            break
    
    if not ragignore_path:
        # No .ragignore found, use defaults
        return default_exclude_dirs, default_exclude_patterns
    
    # Parse .ragignore file
    try:
        with open(ragignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Handle directory patterns (ending with /)
                if line.endswith('/'):
                    exclude_dirs.add(line.rstrip('/'))
                else:
                    # File patterns
                    exclude_patterns.add(line)
        
        console_logger.info(f"Loaded .ragignore from {ragignore_path}")
        console_logger.debug(f"Exclude dirs: {len(exclude_dirs)}, patterns: {len(exclude_patterns)}")
        
    except Exception as e:
        console_logger.warning(f"Error reading .ragignore: {e}, using defaults")
        return default_exclude_dirs, default_exclude_patterns
    
    # If .ragignore was empty or had no valid patterns, use defaults
    if not exclude_dirs and not exclude_patterns:
        return default_exclude_dirs, default_exclude_patterns
    
    return exclude_dirs, exclude_patterns

@mcp.tool()
def index_directory(directory: str = None, patterns: List[str] = None, recursive: bool = True) -> Dict[str, Any]:
    """
    Index files in a directory.
    
    Args:
        directory: Directory to index (REQUIRED - must be absolute path or will be resolved from client's context)
        patterns: File patterns to include
        recursive: Whether to search recursively
    """
    try:
        if patterns is None:
            patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go", "*.rs", 
                       "*.sh", "*.bash", "*.zsh", "*.fish",  # Shell scripts
                       "*.json", "*.yaml", "*.yml", "*.xml", "*.toml", "*.ini",
                       "*.md", "*.markdown", "*.rst", "*.txt",  # Documentation files
                       ".gitignore", ".dockerignore", ".prettierrc*", ".eslintrc*", 
                       ".editorconfig", ".npmrc", ".yarnrc", ".ragignore"]
        
        # Validate directory parameter
        if not directory:
            return {
                "error": "Directory parameter is required",
                "error_code": "MISSING_DIRECTORY",
                "details": "Please specify a directory to index. Use absolute paths for best results."
            }
        
        # Resolve directory
        if directory == ".":
            # Special handling for current directory
            # Try to use client's working directory if available
            client_cwd = os.environ.get('MCP_CLIENT_CWD')
            if client_cwd:
                dir_path = Path(client_cwd).resolve()
                console_logger.info(f"Using client working directory from MCP_CLIENT_CWD: {dir_path}")
            else:
                # Warn about potential mismatch
                dir_path = Path.cwd()
                console_logger.warning(f"Using MCP server's working directory ({dir_path}) for '.'. "
                             "This may not match your actual location. Use absolute paths or set MCP_CLIENT_CWD.")
                # Include warning in response
                console_logger.warning("⚠️  Using MCP server's directory. For accurate indexing, use absolute paths.")
        else:
            # Use provided directory
            dir_path = Path(directory).resolve()
            if not dir_path.is_absolute():
                console_logger.info(f"Resolved relative path {directory} to {dir_path}")
        
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}
        
        # Get current project info based on the directory being indexed
        current_project = get_current_project(client_directory=str(dir_path))
        
        # Load exclusion patterns from .ragignore file
        exclude_dirs, exclude_patterns = load_ragignore_patterns(dir_path)
        
        # Create a progress callback
        start_time = time.time()
        
        def report_progress(current: int, total: int, current_file: str = ""):
            """Report progress to logger"""
            if total == 0:
                return
            
            percent = (current / total) * 100
            elapsed = time.time() - start_time
            
            if current > 0:
                avg_time_per_file = elapsed / current
                remaining_files = total - current
                eta_seconds = avg_time_per_file * remaining_files
                eta_str = f", ETA: {int(eta_seconds)}s"
            else:
                eta_str = ""
            
            logger = get_logger()
            console_logger.info(
                f"Indexing progress: {current}/{total} files ({percent:.1f}%){eta_str}",
                extra={
                    "operation": "index_directory_progress",
                    "current": current,
                    "total": total,
                    "percent": percent,
                    "current_file": current_file
                }
            )
        
        indexed_files = []
        errors = []
        collections_used = set()
        
        # First, collect all files to process (for accurate progress reporting)
        files_to_process = []
        for pattern in patterns:
            glob_func = dir_path.rglob if recursive else dir_path.glob
            for file_path in glob_func(pattern):
                if file_path.is_file():
                    try:
                        # Check if path contains any excluded directory
                        path_str = str(file_path)
                        should_skip = False
                        for exclude_dir in exclude_dirs:
                            if f"/{exclude_dir}/" in path_str or path_str.startswith(f"{exclude_dir}/"):
                                should_skip = True
                                break
                        
                        if should_skip:
                            continue
                        
                        # Check if filename matches exclusion patterns
                        file_name = file_path.name
                        should_skip = False
                        
                        # Check against exclude patterns (supports wildcards)
                        for pattern in exclude_patterns:
                            if fnmatch.fnmatch(file_name, pattern):
                                should_skip = True
                                break
                        
                        if should_skip:
                            continue
                        
                        files_to_process.append(file_path)
                    except Exception as e:
                        errors.append({"file": str(file_path), "error": str(e)})
        
        # Now process files with progress reporting
        total_files = len(files_to_process)
        if total_files > 0:
            logger = get_logger()
            console_logger.info(f"Starting to index {total_files} files from {directory}")
            
            # Report initial progress
            report_progress(0, total_files)
            
            for idx, file_path in enumerate(files_to_process):
                try:
                    # Determine file type
                    ext = file_path.suffix.lower()
                    if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                        result = index_config(str(file_path))
                    elif ext in ['.md', '.markdown', '.rst', '.txt']:
                        result = index_documentation(str(file_path))
                    else:
                        result = index_code(str(file_path))
                    
                    if "error" not in result:
                        indexed_files.append(result.get("file_path", str(file_path)))
                        if "collection" in result:
                            collections_used.add(result["collection"])
                    else:
                        errors.append({"file": str(file_path), "error": result["error"]})
                    
                    # Report progress every 10 files or at the end
                    if (idx + 1) % 10 == 0 or (idx + 1) == total_files:
                        report_progress(idx + 1, total_files, str(file_path.name))
                    
                except Exception as e:
                    errors.append({"file": str(file_path), "error": str(e)})
        
        return {
            "indexed_files": indexed_files,
            "total": len(indexed_files),
            "collections": list(collections_used),
            "project_context": current_project["name"] if current_project else "no project",
            "directory": str(dir_path),
            "errors": errors if errors else None
        }
        
    except Exception as e:
        return {"error": str(e), "directory": directory}

@mcp.tool()
def reindex_directory(
    directory: str = ".", 
    patterns: List[str] = None, 
    recursive: bool = True,
    force: bool = False,
    incremental: bool = True
) -> Dict[str, Any]:
    """
    Reindex a directory by first clearing existing project data.
    
    This prevents stale data from deleted/moved files.
    Use this when files have been renamed, moved, or deleted.
    
    Args:
        directory: Directory to reindex (default: current directory)
        patterns: Optional file patterns to include
        recursive: Whether to search subdirectories
        force: Skip confirmation (for automation)
        incremental: If True, use smart reindex to only process changed files (default: True)
    
    Returns:
        Reindex results including what was cleared and indexed
    """
    logger = get_logger()
    start_time = time.time()
    
    console_logger.info(f"Starting reindex_directory for {directory}", extra={
        "operation": "reindex_directory",
        "directory": directory,
        "recursive": recursive,
        "force": force,
        "incremental": incremental
    })
    
    try:
        # Get current project context
        current_project = get_current_project()
        if not current_project and not force:
            return {
                "error": "No project context found. Use force=True to reindex anyway.",
                "directory": directory
            }
        
        # If force=True or incremental=False, use the original behavior (clear all collections)
        if force or not incremental:
            console_logger.info("Using full reindex mode (clearing all collections)")
            
            # Clear existing collections
            clear_result = clear_project_collections()
            
            # Check if clear had errors
            if clear_result.get("errors"):
                return {
                    "error": "Failed to clear some collections",
                    "clear_errors": clear_result["errors"],
                    "directory": directory
                }
            
            # Now index the directory
            index_result = index_directory(directory, patterns, recursive)
            
            # Combine results
            duration_ms = (time.time() - start_time) * 1000
            result = {
                "directory": directory,
                "mode": "full_reindex",
                "cleared_collections": clear_result.get("cleared_collections", []),
                "indexed_files": index_result.get("indexed_files", []),
                "total_indexed": index_result.get("total", 0),
                "collections": index_result.get("collections", []),
                "project_context": current_project["name"] if current_project else "no project",
                "errors": index_result.get("errors"),
                "message": f"Full reindex: {index_result.get('total', 0)} files after clearing {len(clear_result.get('cleared_collections', []))} collections"
            }
        
        else:
            # Incremental reindex mode - use detect_changes to only process changed files
            console_logger.info("Using incremental reindex mode (smart change detection)")
            
            # Detect changes first
            changes_result = detect_changes(directory)
            if "error" in changes_result:
                return {
                    "error": f"Failed to detect changes: {changes_result['error']}",
                    "directory": directory
                }
            
            # Get lists of files to process
            added_files = changes_result.get("added", [])
            modified_files = changes_result.get("modified", [])
            deleted_files = changes_result.get("deleted", [])
            unchanged_files = changes_result.get("unchanged", [])
            
            # Process deletions first - remove chunks for deleted files
            deleted_count = 0
            deletion_errors = []
            for file_path in deleted_files:
                try:
                    # Determine collection name based on file type
                    path_obj = Path(file_path)
                    ext = path_obj.suffix.lower()
                    if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                        collection_type = "config"
                    elif ext in ['.md', '.markdown', '.rst', '.txt']:
                        collection_type = "documentation"
                    else:
                        collection_type = "code"
                    
                    collection_name = get_collection_name(file_path, collection_type)
                    delete_result = delete_file_chunks(file_path, collection_name)
                    
                    if "error" not in delete_result:
                        deleted_count += delete_result.get("deleted_points", 0)
                        console_logger.info(f"Removed chunks for deleted file: {file_path}")
                    else:
                        deletion_errors.append({
                            "file": file_path,
                            "error": delete_result["error"]
                        })
                except Exception as e:
                    deletion_errors.append({
                        "file": file_path,
                        "error": str(e)
                    })
            
            # Process added and modified files
            files_to_index = added_files + modified_files
            indexed_files = []
            indexing_errors = []
            collections_used = set()
            
            if files_to_index:
                console_logger.info(f"Indexing {len(files_to_index)} changed files ({len(added_files)} added, {len(modified_files)} modified)")
                
                for file_path in files_to_index:
                    try:
                        # If file was modified, delete existing chunks first
                        if file_path in modified_files:
                            path_obj = Path(file_path)
                            ext = path_obj.suffix.lower()
                            if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                                collection_type = "config"
                            elif ext in ['.md', '.markdown', '.rst', '.txt']:
                                collection_type = "documentation"
                            else:
                                collection_type = "code"
                            
                            collection_name = get_collection_name(file_path, collection_type)
                            delete_file_chunks(file_path, collection_name)
                        
                        # Index the file
                        path_obj = Path(file_path)
                        ext = path_obj.suffix.lower()
                        if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                            result = index_config(file_path)
                        elif ext in ['.md', '.markdown', '.rst', '.txt']:
                            result = index_documentation(file_path)
                        else:
                            result = index_code(file_path)
                        
                        if "error" not in result:
                            indexed_files.append(file_path)
                            if "collection" in result:
                                collections_used.add(result["collection"])
                        else:
                            indexing_errors.append({
                                "file": file_path,
                                "error": result["error"]
                            })
                    
                    except Exception as e:
                        indexing_errors.append({
                            "file": file_path,
                            "error": str(e)
                        })
            
            # Combine results
            duration_ms = (time.time() - start_time) * 1000
            total_errors = deletion_errors + indexing_errors
            
            result = {
                "directory": directory,
                "mode": "incremental_reindex",
                "changes_detected": {
                    "added": len(added_files),
                    "modified": len(modified_files),
                    "deleted": len(deleted_files),
                    "unchanged": len(unchanged_files)
                },
                "indexed_files": indexed_files,
                "total_indexed": len(indexed_files),
                "deleted_chunks": deleted_count,
                "collections": list(collections_used),
                "project_context": current_project["name"] if current_project else "no project",
                "errors": total_errors if total_errors else None,
                "message": f"Incremental reindex: {len(indexed_files)} files indexed, {deleted_count} chunks removed from {len(deleted_files)} deleted files, {len(unchanged_files)} files unchanged"
            }
        
        console_logger.info(f"Completed reindex_directory for {directory}", extra={
            "operation": "reindex_directory",
            "directory": directory,
            "duration_ms": duration_ms,
            "mode": result.get("mode", "unknown"),
            "files_indexed": result.get("total_indexed", 0),
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Failed reindex_directory for {directory}: {str(e)}", extra={
            "operation": "reindex_directory",
            "directory": directory,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "directory": directory}

@mcp.tool()
def search(
    query: str, 
    n_results: int = 5, 
    cross_project: bool = False, 
    search_mode: str = "hybrid", 
    include_dependencies: bool = False, 
    include_context: bool = True, 
    context_chunks: int = 1,
    # New progressive context parameters
    context_level: str = "auto",
    progressive_mode: Optional[bool] = None,
    include_expansion_options: bool = True,
    semantic_cache: bool = True
) -> Dict[str, Any]:
    """
    Search indexed content (defaults to current project only)
    
    Args:
        query: Search query
        n_results: Number of results
        cross_project: If True, search across all projects (default: False)
        search_mode: Search mode - "vector", "keyword", or "hybrid" (default: "hybrid")
        include_dependencies: If True, include files that import/are imported by the results
        include_context: If True, include surrounding chunks for more context (default: True)
        context_chunks: Number of chunks before/after to include (default: 1, max: 3)
        context_level: Granularity level ("auto", "file", "class", "method", "full") - new in v0.3.2
        progressive_mode: Enable progressive features (None = auto-detect) - new in v0.3.2
        include_expansion_options: Include drill-down options - new in v0.3.2
        semantic_cache: Use semantic similarity caching - new in v0.3.2
    """
    logger = get_logger()
    start_time = time.time()
    
    # Input validation
    if not query or not isinstance(query, str):
        return {
            "error": "Invalid query",
            "error_code": "INVALID_INPUT",
            "details": "Query must be a non-empty string"
        }
    
    if len(query) > 1000:  # Reasonable limit
        return {
            "error": "Query too long",
            "error_code": "QUERY_TOO_LONG",
            "details": "Query must be less than 1000 characters"
        }
    
    if n_results < 1 or n_results > 100:
        return {
            "error": "Invalid result count",
            "error_code": "INVALID_RESULT_COUNT",
            "details": "n_results must be between 1 and 100"
        }
    
    console_logger.info(f"Starting search: {query[:50]}...", extra={
        "operation": "search",
        "query_length": len(query),
        "n_results": n_results,
        "cross_project": cross_project,
        "search_mode": search_mode
    })
    
    # Check if progressive context is enabled
    config = get_config()
    progressive_enabled = config.get("progressive_context", {}).get("enabled", False)
    
    if progressive_mode is None:
        # Auto-detect based on context_level
        progressive_mode = progressive_enabled and context_level != "full"
    
    # If progressive mode is requested and enabled, use progressive context manager
    if progressive_mode and progressive_enabled:
        try:
            # Get progressive context manager
            embedding_model = get_embedding_model()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embedding_model,
                config.get("progressive_context", {})
            )
            
            # Use progressive context
            progressive_result = progressive_manager.get_progressive_context(
                query=query,
                level=context_level,
                n_results=n_results,
                cross_project=cross_project,
                search_mode=search_mode,
                include_dependencies=include_dependencies,
                semantic_cache=semantic_cache
            )
            
            # Convert to standard response format with progressive metadata
            response = {
                "results": progressive_result.results,
                "query": query,
                "total": len(progressive_result.results),
                "search_mode": search_mode,
                "project_context": get_current_project()["name"] if get_current_project() else None,
                "search_scope": "cross-project" if cross_project else "current project"
            }
            
            # Add progressive metadata
            if include_expansion_options or progressive_result.level != "full":
                response["progressive"] = {
                    "level_used": progressive_result.level,
                    "token_estimate": progressive_result.token_estimate,
                    "token_reduction": progressive_result.token_reduction_percent,
                    "expansion_options": [
                        {
                            "type": opt.target_level,
                            "path": opt.target_path,
                            "estimated_tokens": opt.estimated_tokens,
                            "relevance": opt.relevance_score
                        }
                        for opt in progressive_result.expansion_options
                    ] if include_expansion_options else [],
                    "cache_hit": progressive_result.from_cache,
                    "query_intent": {
                        "type": progressive_result.query_intent.exploration_type,
                        "confidence": progressive_result.query_intent.confidence
                    } if progressive_result.query_intent else None
                }
            
            # Track in context tracking
            tracker = get_context_tracker()
            if tracker:
                tracker.track_search(query, response["results"], search_type="progressive")
            
            # Log completion
            console_logger.info(f"Progressive search completed", extra={
                "operation": "search_progressive",
                "duration": time.time() - start_time,
                "results_count": len(response["results"]),
                "level": progressive_result.level,
                "cache_hit": progressive_result.from_cache,
                "token_reduction": progressive_result.token_reduction_percent
            })
            
            return response
            
        except Exception as e:
            # Log error but fall back to regular search
            console_logger.warning(f"Progressive search failed, falling back to regular search: {e}", extra={
                "operation": "search_progressive_fallback",
                "error": str(e)
            })
            # Continue with regular search below
    
    # Regular search implementation (existing code)
    try:
        embedding_model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Determine which collections to search
        all_collections = [c.name for c in qdrant_client.get_collections().collections]
        
        if cross_project:
            # Search all collections
            search_collections = all_collections
        else:
            # Search only current project collections
            current_project = get_current_project()
            if current_project:
                search_collections = [
                    c for c in all_collections 
                    if c.startswith(current_project['collection_prefix'])
                ]
            else:
                # No project context - search global collections only
                search_collections = [
                    c for c in all_collections 
                    if c.startswith('global_')
                ]
        
        # Search across collections
        all_results = []
        hybrid_searcher = get_hybrid_searcher()
        
        for collection in search_collections:
            try:
                if search_mode == "vector":
                    # Vector search only
                    results = qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        limit=n_results
                    )
                    
                    for result in results:
                        payload = result.payload
                        all_results.append({
                            "score": result.score,
                            "type": "code" if "_code" in collection else "config",
                            "collection": collection,
                            **payload
                        })
                        
                elif search_mode == "keyword":
                    # BM25 keyword search only
                    bm25_results = hybrid_searcher.bm25_manager.search(
                        collection_name=collection,
                        query=query,
                        k=n_results
                    )
                    
                    # Fetch full documents from Qdrant
                    for doc_id, score in bm25_results:
                        # Parse doc_id to get file_path and chunk_index
                        parts = doc_id.rsplit('_', 1)
                        if len(parts) == 2:
                            file_path = parts[0]
                            chunk_index = int(parts[1])
                            
                            # Fetch from Qdrant
                            filter_conditions = {
                                "must": [
                                    {"key": "file_path", "match": {"value": file_path}},
                                    {"key": "chunk_index", "match": {"value": chunk_index}}
                                ]
                            }
                            
                            search_result = qdrant_client.search(
                                collection_name=collection,
                                query_vector=[0.0] * 384,  # Dummy vector
                                query_filter=filter_conditions,
                                limit=1
                            )
                            
                            if search_result:
                                payload = search_result[0].payload
                                all_results.append({
                                    "score": score,
                                    "type": "code" if "_code" in collection else "config",
                                    "collection": collection,
                                    **payload
                                })
                                
                else:  # hybrid mode
                    # Get vector search results
                    vector_results = []
                    vector_scores_map = {}  # Store vector scores by doc_id
                    search_results = qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        limit=n_results * 2  # Get more for fusion
                    )
                    
                    for result in search_results:
                        doc_id = f"{result.payload['file_path']}_{result.payload['chunk_index']}"
                        score = float(result.score)
                        vector_results.append((doc_id, score))
                        vector_scores_map[doc_id] = score
                    
                    # Get BM25 results
                    bm25_results = hybrid_searcher.bm25_manager.search(
                        collection_name=collection,
                        query=query,
                        k=n_results * 2
                    )
                    bm25_scores_map = {doc_id: score for doc_id, score in bm25_results}
                    
                    # Fuse results using linear combination for better score accuracy
                    fused_results = hybrid_searcher.linear_combination(
                        vector_results=vector_results,
                        bm25_results=bm25_results,
                        vector_weight=0.7,
                        bm25_weight=0.3
                    )
                    
                    # Fetch full documents for top results
                    for result in fused_results[:n_results]:
                        doc_id = result.content  # doc_id is stored in content field
                        parts = doc_id.rsplit('_', 1)
                        if len(parts) == 2:
                            file_path = parts[0]
                            chunk_index = int(parts[1])
                            
                            # Fetch from Qdrant
                            filter_conditions = {
                                "must": [
                                    {"key": "file_path", "match": {"value": file_path}},
                                    {"key": "chunk_index", "match": {"value": chunk_index}}
                                ]
                            }
                            
                            search_result = qdrant_client.search(
                                collection_name=collection,
                                query_vector=[0.0] * 384,  # Dummy vector
                                query_filter=filter_conditions,
                                limit=1
                            )
                            
                            if search_result:
                                payload = search_result[0].payload
                                all_results.append({
                                    "score": result.combined_score,
                                    "vector_score": vector_scores_map.get(doc_id, None),
                                    "bm25_score": bm25_scores_map.get(doc_id, None),
                                    "type": "code" if "_code" in collection else "config",
                                    "collection": collection,
                                    "search_mode": search_mode,
                                    **payload
                                })
                                
            except Exception as e:
                # Skip if collection doesn't exist
                logger.debug(f"Error searching collection {collection}: {e}")
                pass
        
        # Handle dependency inclusion if requested
        if include_dependencies and all_results:
            try:
                from utils.dependency_resolver import DependencyResolver
                
                # Collect unique file paths from results
                result_files = set()
                for result in all_results:
                    if 'file_path' in result:
                        result_files.add(result['file_path'])
                
                # Find dependent files using stored metadata
                dependent_files = set()
                
                for collection in search_collections:
                    if "_code" in collection:  # Only process code collections
                        resolver = DependencyResolver(qdrant_client, collection)
                        resolver.load_dependencies_from_collection()
                        
                        # Find dependencies for this collection
                        collection_deps = resolver.find_dependencies_for_files(result_files)
                        dependent_files.update(collection_deps)
                
                # Fetch dependent files from Qdrant
                if dependent_files:
                    for collection in search_collections:
                        for dep_file in dependent_files:
                            # Search for the first chunk of each dependent file
                            filter_conditions = {
                                "must": [
                                    {"key": "file_path", "match": {"value": dep_file}},
                                    {"key": "chunk_index", "match": {"value": 0}}
                                ]
                            }
                            
                            dep_results = qdrant_client.search(
                                collection_name=collection,
                                query_vector=query_embedding,
                                query_filter=filter_conditions,
                                limit=1
                            )
                            
                            if dep_results:
                                result = dep_results[0]
                                payload = result.payload
                                # Mark as dependency result with lower score
                                all_results.append({
                                    "score": result.score * 0.7,  # Reduce score for dependencies
                                    "type": "code" if "_code" in collection else "config",
                                    "collection": collection,
                                    "is_dependency": True,
                                    "dependency_type": "related",
                                    **payload
                                })
                                
            except Exception as e:
                logger.warning(f"Failed to include dependencies: {e}")
        
        # Apply enhanced ranking
        if all_results and search_mode == "hybrid":
            # Build dependency graph for ranking
            dependency_graph = {}
            if include_dependencies:
                try:
                    from utils.dependency_resolver import DependencyResolver
                    for collection in search_collections:
                        if "_code" in collection:
                            resolver = DependencyResolver(qdrant_client, collection)
                            resolver.load_dependencies_from_collection()
                            # Add to graph
                            for file_path, deps in resolver.dependency_graph.items():
                                dependency_graph[file_path] = deps
                except:
                    pass
            
            # Get query context (e.g., current file if available)
            query_context = {}
            current_project = get_current_project()
            if current_project and "root_path" in current_project:
                # Could add current file context here if available
                pass
            
            # Apply enhanced ranking
            config = get_config()
            ranking_config = config.get_section("search").get("enhanced_ranking", {})
            enhanced_ranker = get_enhanced_ranker(ranking_config)
            all_results = enhanced_ranker.rank_results(
                results=all_results,
                query_context=query_context,
                dependency_graph=dependency_graph
            )
            
            # Use enhanced score as primary score
            for result in all_results:
                if "enhanced_score" in result:
                    result["score"] = result["enhanced_score"]
        else:
            # For non-hybrid modes, just sort by existing score
            all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested number of results
        all_results = all_results[:n_results]
        
        # Expand context if requested
        if include_context and all_results:
            all_results = _expand_search_context(all_results, qdrant_client, search_collections, context_chunks)
        
        # Truncate content in results to prevent token limit issues
        for result in all_results:
            if "content" in result:
                result["content"] = _truncate_content(result["content"], max_length=1500)
            if "expanded_content" in result:
                result["expanded_content"] = _truncate_content(result["expanded_content"], max_length=2000)
        
        # Get context info
        current_project = get_current_project()
        
        # Track the search in context
        tracker = get_context_tracker()
        tracker.track_search(query, all_results, search_type="general")
        
        duration_ms = (time.time() - start_time) * 1000
        result = {
            "results": all_results,
            "query": query,
            "total": len(all_results),
            "project_context": current_project["name"] if current_project else "no project",
            "search_scope": "all projects" if cross_project else "current project",
            "search_mode": search_mode,
            "collections_searched": search_collections
        }
        
        console_logger.info(f"Completed search: {query[:50]}...", extra={
            "operation": "search",
            "query_length": len(query),
            "duration_ms": duration_ms,
            "results_found": len(all_results),
            "collections_searched": len(search_collections),
            "search_mode": search_mode,
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Failed search: {query[:50]}... - {str(e)}", extra={
            "operation": "search",
            "query_length": len(query),
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "query": query}

@mcp.tool()
def search_code(
    query: str, 
    language: Optional[str] = None, 
    n_results: int = 5, 
    cross_project: bool = False, 
    search_mode: str = "hybrid", 
    include_dependencies: bool = False, 
    include_context: bool = True, 
    context_chunks: int = 1,
    # New progressive context parameters
    context_level: str = "auto",
    progressive_mode: Optional[bool] = None,
    include_expansion_options: bool = True,
    semantic_cache: bool = True
) -> Dict[str, Any]:
    """
    Search specifically in code files (defaults to current project)
    
    Args:
        query: Search query
        language: Filter by programming language
        n_results: Number of results
        cross_project: If True, search across all projects
        search_mode: Search mode - "vector", "keyword", or "hybrid" (default: "hybrid")
        include_dependencies: If True, include files that import/are imported by the results
        include_context: If True, include surrounding chunks for more context (default: True)
        context_chunks: Number of chunks before/after to include (default: 1, max: 3)
        context_level: Granularity level ("auto", "file", "class", "method", "full") - new in v0.3.2
        progressive_mode: Enable progressive features (None = auto-detect) - new in v0.3.2
        include_expansion_options: Include drill-down options - new in v0.3.2
        semantic_cache: Use semantic similarity caching - new in v0.3.2
    """
    logger = get_logger()
    start_time = time.time()
    
    # Check if progressive context is enabled
    config = get_config()
    progressive_enabled = config.get("progressive_context", {}).get("enabled", False)
    
    if progressive_mode is None:
        # Auto-detect based on context_level
        progressive_mode = progressive_enabled and context_level != "full"
    
    # If progressive mode is requested and enabled, use progressive context manager
    if progressive_mode and progressive_enabled:
        try:
            # Import here to avoid circular imports
            from utils.progressive_context import get_progressive_manager
            
            # Get services
            embedding_model = get_embedding_model()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embedding_model,
                config.get("progressive_context", {})
            )
            
            # Use progressive context for code search
            progressive_result = progressive_manager.get_progressive_context(
                query=query,
                level=context_level,
                n_results=n_results,
                cross_project=cross_project,
                search_mode=search_mode,
                include_dependencies=include_dependencies,
                semantic_cache=semantic_cache
            )
            
            # Filter results by language if specified
            if language:
                progressive_result.results = [
                    r for r in progressive_result.results 
                    if r.get("language", "").lower() == language.lower()
                ]
            
            # Convert to standard response format with progressive metadata
            response = {
                "results": progressive_result.results,
                "query": query,
                "language_filter": language,
                "total": len(progressive_result.results),
                "project_context": get_current_project()["name"] if get_current_project() else None,
                "search_scope": "all projects" if cross_project else "current project"
            }
            
            # Add progressive metadata
            if include_expansion_options or progressive_result.level != "full":
                response["progressive"] = {
                    "level_used": progressive_result.level,
                    "token_estimate": progressive_result.token_estimate,
                    "token_reduction": progressive_result.token_reduction_percent,
                    "expansion_options": [
                        {
                            "type": opt.target_level,
                            "path": opt.target_path,
                            "estimated_tokens": opt.estimated_tokens,
                            "relevance": opt.relevance_score
                        }
                        for opt in progressive_result.expansion_options
                    ] if include_expansion_options else [],
                    "cache_hit": progressive_result.from_cache,
                    "query_intent": {
                        "type": progressive_result.query_intent.exploration_type,
                        "confidence": progressive_result.query_intent.confidence
                    } if progressive_result.query_intent else None
                }
            
            # Track in context tracking
            tracker = get_context_tracker()
            if tracker:
                tracker.track_search(query, response["results"], search_type="code_progressive")
            
            # Log completion
            console_logger.info(f"Progressive code search completed", extra={
                "operation": "search_code_progressive",
                "duration": time.time() - start_time,
                "results_count": len(response["results"]),
                "level": progressive_result.level,
                "cache_hit": progressive_result.from_cache,
                "language_filter": language
            })
            
            return response
            
        except Exception as e:
            # Log error but fall back to regular search
            console_logger.warning(f"Progressive code search failed, falling back to regular search: {e}", extra={
                "operation": "search_code_progressive_fallback",
                "error": str(e)
            })
            # Continue with regular search below
    
    # Regular search implementation (existing code)
    try:
        embedding_model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
        # Generate query embedding
        query_embedding = embedding_model.encode(query).tolist()
        
        # Determine collections
        all_collections = [c.name for c in qdrant_client.get_collections().collections]
        
        if cross_project:
            search_collections = [c for c in all_collections if "_code" in c]
        else:
            current_project = get_current_project()
            if current_project:
                search_collections = [
                    c for c in all_collections 
                    if c.startswith(current_project['collection_prefix']) and "_code" in c
                ]
            else:
                search_collections = ["global_code"]
        
        # Build filter if language specified
        filter_dict = None
        if language:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            filter_dict = Filter(
                must=[FieldCondition(key="language", match=MatchValue(value=language))]
            )
        
        # Search across collections
        all_results = []
        
        for collection in search_collections:
            try:
                results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    query_filter=filter_dict,
                    limit=n_results
                )
                
                for result in results:
                    payload = result.payload
                    all_results.append({
                        "score": result.score,
                        "file_path": payload.get("file_path", ""),  # Keep actual file_path for internal use
                        "display_path": payload.get("display_path", payload.get("file_path", "")),  # Add display_path separately
                        "language": payload.get("language", ""),
                        "line_range": {
                            "start": payload.get("line_start", 0),
                            "end": payload.get("line_end", 0)
                        },
                        "content": payload.get("content", ""),
                        "chunk_type": payload.get("chunk_type", "general"),
                        "project": payload.get("project", "unknown"),
                        "chunk_index": payload.get("chunk_index", 0),
                        "collection": collection
                    })
            except:
                pass
        
        # Handle dependency inclusion if requested
        if include_dependencies and all_results:
            try:
                from utils.dependency_resolver import DependencyResolver
                
                # Collect unique file paths from results
                result_files = set()
                for result in all_results:
                    file_path = result.get("file_path", "")
                    if file_path:
                        result_files.add(file_path)
                
                # Find dependent files using stored metadata
                dependent_files = set()
                
                for collection in search_collections:
                    resolver = DependencyResolver(qdrant_client, collection)
                    resolver.load_dependencies_from_collection()
                    
                    # Find dependencies for this collection
                    collection_deps = resolver.find_dependencies_for_files(result_files)
                    dependent_files.update(collection_deps)
                
                # Fetch dependent files from Qdrant
                if dependent_files:
                    for collection in search_collections:
                        for dep_file in dependent_files:
                            # Search for the first chunk of each dependent file
                            dep_filter = {
                                "must": [
                                    {"key": "file_path", "match": {"value": dep_file}},
                                    {"key": "chunk_index", "match": {"value": 0}}
                                ]
                            }
                            
                            if language:
                                dep_filter["must"].append({"key": "language", "match": {"value": language}})
                            
                            dep_results = qdrant_client.search(
                                collection_name=collection,
                                query_vector=query_embedding,
                                query_filter=dep_filter,
                                limit=1
                            )
                            
                            if dep_results:
                                result = dep_results[0]
                                payload = result.payload
                                # Mark as dependency result with lower score
                                all_results.append({
                                    "score": result.score * 0.7,  # Reduce score for dependencies
                                    "file_path": payload.get("display_path", payload.get("file_path", "")),
                                    "language": payload.get("language", ""),
                                    "line_range": {
                                        "start": payload.get("line_start", 0),
                                        "end": payload.get("line_end", 0)
                                    },
                                    "content": payload.get("content", ""),
                                    "chunk_type": payload.get("chunk_type", "general"),
                                    "project": payload.get("project", "unknown"),
                                    "is_dependency": True,
                                    "dependency_type": "related"
                                })
                                
            except Exception as e:
                logger.warning(f"Failed to include dependencies: {e}")
        
        # Apply enhanced ranking for hybrid search
        if all_results and search_mode == "hybrid":
            # Build dependency graph for ranking
            dependency_graph = {}
            if include_dependencies:
                try:
                    from utils.dependency_resolver import DependencyResolver
                    for collection in search_collections:
                        if "_code" in collection:
                            resolver = DependencyResolver(qdrant_client, collection)
                            resolver.load_dependencies_from_collection()
                            # Add to graph
                            for file_path, deps in resolver.dependency_graph.items():
                                dependency_graph[file_path] = deps
                except:
                    pass
            
            # Get query context
            query_context = {}
            current_project = get_current_project()
            if current_project and "root_path" in current_project:
                # Could add current file context here if available
                pass
            
            # Apply enhanced ranking
            config = get_config()
            ranking_config = config.get_section("search").get("enhanced_ranking", {})
            enhanced_ranker = get_enhanced_ranker(ranking_config)
            all_results = enhanced_ranker.rank_results(
                results=all_results,
                query_context=query_context,
                dependency_graph=dependency_graph
            )
            
            # Use enhanced score as primary score
            for result in all_results:
                if "enhanced_score" in result:
                    result["score"] = result["enhanced_score"]
        else:
            # For non-hybrid modes, just sort by existing score
            all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to requested number of results
        all_results = all_results[:n_results]
        
        # Expand context if requested
        if include_context and all_results:
            # Convert search_code results to standard format for context expansion
            formatted_results = []
            for result in all_results:
                formatted_result = {
                    "score": result["score"],
                    "file_path": result.get("file_path", ""),
                    "chunk_index": result.get("chunk_index", 0),
                    "collection": result.get("collection", ""),
                    "line_start": result.get("line_range", {}).get("start", 0),
                    "line_end": result.get("line_range", {}).get("end", 0),
                    "content": result.get("content", ""),
                    **result  # Include all other fields
                }
                formatted_results.append(formatted_result)
            
            # Expand context
            expanded_results = _expand_search_context(formatted_results, qdrant_client, search_collections, context_chunks)
            
            # Convert back to search_code format
            all_results = []
            for expanded in expanded_results:
                result = {
                    "score": expanded["score"],
                    "file_path": expanded.get("display_path", expanded.get("file_path", "")),  # Use display_path for user output
                    "language": expanded.get("language", ""),
                    "line_range": {
                        "start": expanded.get("line_start", 0),
                        "end": expanded.get("line_end", 0)
                    },
                    "content": expanded.get("content", ""),
                    "chunk_type": expanded.get("chunk_type", "general"),
                    "project": expanded.get("project", "unknown")
                }
                
                # Add expanded content if available
                if "expanded_content" in expanded:
                    result["expanded_content"] = expanded["expanded_content"]
                    result["has_context"] = expanded.get("has_context", False)
                    result["total_line_range"] = expanded.get("total_line_range", result["line_range"])
                
                # Preserve dependency info
                if expanded.get("is_dependency"):
                    result["is_dependency"] = True
                    result["dependency_type"] = expanded.get("dependency_type", "related")
                
                all_results.append(result)
        
        # Truncate content in results to prevent token limit issues
        for result in all_results:
            if "content" in result:
                result["content"] = _truncate_content(result["content"], max_length=1500)
            if "expanded_content" in result:
                result["expanded_content"] = _truncate_content(result["expanded_content"], max_length=2000)
        
        current_project = get_current_project()
        
        # Track the search in context
        tracker = get_context_tracker()
        tracker.track_search(query, all_results, search_type="code")
        
        return {
            "results": all_results,
            "query": query,
            "language_filter": language,
            "total": len(all_results),
            "project_context": current_project["name"] if current_project else "no project",
            "search_scope": "all projects" if cross_project else "current project"
        }
        
    except Exception as e:
        return {"error": str(e), "query": query}

@mcp.tool()
def search_docs(
    query: str, 
    doc_type: Optional[str] = None, 
    n_results: int = 5, 
    cross_project: bool = False, 
    search_mode: str = "hybrid", 
    include_context: bool = True, 
    context_chunks: int = 1,
    # New progressive context parameters
    context_level: str = "auto",
    progressive_mode: Optional[bool] = None,
    include_expansion_options: bool = True,
    semantic_cache: bool = True
) -> Dict[str, Any]:
    """
    Search specifically in documentation files
    
    Args:
        query: Search query
        doc_type: Filter by documentation type (e.g., 'markdown', 'rst')
        n_results: Number of results
        cross_project: If True, search across all projects
        search_mode: Search mode - "vector", "keyword", or "hybrid" (default: "hybrid")
        include_context: If True, include surrounding chunks for more context
        context_chunks: Number of chunks before/after to include (default: 1, max: 3)
        context_level: Granularity level ("auto", "file", "class", "method", "full") - new in v0.3.2
        progressive_mode: Enable progressive features (None = auto-detect) - new in v0.3.2
        include_expansion_options: Include drill-down options - new in v0.3.2
        semantic_cache: Use semantic similarity caching - new in v0.3.2
    """
    logger = get_logger()
    start_time = time.time()
    
    # Check if progressive context is enabled
    config = get_config()
    progressive_enabled = config.get("progressive_context", {}).get("enabled", False)
    
    if progressive_mode is None:
        # Auto-detect based on context_level
        progressive_mode = progressive_enabled and context_level != "full"
    
    # If progressive mode is requested and enabled, use progressive context manager
    if progressive_mode and progressive_enabled:
        try:
            # Import here to avoid circular imports
            from utils.progressive_context import get_progressive_manager
            
            # Get services
            embedding_model = get_embedding_model()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embedding_model,
                config.get("progressive_context", {})
            )
            
            # Use progressive context for documentation search
            progressive_result = progressive_manager.get_progressive_context(
                query=query,
                level=context_level,
                n_results=n_results,
                cross_project=cross_project,
                search_mode=search_mode,
                include_dependencies=False,  # Not applicable for docs
                semantic_cache=semantic_cache
            )
            
            # Filter results by doc type if specified
            if doc_type:
                progressive_result.results = [
                    r for r in progressive_result.results 
                    if r.get("doc_type", "").lower() == doc_type.lower() or r.get("file_path", "").endswith(f".{doc_type}")
                ]
            
            # Convert to standard response format with progressive metadata
            response = {
                "results": progressive_result.results,
                "query": query,
                "doc_type_filter": doc_type,
                "total": len(progressive_result.results),
                "search_mode": search_mode,
                "project_context": get_current_project()["name"] if get_current_project() else None,
                "search_scope": "all projects" if cross_project else "current project"
            }
            
            # Add progressive metadata
            if include_expansion_options or progressive_result.level != "full":
                response["progressive"] = {
                    "level_used": progressive_result.level,
                    "token_estimate": progressive_result.token_estimate,
                    "token_reduction": progressive_result.token_reduction_percent,
                    "expansion_options": [
                        {
                            "type": opt.target_level,
                            "path": opt.target_path,
                            "estimated_tokens": opt.estimated_tokens,
                            "relevance": opt.relevance_score
                        }
                        for opt in progressive_result.expansion_options
                    ] if include_expansion_options else [],
                    "cache_hit": progressive_result.from_cache,
                    "query_intent": {
                        "type": progressive_result.query_intent.exploration_type,
                        "confidence": progressive_result.query_intent.confidence
                    } if progressive_result.query_intent else None
                }
            
            # Track in context tracking
            tracker = get_context_tracker()
            if tracker:
                tracker.track_search(query, response["results"], search_type="docs_progressive")
            
            # Log completion
            console_logger.info(f"Progressive documentation search completed", extra={
                "operation": "search_docs_progressive",
                "duration": time.time() - start_time,
                "results_count": len(response["results"]),
                "level": progressive_result.level,
                "cache_hit": progressive_result.from_cache,
                "doc_type_filter": doc_type
            })
            
            return response
            
        except Exception as e:
            # Log error but fall back to regular search
            console_logger.warning(f"Progressive docs search failed, falling back to regular search: {e}", extra={
                "operation": "search_docs_progressive_fallback",
                "error": str(e)
            })
            # Continue with regular search below
    
    # Input validation
    if not query or not isinstance(query, str):
        return {
            "error": "Invalid query",
            "error_code": "INVALID_INPUT",
            "details": "Query must be a non-empty string"
        }
    
    # Sanitize query
    query = query.strip()
    if len(query) > 1000:
        query = query[:1000]
    
    n_results = max(1, min(50, n_results))
    search_mode = search_mode.lower()
    if search_mode not in ["vector", "keyword", "hybrid"]:
        search_mode = "hybrid"
    
    try:
        qdrant_client = get_qdrant_client()
        model = get_embedding_model()
        
        # Get current project info
        current_project = get_current_project()
        
        # Determine collections to search
        search_collections = []
        if cross_project:
            # Search all documentation collections
            all_collections = [c.name for c in qdrant_client.get_collections().collections]
            search_collections = [c for c in all_collections if c.endswith("_documentation")]
        else:
            # Search only current project documentation
            if current_project:
                project_collection = f"{current_project['collection_prefix']}_documentation"
                # Check if collection exists
                existing = [c.name for c in qdrant_client.get_collections().collections]
                if project_collection in existing:
                    search_collections = [project_collection]
        
        if not search_collections:
            return {
                "results": [],
                "query": query,
                "message": "No documentation collections found to search"
            }
        
        # Generate query embedding for vector search
        query_embedding = model.encode(query).tolist()
        
        all_results = []
        
        if search_mode in ["vector", "hybrid"]:
            # Vector search
            for collection in search_collections:
                try:
                    # Build search filter
                    search_filter = None
                    if doc_type:
                        search_filter = {
                            "must": [
                                {"key": "doc_type", "match": {"value": doc_type.lower()}}
                            ]
                        }
                    
                    results = qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        query_filter=search_filter,
                        limit=n_results if not search_mode == "hybrid" else n_results * 2
                    )
                    
                    for result in results:
                        payload = result.payload
                        all_results.append({
                            "score": result.score,
                            "vector_score": result.score,
                            "file_path": payload.get("file_path", ""),
                            "file_name": payload.get("file_name", ""),
                            "doc_type": payload.get("doc_type", "markdown"),
                            "title": payload.get("title", ""),
                            "heading": payload.get("heading", ""),
                            "heading_hierarchy": payload.get("heading_hierarchy", []),
                            "heading_level": payload.get("heading_level", 0),
                            "content": _truncate_content(payload.get("content", "")),
                            "chunk_index": payload.get("chunk_index", 0),
                            "chunk_type": payload.get("chunk_type", "section"),
                            "has_code_blocks": payload.get("has_code_blocks", False),
                            "code_languages": payload.get("code_languages", []),
                            "collection": collection,
                            "project": payload.get("project", "unknown")
                        })
                except Exception as e:
                    logger.warning(f"Error searching collection {collection}: {e}")
        
        if search_mode in ["keyword", "hybrid"]:
            # BM25 keyword search
            hybrid_searcher = get_hybrid_searcher()
            
            for collection in search_collections:
                try:
                    bm25_results = hybrid_searcher.bm25_manager.search(
                        collection_name=collection,
                        query=query,
                        k=n_results if not search_mode == "hybrid" else n_results * 2
                    )
                    
                    for doc in bm25_results:
                        # Check if already in results (for hybrid mode)
                        existing = next((r for r in all_results if r["file_path"] == doc["file_path"] and r["chunk_index"] == doc["chunk_index"]), None)
                        
                        if existing and search_mode == "hybrid":
                            # Update with BM25 score
                            existing["bm25_score"] = doc.get("score", 0.5)
                            existing["combined_score"] = (existing["vector_score"] + doc.get("score", 0.5)) / 2
                        elif not existing:
                            # Add new result
                            all_results.append({
                                "score": doc.get("score", 0.5),
                                "bm25_score": doc.get("score", 0.5),
                                "file_path": doc.get("file_path", ""),
                                "file_name": Path(doc.get("file_path", "")).name,
                                "doc_type": doc.get("doc_type", "markdown"),
                                "title": doc.get("title", ""),
                                "heading": doc.get("heading", ""),
                                "heading_hierarchy": doc.get("heading_hierarchy", []),
                                "heading_level": doc.get("heading_level", 0),
                                "content": _truncate_content(doc.get("content", "")),
                                "chunk_index": doc.get("chunk_index", 0),
                                "chunk_type": doc.get("chunk_type", "section"),
                                "has_code_blocks": doc.get("has_code_blocks", False),
                                "code_languages": doc.get("code_languages", []),
                                "collection": collection,
                                "project": doc.get("project", "unknown")
                            })
                except Exception as e:
                    logger.warning(f"BM25 search error for collection {collection}: {e}")
        
        # Apply enhanced ranking for hybrid search
        if all_results and search_mode == "hybrid":
            ranker = get_enhanced_ranker()
            # Convert to format expected by ranker
            ranker_results = []
            for r in all_results:
                ranker_results.append({
                    "score": r.get("combined_score", r["score"]),
                    "vector_score": r.get("vector_score", r["score"]),
                    "bm25_score": r.get("bm25_score", 0),
                    "file_path": r["file_path"],
                    "chunk_index": r["chunk_index"],
                    "content": r["content"],
                    "heading": r.get("heading", ""),
                    "heading_level": r.get("heading_level", 0)
                })
            
            # Apply ranking with documentation-specific weights from config
            config = get_config()
            doc_ranking_config = config.get_section("search").get("documentation_ranking", {
                "base_score_weight": 0.4,
                "file_proximity_weight": 0.2,
                "dependency_distance_weight": 0.0,
                "code_structure_weight": 0.0,
                "recency_weight": 0.4
            })
            
            # Convert config format to ranker format
            doc_weights = {
                "base_score": doc_ranking_config.get("base_score_weight", 0.4),
                "file_proximity": doc_ranking_config.get("file_proximity_weight", 0.2),
                "dependency_distance": doc_ranking_config.get("dependency_distance_weight", 0.0),
                "code_structure": doc_ranking_config.get("code_structure_weight", 0.0),
                "recency": doc_ranking_config.get("recency_weight", 0.4)
            }
            
            # Store current weights and update for documentation search
            original_weights = dict(ranker.weights)
            ranker.update_weights(doc_weights)
            
            ranked_results = ranker.rank_results(
                results=ranker_results,
                query_context=current_project
            )
            
            # Restore original weights
            ranker.update_weights(original_weights)
            
            # Update scores and add ranking signals
            for i, ranked in enumerate(ranked_results):
                if i < len(all_results):
                    matching_result = next((r for r in all_results if r["file_path"] == ranked["file_path"] and r["chunk_index"] == ranked["chunk_index"]), None)
                    if matching_result:
                        matching_result["score"] = ranked.get("enhanced_score", ranked.get("score", 0))
                        matching_result["ranking_signals"] = ranked.get("ranking_signals", {})
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results
        all_results = all_results[:n_results]
        
        # Expand context if requested
        if include_context and all_results:
            # Convert to format expected by context expander
            formatted_results = []
            for result in all_results:
                formatted_result = {
                    "score": result["score"],
                    "file_path": result["file_path"],
                    "chunk_index": result["chunk_index"],
                    "collection": result["collection"],
                    "content": result["content"],
                    **result
                }
                formatted_results.append(formatted_result)
            
            # Expand context
            expanded_results = _expand_search_context(formatted_results, qdrant_client, search_collections, context_chunks)
            
            # Update results with expanded content
            for i, expanded in enumerate(expanded_results):
                if i < len(all_results):
                    if expanded.get("expanded_content"):
                        all_results[i]["content"] = _truncate_content(expanded["expanded_content"], 2000)
                        all_results[i]["has_context"] = True
                        all_results[i]["context_chunks_before"] = expanded.get("context_chunks_before", 0)
                        all_results[i]["context_chunks_after"] = expanded.get("context_chunks_after", 0)
        
        duration_ms = (time.time() - start_time) * 1000
        
        console_logger.info(f"Documentation search completed", extra={
            "operation": "search_docs",
            "query": query,
            "results": len(all_results),
            "collections_searched": len(search_collections),
            "search_mode": search_mode,
            "duration_ms": duration_ms,
            "status": "success"
        })
        
        # Track the search in context
        tracker = get_context_tracker()
        tracker.track_search(query, all_results, search_type="documentation")
        
        return {
            "results": all_results,
            "query": query,
            "doc_type_filter": doc_type,
            "total": len(all_results),
            "search_mode": search_mode,
            "project_context": current_project["name"] if current_project else "no project",
            "search_scope": "all projects" if cross_project else "current project"
        }
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Documentation search failed", extra={
            "operation": "search_docs",
            "query": query,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {
            "results": [],
            "query": query,
            "error": str(e),
            "error_code": "SEARCH_ERROR",
            "total": 0,
            "search_mode": search_mode,
            "message": f"Documentation search failed: {str(e)}"
        }

@mcp.tool()
def switch_project(project_path: str) -> Dict[str, Any]:
    """Switch to a different project context"""
    try:
        path = Path(project_path).resolve()
        if not path.exists():
            return {"error": f"Path not found: {project_path}"}
        
        # Change working directory
        os.chdir(path)
        
        # Reset current project cache
        global _current_project
        _current_project = None
        
        # Get new project info
        new_project = get_current_project()
        
        if new_project:
            return {
                "switched_to": new_project["name"],
                "project_root": new_project["root"],
                "message": f"Switched to project: {new_project['name']}"
            }
        else:
            return {
                "switched_to": str(path),
                "message": f"Switched to directory: {path}"
            }
            
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def index_config(file_path: str, force_global: bool = False) -> Dict[str, Any]:
    """Index a configuration file"""
    try:
        from qdrant_client.http.models import PointStruct
        
        # Resolve to absolute path
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Get file modification time and hash
        try:
            mod_time = abs_path.stat().st_mtime
            file_hash = calculate_file_hash(str(abs_path))
        except:
            mod_time = time.time()  # Use current time as fallback
            file_hash = None  # No hash if file reading fails
        
        # Get services
        config_indexer = get_config_indexer()
        embedding_model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
        # Determine collection
        if force_global:
            collection_name = "global_config"
        else:
            collection_name = get_collection_name(str(abs_path), "config")
        ensure_collection(collection_name)
        
        # Index file
        chunks = config_indexer.index_file(str(abs_path))
        
        # Convert to points
        points = []
        current_project = get_current_project()
        
        # Calculate relative path if in current project
        display_path = str(abs_path)
        if current_project:
            try:
                rel_path = abs_path.relative_to(Path(current_project["root"]))
                display_path = str(rel_path)
            except ValueError:
                pass
        
        for chunk in chunks:
            embedding = embedding_model.encode(chunk.content).tolist()
            
            # Generate unique chunk ID
            chunk_id = hashlib.md5(f"{abs_path}_{chunk.chunk_index}".encode()).hexdigest()
            
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "file_path": str(abs_path),
                    "display_path": display_path,
                    "chunk_index": chunk.chunk_index,
                    "file_type": chunk.metadata.get("file_type", ""),
                    "path": chunk.metadata.get("path", ""),
                    "content": chunk.content,
                    "value": chunk.metadata.get("value", ""),
                    "project": collection_name.rsplit('_', 1)[0],
                    "modified_at": mod_time,
                    "file_hash": file_hash
                }
            )
            points.append(point)
        
        # Store in Qdrant
        if points:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            # Update BM25 index
            hybrid_searcher = get_hybrid_searcher()
            documents = []
            for chunk in chunks:
                doc = {
                    "id": f"{str(abs_path)}_{chunk.chunk_index}",
                    "content": chunk.content,
                    "file_path": str(abs_path),
                    "chunk_index": chunk.chunk_index,
                    "file_type": chunk.metadata.get("file_type", ""),
                    "path": chunk.metadata.get("path", ""),
                    "value": chunk.metadata.get("value", "")
                }
                documents.append(doc)
            
            # Get all documents in collection for BM25 update
            all_docs = []
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            for point in scroll_result[0]:
                doc = {
                    "id": f"{point.payload['file_path']}_{point.payload['chunk_index']}",
                    "content": point.payload.get("content", ""),
                    "file_path": point.payload.get("file_path", ""),
                    "chunk_index": point.payload.get("chunk_index", 0),
                    "file_type": point.payload.get("file_type", ""),
                    "path": point.payload.get("path", ""),
                    "value": point.payload.get("value", "")
                }
                all_docs.append(doc)
                
            hybrid_searcher.bm25_manager.update_index(collection_name, all_docs)
        
        return {
            "indexed": len(chunks),
            "file_path": display_path,
            "collection": collection_name,
            "file_type": chunks[0].metadata.get("file_type", "unknown") if chunks else "unknown",
            "project_context": current_project["name"] if current_project else "global"
        }
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    Check the health status of all services.
    
    Returns status of:
    - Qdrant connection
    - Embedding model
    - Disk space
    - Memory usage
    - Current project context
    """
    logger = get_logger()
    console_logger.info("Running health check")
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": __version__,
        "services": {},
        "system": {},
        "project": None
    }
    
    # Check Qdrant connection
    try:
        client = get_qdrant_client()
        collections = client.get_collections()
        health_status["services"]["qdrant"] = {
            "status": "healthy",
            "collections_count": len(collections.collections),
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333"))
        }
    except Exception as e:
        health_status["services"]["qdrant"] = {
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__
        }
        health_status["status"] = "unhealthy"
    
    # Check embedding model
    try:
        model = get_embedding_model()
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        # Test with a simple embedding
        test_embedding = model.encode("test")
        health_status["services"]["embedding_model"] = {
            "status": "healthy",
            "model": model_name,
            "embedding_dim": len(test_embedding)
        }
    except Exception as e:
        health_status["services"]["embedding_model"] = {
            "status": "unhealthy",
            "error": str(e),
            "error_type": type(e).__name__
        }
        health_status["status"] = "unhealthy"
    
    # Check disk space
    try:
        import shutil
        disk_usage = shutil.disk_usage("/")
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        used_percent = (disk_usage.used / disk_usage.total) * 100
        
        health_status["system"]["disk"] = {
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "used_percent": round(used_percent, 1),
            "status": "healthy" if free_gb > 1 else "warning"
        }
        
        if free_gb < 1:
            health_status["status"] = "warning"
    except Exception as e:
        health_status["system"]["disk"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        health_status["system"]["memory"] = {
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "used_percent": memory.percent,
            "status": "healthy" if memory.percent < 90 else "warning"
        }
        
        if memory.percent > 90:
            health_status["status"] = "warning"
    except ImportError:
        health_status["system"]["memory"] = {
            "status": "unknown",
            "note": "psutil not installed"
        }
    except Exception as e:
        health_status["system"]["memory"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    # Check current project context
    try:
        # Use client's working directory if available
        client_cwd = os.environ.get('MCP_CLIENT_CWD')
        project = get_current_project(client_directory=client_cwd)
        if project:
            health_status["project"] = {
                "name": project["name"],
                "path": project["root"],
                "collection_prefix": project["collection_prefix"]
            }
            # Add note if using client CWD
            if client_cwd:
                health_status["project"]["client_cwd"] = client_cwd
    except Exception as e:
        health_status["project"] = {
            "error": str(e)
        }
    
    # Log health check result
    console_logger.info(
        f"Health check completed: {health_status['status']}",
        extra={
            "operation": "health_check",
            "status": health_status["status"],
            "qdrant_status": health_status["services"].get("qdrant", {}).get("status", "unknown"),
            "model_status": health_status["services"].get("embedding_model", {}).get("status", "unknown")
        }
    )
    
    return health_status


# GitHub Integration - Import modules with graceful fallback
try:
    from github_integration.client import get_github_client
    from github_integration.issue_analyzer import get_issue_analyzer
    from github_integration.code_generator import get_code_generator
    from github_integration.workflows import get_github_workflows
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    console_logger.warning("GitHub integration not available. Install with: pip install PyGithub GitPython")


# Global GitHub instances (lazy initialization)
_github_client = None
_issue_analyzer = None
_code_generator = None
_github_workflows = None


def get_github_instances():
    """Get or create GitHub service instances."""
    global _github_client, _issue_analyzer, _code_generator, _github_workflows
    
    if not GITHUB_AVAILABLE:
        raise ImportError("GitHub integration not available. Install with: pip install PyGithub GitPython")
    
    if _github_client is None:
        _github_client = get_github_client()
    
    # Always get the latest config and update issue analyzer
    search_functions = {
        "search": search,
        "search_code": search_code,
        "search_docs": search_docs
    }
    # Get GitHub config from global config
    config = get_config()
    github_config = config.get("github", {})
    
    if _issue_analyzer is None:
        _issue_analyzer = get_issue_analyzer(_github_client, search_functions, github_config)
    else:
        # Update configuration on existing instance
        _issue_analyzer = get_issue_analyzer(_github_client, search_functions, github_config)
    
    if _code_generator is None:
        _code_generator = get_code_generator()
    
    if _github_workflows is None:
        _github_workflows = get_github_workflows(_github_client, _issue_analyzer, _code_generator, github_config)
    else:
        # Update configuration on existing instance
        _github_workflows = get_github_workflows(_github_client, _issue_analyzer, _code_generator, github_config)
    
    return _github_client, _issue_analyzer, _code_generator, _github_workflows


# GitHub MCP Tools
@mcp.tool()
def github_list_repositories(owner: Optional[str] = None) -> Dict[str, Any]:
    """
    List GitHub repositories for a user/organization.
    
    Args:
        owner: Repository owner (defaults to authenticated user)
        
    Returns:
        List of repository information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        repositories = github_client.list_repositories(owner)
        
        console_logger.info(f"Listed {len(repositories)} repositories for {owner or 'authenticated user'}")
        
        return {
            "repositories": repositories,
            "count": len(repositories),
            "owner": owner or "authenticated_user"
        }
        
    except Exception as e:
        error_msg = f"Failed to list repositories: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_switch_repository(owner: str, repo: str) -> Dict[str, Any]:
    """
    Switch to a different GitHub repository context.
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        Repository information and switch status
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        repository = github_client.set_repository(owner, repo)
        
        # Optional: Auto-index repository if configured
        config = get_config()
        if config.get("github", {}).get("repository", {}).get("auto_index_on_switch", True):
            try:
                # This would trigger repository indexing - placeholder for now
                console_logger.info(f"Auto-indexing enabled for {owner}/{repo} (feature not yet implemented)")
            except Exception as e:
                console_logger.warning(f"Auto-indexing failed for {owner}/{repo}: {e}")
        
        console_logger.info(
            f"Switched to repository {owner}/{repo}",
            extra={
                "operation": "github_switch_repository",
                "owner": owner,
                "repo": repo,
                "full_name": repository.full_name
            }
        )
        
        return {
            "repository": {
                "owner": owner,
                "name": repo,
                "full_name": repository.full_name,
                "description": repository.description,
                "private": repository.private,
                "language": repository.language,
                "stars": repository.stargazers_count,
                "forks": repository.forks_count
            },
            "message": f"Successfully switched to {owner}/{repo}"
        }
        
    except Exception as e:
        error_msg = f"Failed to switch repository: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_switch_repository",
                "error": str(e),
                "owner": owner,
                "repo": repo
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_fetch_issues(state: str = "open", labels: Optional[List[str]] = None, 
                       limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Fetch GitHub issues from current repository.
    
    Args:
        state: Issue state (open, closed, all)
        labels: Filter by labels
        limit: Maximum number of issues
        
    Returns:
        List of issue information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        issues = github_client.get_issues(state=state, labels=labels, limit=limit)
        
        console_logger.info(
            f"Fetched {len(issues)} issues",
            extra={
                "operation": "github_fetch_issues",
                "state": state,
                "labels": labels,
                "count": len(issues)
            }
        )
        
        return {
            "issues": issues,
            "count": len(issues),
            "state": state,
            "labels": labels,
            "repository": github_client.get_current_repository().full_name
        }
        
    except Exception as e:
        error_msg = f"Failed to fetch issues: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_fetch_issues",
                "error": str(e),
                "state": state,
                "labels": labels
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_get_issue(issue_number: int) -> Dict[str, Any]:
    """
    Get detailed information about a specific GitHub issue.
    
    Args:
        issue_number: Issue number
        
    Returns:
        Detailed issue information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        issue = github_client.get_issue(issue_number)
        
        console_logger.info(
            f"Retrieved issue #{issue_number}",
            extra={
                "operation": "github_get_issue",
                "issue_number": issue_number,
                "title": issue["title"]
            }
        )
        
        return {
            "issue": issue,
            "repository": github_client.get_current_repository().full_name
        }
        
    except Exception as e:
        error_msg = f"Failed to get issue #{issue_number}: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_get_issue",
                "error": str(e),
                "issue_number": issue_number
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_create_issue(title: str, body: str = "", labels: Optional[List[str]] = None,
                       assignees: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create a new GitHub issue.
    
    Args:
        title: Issue title
        body: Issue description/body (optional)
        labels: List of label names to apply (optional)
        assignees: List of usernames to assign (optional)
        
    Returns:
        Created issue information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        issue = github_client.create_issue(title, body, labels, assignees)
        
        console_logger.info(
            f"Created issue #{issue['number']}: {title}",
            extra={
                "operation": "github_create_issue",
                "issue_number": issue["number"],
                "title": title,
                "labels": labels or [],
                "assignees": assignees or []
            }
        )
        
        return {
            "issue": issue,
            "repository": github_client.get_current_repository().full_name,
            "message": f"Successfully created issue #{issue['number']}"
        }
        
    except Exception as e:
        error_msg = f"Failed to create issue: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_create_issue",
                "error": str(e),
                "title": title
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_add_comment(issue_number: int, body: str) -> Dict[str, Any]:
    """
    Add a comment to an existing GitHub issue.
    
    Args:
        issue_number: Issue number to comment on
        body: Comment body text
        
    Returns:
        Comment information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        comment = github_client.add_comment(issue_number, body)
        
        console_logger.info(
            f"Added comment to issue #{issue_number}",
            extra={
                "operation": "github_add_comment",
                "issue_number": issue_number,
                "comment_id": comment["id"]
            }
        )
        
        return {
            "comment": comment,
            "repository": github_client.get_current_repository().full_name,
            "message": f"Successfully added comment to issue #{issue_number}"
        }
        
    except Exception as e:
        error_msg = f"Failed to add comment: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_add_comment",
                "error": str(e),
                "issue_number": issue_number
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_analyze_issue(issue_number: int) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a GitHub issue using RAG search.
    
    Args:
        issue_number: Issue number to analyze
        
    Returns:
        Analysis results with search results and recommendations
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, workflows = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        # Run analysis workflow
        result = workflows.analyze_issue_workflow(issue_number)
        
        console_logger.info(
            f"Analyzed issue #{issue_number}",
            extra={
                "operation": "github_analyze_issue",
                "issue_number": issue_number,
                "workflow_status": result.get("workflow_status"),
                "confidence": result.get("analysis", {}).get("analysis", {}).get("confidence_score", 0)
            }
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to analyze issue #{issue_number}: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_analyze_issue",
                "error": str(e),
                "issue_number": issue_number
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_suggest_fix(issue_number: int) -> Dict[str, Any]:
    """
    Generate fix suggestions for a GitHub issue using RAG analysis.
    
    Args:
        issue_number: Issue number
        
    Returns:
        Fix suggestions and implementation plan
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, workflows = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        # Run fix suggestion workflow
        result = workflows.suggest_fix_workflow(issue_number)
        
        console_logger.info(
            f"Generated fix suggestions for issue #{issue_number}",
            extra={
                "operation": "github_suggest_fix",
                "issue_number": issue_number,
                "workflow_status": result.get("workflow_status"),
                "fix_count": len(result.get("suggestions", {}).get("fixes", [])),
                "confidence": result.get("suggestions", {}).get("confidence_level", "unknown")
            }
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to generate fix suggestions for issue #{issue_number}: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_suggest_fix",
                "error": str(e),
                "issue_number": issue_number
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_create_pull_request(title: str, body: str, head: str, base: str = "main",
                              files: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Create a GitHub pull request.
    
    Args:
        title: PR title
        body: PR description
        head: Head branch
        base: Base branch (default: main)
        files: List of files to include (for reference only)
        
    Returns:
        Pull request information
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, _ = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        # Create pull request
        pr = github_client.create_pull_request(
            title=title,
            body=body,
            head=head,
            base=base,
            files=files
        )
        
        console_logger.info(
            f"Created pull request #{pr['number']}",
            extra={
                "operation": "github_create_pull_request",
                "pr_number": pr["number"],
                "title": title,
                "head": head,
                "base": base
            }
        )
        
        return {
            "pull_request": pr,
            "repository": github_client.get_current_repository().full_name,
            "message": f"Successfully created PR #{pr['number']}"
        }
        
    except Exception as e:
        error_msg = f"Failed to create pull request: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_create_pull_request",
                "error": str(e),
                "title": title,
                "head": head,
                "base": base
            }
        )
        return {"error": error_msg}


@mcp.tool()
def github_resolve_issue(issue_number: int, dry_run: bool = True) -> Dict[str, Any]:
    """
    Attempt to resolve a GitHub issue with automated analysis and PR creation.
    
    Args:
        issue_number: Issue number to resolve
        dry_run: If True, only show what would be done (default: True for safety)
        
    Returns:
        Resolution workflow results
    """
    try:
        if not GITHUB_AVAILABLE:
            return {
                "error": "GitHub integration not available",
                "message": "Install with: pip install PyGithub GitPython"
            }
        
        github_client, _, _, workflows = get_github_instances()
        
        if not github_client.get_current_repository():
            return {
                "error": "No repository context set",
                "message": "Use github_switch_repository first"
            }
        
        # Run complete resolution workflow
        result = workflows.resolve_issue_workflow(issue_number, dry_run=dry_run)
        
        console_logger.info(
            f"Issue resolution workflow for #{issue_number} (dry_run={dry_run})",
            extra={
                "operation": "github_resolve_issue",
                "issue_number": issue_number,
                "dry_run": dry_run,
                "workflow_status": result.get("workflow_status")
            }
        )
        
        return result
        
    except Exception as e:
        error_msg = f"Failed to resolve issue #{issue_number}: {str(e)}"
        console_logger.error(
            error_msg,
            extra={
                "operation": "github_resolve_issue",
                "error": str(e),
                "issue_number": issue_number,
                "dry_run": dry_run
            }
        )
        return {"error": error_msg}


# Context tracking tools
@mcp.tool()
def get_context_status() -> Dict[str, Any]:
    """
    Get current context window usage and statistics
    
    Returns information about:
    - Session ID and uptime
    - Token usage estimates
    - Files read and searches performed
    - Top resource-consuming operations
    """
    tracker = get_context_tracker()
    
    # Get summary
    summary = tracker.get_session_summary()
    
    # Get context usage warning if applicable
    usage_check = check_context_usage(tracker)
    
    # Get top files by tokens
    top_files = tracker.get_top_files_by_tokens(limit=5)
    
    return {
        "session_id": summary["session_id"],
        "uptime_minutes": summary["uptime_minutes"],
        "context_usage": {
            "estimated_tokens": summary["total_tokens_estimate"],
            "percentage_used": summary["context_usage_percentage"],
            "tokens_remaining": 50000 - summary["total_tokens_estimate"]
        },
        "activity_summary": {
            "files_read": summary["files_read"],
            "searches_performed": summary["searches_performed"],
            "indexed_directories": summary["indexed_directories"]
        },
        "token_breakdown": summary["token_usage_by_category"],
        "top_files": [{"path": path, "tokens": tokens} for path, tokens in top_files],
        "usage_status": usage_check,
        "current_project": summary["current_project"]
    }

@mcp.tool()
def get_context_timeline() -> List[Dict[str, Any]]:
    """
    Get chronological timeline of context events
    
    Returns a list of all context-consuming events in chronological order,
    including file reads, searches, and tool uses.
    """
    tracker = get_context_tracker()
    
    # Get all events
    events = []
    for event in tracker.context_events:
        events.append({
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "tokens_estimate": event.tokens_estimate,
            "details": event.details
        })
    
    return events

@mcp.tool()
def get_context_summary() -> str:
    """
    Get a natural language summary of current context
    
    Returns a human-readable summary of what Claude currently knows
    about the session, including key files, searches, and project context.
    """
    tracker = get_context_tracker()
    summary = tracker.get_session_summary()
    
    # Build natural language summary
    lines = [
        f"Current Session Context Summary:",
        f"- Session started: {summary['uptime_minutes']} minutes ago",
        f"- Current project: {summary['current_project']['name'] if summary['current_project'] else 'No project context'}",
        f"",
        f"Activity:",
        f"- Files read: {summary['files_read']} files",
        f"- Searches performed: {summary['searches_performed']} searches",
        f"- Directories indexed: {summary['indexed_directories']} directories",
        f"",
        f"Context usage: {summary['context_usage_percentage']:.1f}% of available window",
        f"- Estimated tokens used: {summary['total_tokens_estimate']:,}",
        f"- Tokens remaining: {50000 - summary['total_tokens_estimate']:,}",
    ]
    
    # Add top files if any
    if tracker.files_read:
        lines.append("")
        lines.append("Top files in context:")
        top_files = tracker.get_top_files_by_tokens(limit=3)
        for path, tokens in top_files:
            lines.append(f"  - {path} (~{tokens:,} tokens)")
    
    # Add recent searches if any
    if tracker.searches_performed:
        lines.append("")
        lines.append("Recent searches:")
        for search in tracker.searches_performed[-3:]:
            lines.append(f"  - \"{search['query']}\" ({search['search_type']}, {search['results_count']} results)")
    
    # Add usage warning if needed
    usage_check = check_context_usage(tracker)
    if usage_check.get("warning"):
        lines.append("")
        lines.append(f"⚠️ {usage_check['message']}")
        lines.append(f"   {usage_check['suggestion']}")
    
    return "\n".join(lines)


# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant RAG MCP Server")
    parser.add_argument("--watch", action="store_true", help="Enable file watching for auto-reindexing")
    parser.add_argument("--watch-dir", default=".", help="Directory to watch (default: current)")
    parser.add_argument("--debounce", type=float, default=3.0, help="Debounce seconds for file watching")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial index on startup")
    parser.add_argument("--client-cwd", help="Client's working directory (overrides MCP_CLIENT_CWD env var)")
    args = parser.parse_args()
    
    # Set client working directory if provided via command line
    if args.client_cwd:
        os.environ['MCP_CLIENT_CWD'] = args.client_cwd
        console_logger.info(f"Client working directory set to: {args.client_cwd}")
    
    # Start file watcher if requested
    observer = None
    if args.watch:
        if not WATCHDOG_AVAILABLE:
            console_logger.error("❌ Watchdog not installed. Install with: pip install watchdog")
            sys.exit(1)
            
        # Perform initial index if requested
        if args.initial_index:
            console_logger.info(f"📦 Performing initial index of {args.watch_dir}...")
            result = index_directory(args.watch_dir)
            console_logger.info(f"✅ Initial index complete: {result.get('total', 0)} files indexed")
        
        # Start file watcher
        event_handler = RagFileWatcher(debounce_seconds=args.debounce)
        observer = Observer()
        observer.schedule(event_handler, args.watch_dir, recursive=True)
        observer.start()
        console_logger.info(f"👀 File watcher started for {args.watch_dir}")
        console_logger.info(f"⏱️  Debounce: {args.debounce}s")
        
    try:
        # Run MCP server
        console_logger.info("🚀 Starting Qdrant RAG MCP Server...")
        if args.watch:
            console_logger.info("✅ Auto-reindexing enabled")
        mcp.run()
    except KeyboardInterrupt:
        console_logger.info("🛑 Shutting down...")
    finally:
        if observer:
            observer.stop()
            observer.join()
            console_logger.info("👋 File watcher stopped")