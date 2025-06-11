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
import gc
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from pathlib import Path
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import version
try:
    from . import __version__
except ImportError:
    __version__ = "0.3.4.post3"  # Fallback version

# Load environment variables from the MCP server directory
from dotenv import load_dotenv
mcp_server_dir = Path(__file__).parent.parent
env_path = mcp_server_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Ensure HuggingFace uses our custom cache directory
# This must be set before any HF imports to avoid cache mismatches
if 'SENTENCE_TRANSFORMERS_HOME' in os.environ:
    cache_dir = os.path.expanduser(os.environ['SENTENCE_TRANSFORMERS_HOME'])
    if not os.environ.get('HF_HOME'):
        os.environ['HF_HOME'] = cache_dir
    if not os.environ.get('HF_HUB_CACHE'):
        os.environ['HF_HUB_CACHE'] = cache_dir

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
from utils.progressive_context import get_progressive_manager
# Import context tracking
from utils.context_tracking import SessionContextTracker, SessionStore, check_context_usage
# Import model registry
from utils.model_registry import get_model_registry
# Import memory manager
from utils.memory_manager import get_memory_manager

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
_embeddings_manager = None  # Unified embeddings manager (v0.3.3)
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

def _expand_search_context(results: List[Dict[str, Any]], qdrant_client, search_collections: List[str], context_chunks: int = 1, embedding_dimension: int = 384) -> List[Dict[str, Any]]:
    """Expand search results with surrounding chunks for better context"""
    logger = get_logger()
    expanded_results = []
    seen_chunks = set()  # Track which chunks we've already added
    
    # Limit context chunks to reasonable amount
    context_chunks = min(context_chunks, 3)
    
    # Cache collection dimensions to avoid repeated API calls
    collection_dimensions = {}
    
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
        
        # Get the actual dimension for this collection
        if collection not in collection_dimensions:
            try:
                collection_info = qdrant_client.get_collection(collection)
                # Get dimension from vector config
                if hasattr(collection_info.config.params, 'vectors'):
                    if isinstance(collection_info.config.params.vectors, dict):
                        # Named vectors - get first one
                        first_vector_config = next(iter(collection_info.config.params.vectors.values()))
                        collection_dimensions[collection] = first_vector_config.size
                    else:
                        # Single vector config
                        collection_dimensions[collection] = collection_info.config.params.vectors.size
                else:
                    # Fallback to default
                    collection_dimensions[collection] = embedding_dimension
            except Exception as e:
                logger.debug(f"Failed to get collection dimension for {collection}: {e}")
                collection_dimensions[collection] = embedding_dimension
        
        actual_dimension = collection_dimensions[collection]
        
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
                        query_vector=[0.0] * actual_dimension,  # Dummy vector for filter-only search
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
                    query_vector=[0.0] * actual_dimension,  # Dummy vector for filter-only search
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

def ensure_collection(collection_name: str, embedding_dimension: Optional[int] = None, 
                     embedding_model_name: Optional[str] = None,
                     content_type: str = "general"):
    """Ensure a collection exists with retry logic and model metadata
    
    Args:
        collection_name: Name of the collection to create
        embedding_dimension: Dimension of the embedding vectors (auto-detected if None)
        embedding_model_name: Name of the embedding model used (auto-detected if None)
        content_type: Type of content (code, config, documentation, general)
    """
    client = get_qdrant_client()
    
    from qdrant_client.http.models import Distance, VectorParams
    from utils.model_registry import get_model_registry
    from utils.embeddings import get_embeddings_manager
    
    def check_and_create():
        existing = [c.name for c in client.get_collections().collections]
        
        if collection_name not in existing:
            config = get_config()
            embeddings_manager = get_embeddings_manager(config)
            
            # Determine model name and dimension based on manager type
            if hasattr(embeddings_manager, 'use_specialized') and embeddings_manager.use_specialized:
                # Using specialized embeddings
                actual_model_name = embeddings_manager.get_model_name(content_type)
                actual_dimension = embeddings_manager.get_dimension(content_type)
            else:
                # Using single model mode
                actual_model_name = embeddings_manager.model_name
                actual_dimension = embeddings_manager.dimension
            
            # Use provided values if specified, otherwise use detected values
            final_model_name = embedding_model_name or actual_model_name
            final_dimension = embedding_dimension or actual_dimension
            
            # Create collection with vectors config
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=final_dimension,
                    distance=Distance.COSINE
                )
            )
            
            # Store model metadata in a special point
            metadata_point_id = hashlib.md5(f"{collection_name}_metadata".encode()).hexdigest()
            
            # Create metadata payload
            metadata_payload = {
                "type": "collection_metadata",
                "embedding_model": final_model_name,
                "embedding_dimension": final_dimension,
                "content_type": content_type,
                "created_at": datetime.now().isoformat(),
                "mcp_version": __version__,
                "specialized_embeddings": hasattr(embeddings_manager, 'use_specialized') and embeddings_manager.use_specialized
            }
            
            # Store metadata as a special point with zero vector
            from qdrant_client.http.models import PointStruct
            metadata_point = PointStruct(
                id=metadata_point_id,
                vector=[0.0] * final_dimension,  # Dummy vector
                payload=metadata_payload
            )
            
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=[metadata_point]
                )
                
                # Register with model registry
                registry = get_model_registry()
                registry.register_collection(collection_name, final_model_name, content_type)
                
                logger = get_logger()
                logger.info(f"Created collection {collection_name} with model {final_model_name} "
                           f"(dimension: {final_dimension}, type: {content_type})")
                
            except Exception as e:
                logger = get_logger()
                logger.warning(f"Failed to store metadata for collection {collection_name}: {e}")
    
    # Use retry logic for collection operations
    retry_operation(check_and_create)

def get_collection_metadata(collection_name: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a collection
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        Dictionary with collection metadata or None if not found
    """
    client = get_qdrant_client()
    
    try:
        # Check if collection exists
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            return None
        
        # Try to retrieve metadata point
        metadata_point_id = hashlib.md5(f"{collection_name}_metadata".encode()).hexdigest()
        
        try:
            points = client.retrieve(
                collection_name=collection_name,
                ids=[metadata_point_id],
                with_payload=True,
                with_vectors=False
            )
            
            if points and len(points) > 0:
                return points[0].payload
            
        except Exception:
            # Metadata point doesn't exist (legacy collection)
            pass
        
        # Fall back to model registry
        from utils.model_registry import get_model_registry
        registry = get_model_registry()
        collection_info = registry.get_collection_info(collection_name)
        
        if collection_info:
            return {
                "embedding_model": collection_info['model'],
                "content_type": collection_info.get('content_type', 'general'),
                "specialized_embeddings": False  # Legacy collections use single model
            }
        
        # No metadata found - assume defaults
        return {
            "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "embedding_dimension": 384,  # Default dimension
            "content_type": "general",
            "specialized_embeddings": False
        }
        
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to retrieve metadata for collection {collection_name}: {e}")
        return None

def check_model_compatibility(collection_name: str, query_model_name: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """Check if a query model is compatible with a collection's indexed model
    
    Args:
        collection_name: Name of the collection to check
        query_model_name: Model name to use for querying (None = current default)
        
    Returns:
        Tuple of (is_compatible, recommended_model, metadata)
        - is_compatible: Whether the models are compatible
        - recommended_model: The model name that should be used
        - metadata: Collection metadata if available
    """
    # Get collection metadata
    metadata = get_collection_metadata(collection_name)
    if not metadata:
        # No metadata - assume compatible with warning
        logger = get_logger()
        logger.warning(f"No metadata found for collection {collection_name}, assuming model compatibility")
        return True, query_model_name, None
    
    collection_model = metadata.get("embedding_model")
    collection_dimension = metadata.get("embedding_dimension")
    
    # If no query model specified, use the collection's model
    if not query_model_name:
        return True, collection_model, metadata
    
    # Check if models match exactly
    if query_model_name == collection_model:
        return True, query_model_name, metadata
    
    # Check dimensions for compatibility
    registry = get_model_registry()
    
    query_dimension = registry.get_model_dimension(query_model_name)
    
    if collection_dimension and query_dimension != collection_dimension:
        logger = get_logger()
        logger.warning(f"Model dimension mismatch for collection {collection_name}: "
                      f"collection uses {collection_model} ({collection_dimension}D), "
                      f"query uses {query_model_name} ({query_dimension}D)")
        return False, collection_model, metadata
    
    # Models have same dimensions - might be compatible but warn
    logger = get_logger()
    logger.info(f"Using different but potentially compatible model for {collection_name}: "
                f"indexed with {collection_model}, querying with {query_model_name}")
    return True, query_model_name, metadata

def get_query_embedding_for_collection(query: str, collection_name: str, embeddings_manager=None) -> List[float]:
    """Generate query embedding using the appropriate model for a collection
    
    Args:
        query: Query text to embed
        collection_name: Collection to search
        embeddings_manager: Optional embeddings manager instance
        
    Returns:
        Query embedding as list of floats
    """
    if embeddings_manager is None:
        embeddings_manager = get_embeddings_manager_instance()
    
    # Get collection metadata to determine the right model
    metadata = get_collection_metadata(collection_name)
    
    if metadata and metadata.get("specialized_embeddings"):
        # Collection was indexed with specialized embeddings
        content_type = metadata.get("content_type", "general")
        embedding_model = metadata.get("embedding_model")
        
        # Check if we're using the same specialized model
        current_model = embeddings_manager.get_model_name(content_type)
        if current_model != embedding_model:
            logger = get_logger()
            logger.warning(f"Model mismatch for collection {collection_name}: "
                          f"indexed with {embedding_model}, current is {current_model}")
        
        # Use the appropriate content type
        return embeddings_manager.encode(query, content_type=content_type).tolist()
    else:
        # Legacy collection or no metadata - use general embedding
        return embeddings_manager.encode(query, content_type="general").tolist()

def get_embeddings_manager_instance():
    """Get or create the global embeddings manager instance
    
    Returns the UnifiedEmbeddingsManager which supports both single and specialized modes
    """
    global _embeddings_manager
    if _embeddings_manager is None:
        from utils.embeddings import get_embeddings_manager
        config = get_config()
        _embeddings_manager = get_embeddings_manager(config)
    return _embeddings_manager

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
    """DEPRECATED: Use get_embeddings_manager() instead
    
    This function is kept for backward compatibility but now returns
    a wrapper around the unified embeddings manager.
    """
    # Get the unified embeddings manager
    from utils.embeddings import get_embeddings_manager
    config = get_config()
    return get_embeddings_manager(config)

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
    """
    Get current project context
    
    WHEN TO USE THIS TOOL:
    - User asks "what project am I working on?"
    - Need to verify current indexing status
    - Before switching projects
    - To check how many files are indexed
    - Debugging search or indexing issues
    
    This tool returns:
    - Current project name and path
    - Active collections in Qdrant
    - Total number of indexed documents
    - Working directory information
    """
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
    
    WHEN TO USE THIS TOOL:
    - Need to see the complete indexed content of a file
    - Debugging why search isn't finding expected content
    - User asks "show me all chunks for file X"
    - Verifying how a file was chunked during indexing
    - Getting full context around a search result
    
    This tool automatically:
    - Retrieves all indexed chunks for a file
    - Shows chunk boundaries and metadata
    - Supports range queries for large files
    - Works with code, config, and documentation files
    
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
    
    WHEN TO USE THIS TOOL:
    - Before reindexing to see what changed
    - User asks "what files have changed?"
    - Checking if index is up to date
    - After pulling changes from git
    - Debugging why search isn't finding new files
    
    This tool automatically:
    - Scans the filesystem for current files
    - Compares with indexed files in Qdrant
    - Detects file modifications using content hashes
    - Identifies added, modified, unchanged, and deleted files
    
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "index this file" for a specific code file
    - After creating a new code file
    - When a specific file isn't showing up in searches
    - Adding individual files without indexing entire directory
    - PREFER index_directory for multiple files
    
    This tool automatically:
    - Detects programming language from file extension
    - Uses AST-based chunking for better code understanding
    - Preserves function/class boundaries
    - Extracts imports and dependencies
    - Uses code-specific embeddings
    
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
        embeddings_manager = get_embeddings_manager_instance()
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
        
        # Get embedding dimension for code content type
        embedding_dimension = embeddings_manager.get_dimension("code") if hasattr(embeddings_manager, 'get_dimension') else None
        model_name = embeddings_manager.get_model_name("code") if hasattr(embeddings_manager, 'get_model_name') else None
        
        # Ensure collection exists with code content type
        ensure_collection(collection_name, embedding_dimension, model_name, content_type="code")
        
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
        
        # Process chunks one by one to avoid batch processing issues with specialized embeddings
        for chunk in chunks:
            # Use embeddings manager with code content type
            embedding_array = embeddings_manager.encode(chunk.content, content_type="code")
            # Handle both 1D and 2D arrays - if 2D with single row, extract it
            if embedding_array.ndim == 2 and embedding_array.shape[0] == 1:
                embedding = embedding_array[0].tolist()
            else:
                embedding = embedding_array.tolist()
            
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
            
            # Update BM25 index - use append for efficiency
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
            
            # Append new documents to BM25 index (more efficient than rebuilding)
            hybrid_searcher.bm25_manager.append_documents(collection_name, documents)
        
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "index this README" or documentation file
    - After creating or updating documentation
    - When documentation isn't showing up in searches
    - Adding individual markdown/rst files
    - PREFER index_directory for multiple files
    
    This tool automatically:
    - Parses markdown structure (headings, sections)
    - Preserves document hierarchy
    - Extracts code blocks and links
    - Uses section-based chunking
    - Uses documentation-specific embeddings
    
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
        
        # Get embeddings manager early to determine dimension
        embeddings_manager = get_embeddings_manager_instance()
        
        # Get embedding dimension for documentation content type
        embedding_dimension = embeddings_manager.get_dimension("documentation") if hasattr(embeddings_manager, 'get_dimension') else None
        model_name = embeddings_manager.get_model_name("documentation") if hasattr(embeddings_manager, 'get_model_name') else None
        
        # Determine collection
        if force_global:
            collection_name = "global_documentation"
        else:
            collection_name = get_collection_name(str(abs_path), "documentation")
        
        # Ensure collection exists with documentation content type
        ensure_collection(collection_name, embedding_dimension, model_name, content_type="documentation")
        
        # Index the file
        chunks = doc_indexer.index_file(str(abs_path))
        
        if not chunks:
            return {
                "error": "No content extracted",
                "error_code": "NO_CONTENT",
                "details": "No documentation chunks could be extracted from the file"
            }
        
        # Get Qdrant client
        qdrant_client = get_qdrant_client()
        
        # Prepare points for Qdrant
        points = []
        
        # Process chunks one by one to avoid batch processing issues with specialized embeddings
        for chunk in chunks:
            # Create unique ID for chunk
            chunk_id = hashlib.md5(
                f"{str(abs_path)}_{chunk['metadata']['chunk_index']}".encode()
            ).hexdigest()
            
            # Generate embedding using documentation content type
            embedding_array = embeddings_manager.encode(chunk['content'], content_type="documentation")
            # Handle both 1D and 2D arrays - if 2D with single row, extract it
            if embedding_array.ndim == 2 and embedding_array.shape[0] == 1:
                embedding = embedding_array[0].tolist()
            else:
                embedding = embedding_array.tolist()
            
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "index this project" or "index this directory"
    - Starting work on a new codebase
    - After cloning a repository
    - When search returns no results (might need indexing)
    - ALWAYS index before searching in a new project
    
    This tool automatically:
    - Indexes code, config, and documentation files
    - Respects .gitignore and .ragignore patterns
    - Processes files with appropriate specialized indexers
    - Creates searchable vector embeddings
    
    Args:
        directory: Directory to index (REQUIRED - must be absolute path or will be resolved from client's context)
        patterns: File patterns to include (optional - defaults to common code/config/doc patterns)
        recursive: Whether to search recursively (default: True)
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
        
        # Group files by type to minimize model switching
        code_files = []
        config_files = []
        doc_files = []
        
        for file_path in files_to_process:
            ext = file_path.suffix.lower()
            if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                config_files.append(file_path)
            elif ext in ['.md', '.markdown', '.rst', '.txt', '.mdx']:
                doc_files.append(file_path)
            else:
                code_files.append(file_path)
        
        # Now process files grouped by type
        total_files = len(files_to_process)
        if total_files > 0:
            logger = get_logger()
            console_logger.info(f"Starting to index {total_files} files from {directory}")
            console_logger.info(f"Files by type: {len(code_files)} code, {len(config_files)} config, {len(doc_files)} docs")
            
            # Report initial progress
            report_progress(0, total_files)
            
            files_processed = 0
            
            # Process each file type in order to minimize model switching
            file_groups = [
                ("code", code_files, index_code),
                ("config", config_files, index_config),
                ("documentation", doc_files, index_documentation)
            ]
            
            for group_name, file_list, index_func in file_groups:
                if not file_list:
                    continue
                
                console_logger.info(f"Indexing {len(file_list)} {group_name} files...")
                
                # Clear memory before switching to a new file type
                if files_processed > 0:
                    embeddings_manager = get_embeddings_manager_instance()
                    if hasattr(embeddings_manager, 'use_specialized') and embeddings_manager.use_specialized:
                        import gc
                        gc.collect()
                        logger.debug(f"Cleared memory before processing {group_name} files")
                
                for idx, file_path in enumerate(file_list):
                    try:
                        # Periodic memory cleanup every 50 files when using specialized embeddings
                        if files_processed > 0 and files_processed % 50 == 0:
                            embeddings_manager = get_embeddings_manager_instance()
                            if hasattr(embeddings_manager, 'use_specialized') and embeddings_manager.use_specialized:
                                # Run garbage collection
                                import gc
                                gc.collect()
                                logger.debug(f"Performed memory cleanup after {files_processed} files")
                        
                        result = index_func(str(file_path))
                        
                        if "error" not in result:
                            indexed_files.append(result.get("file_path", str(file_path)))
                            if "collection" in result:
                                collections_used.add(result["collection"])
                        else:
                            errors.append({"file": str(file_path), "error": result["error"]})
                        
                        files_processed += 1
                        
                        # Report progress every 10 files or at the end
                        if files_processed % 10 == 0 or files_processed == total_files:
                            report_progress(files_processed, total_files, str(file_path.name))
                    
                    except Exception as e:
                        errors.append({"file": str(file_path), "error": str(e)})
                        files_processed += 1
        
        # Build/rebuild BM25 indices for all affected collections after indexing
        if indexed_files and collections_used:
            console_logger.info(f"Building BM25 indices for {len(collections_used)} collections")
            hybrid_searcher = get_hybrid_searcher()
            qdrant_client = get_qdrant_client()
            
            for collection_name in collections_used:
                try:
                    # Build BM25 index from all documents in the collection
                    hybrid_searcher.bm25_manager.build_from_qdrant(collection_name, qdrant_client)
                    console_logger.debug(f"Built BM25 index for {collection_name}")
                except Exception as e:
                    console_logger.warning(f"Failed to build BM25 index for {collection_name}: {e}")
        
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
    
    WHEN TO USE THIS TOOL:
    - Files have been renamed, moved, or deleted
    - User asks to "reindex" or "refresh" the index
    - Search results contain outdated or stale data
    - After major code refactoring
    - When switching branches in git
    
    This tool automatically:
    - Detects file changes (added/modified/deleted) when incremental=True
    - Only processes changed files for efficiency
    - Clears stale data from moved/deleted files
    - Rebuilds search indices
    
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
            collections_used = set()  # Track all affected collections
            
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
                        # Track the collection that was affected
                        if "collection" in delete_result:
                            collections_used.add(delete_result["collection"])
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
            
            # Build/rebuild BM25 indices for affected collections after incremental indexing
            if (indexed_files or deleted_files) and collections_used:
                console_logger.info(f"Rebuilding BM25 indices for {len(collections_used)} collections after incremental reindex")
                hybrid_searcher = get_hybrid_searcher()
                qdrant_client = get_qdrant_client()
                
                for collection_name in collections_used:
                    try:
                        # Rebuild BM25 index from all documents in the collection
                        hybrid_searcher.bm25_manager.build_from_qdrant(collection_name, qdrant_client)
                        console_logger.debug(f"Rebuilt BM25 index for {collection_name}")
                    except Exception as e:
                        console_logger.warning(f"Failed to rebuild BM25 index for {collection_name}: {e}")
            
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

def _perform_hybrid_search(
    qdrant_client,
    embedding_model,
    query: str,
    query_embedding: List[float],
    search_collections: List[str],
    n_results: int,
    search_mode: str = "hybrid",
    collection_filter: Optional[Dict] = None,
    result_processor: Optional[Callable] = None,
    metadata_extractor: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """
    Reusable hybrid search component for all search functions.
    
    Args:
        qdrant_client: Qdrant client instance
        embedding_model: Embedding model instance
        query: Search query text
        query_embedding: Pre-computed query embedding
        search_collections: Collections to search
        n_results: Number of results to return
        search_mode: "vector", "keyword", or "hybrid"
        collection_filter: Optional filter for collection-specific queries
        result_processor: Optional function to process each result
        metadata_extractor: Optional function to extract type-specific metadata
    
    Returns:
        List of search results with scores and metadata
    """
    logger = get_logger()
    all_results = []
    
    for collection in search_collections:
        try:
            if search_mode == "vector":
                # Simple vector search
                results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    query_filter=collection_filter,
                    limit=n_results
                )
                
                for result in results:
                    # Ensure payload is a dictionary
                    if not isinstance(result.payload, dict):
                        logger.error(f"Invalid payload type in vector search: {type(result.payload)} - {result.payload}")
                        continue
                    
                    base_result = {
                        "score": float(result.score),
                        "vector_score": float(result.score),  # Always include vector_score
                        "collection": collection,
                        "search_mode": search_mode,
                        "file_path": result.payload.get("file_path", ""),
                        "content": result.payload.get("content", ""),
                        "chunk_index": result.payload.get("chunk_index", 0),
                        "project": result.payload.get("project", "unknown")
                    }
                    
                    # Apply type-specific metadata extraction
                    if metadata_extractor:
                        base_result.update(metadata_extractor(result.payload))
                    
                    # Apply result processing
                    if result_processor:
                        base_result = result_processor(base_result)
                    
                    all_results.append(base_result)
                    
            elif search_mode == "keyword":
                # BM25 keyword search only
                hybrid_searcher = get_hybrid_searcher()
                if not hybrid_searcher:
                    logger.warning("Hybrid searcher not available for keyword search")
                    continue
                    
                bm25_results = hybrid_searcher.bm25_manager.search(
                    collection_name=collection,
                    query=query,
                    k=n_results,
                    qdrant_client=qdrant_client
                )
                
                # Fetch full documents from Qdrant
                for doc_id, score in bm25_results:
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
                            query_vector=query_embedding,  # Use actual query vector
                            query_filter=filter_conditions,
                            limit=1
                        )
                        
                        if search_result:
                            payload = search_result[0].payload
                            
                            # Ensure payload is a dictionary
                            if not isinstance(payload, dict):
                                logger.error(f"Invalid payload type in keyword search: {type(payload)} - {payload}")
                                continue
                            
                            base_result = {
                                "score": score,
                                "collection": collection,
                                "search_mode": search_mode,
                                "file_path": payload.get("file_path", ""),
                                "content": payload.get("content", ""),
                                "chunk_index": payload.get("chunk_index", 0),
                                "project": payload.get("project", "unknown")
                            }
                            
                            # Apply type-specific metadata extraction
                            if metadata_extractor:
                                base_result.update(metadata_extractor(payload))
                            
                            # Apply result processing
                            if result_processor:
                                base_result = result_processor(base_result)
                            
                            all_results.append(base_result)
                            
            else:  # hybrid mode
                # Get vector search results first
                vector_results = []
                vector_scores_map = {}
                result_objects_map = {}  # Store original result objects
                
                search_results = qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_embedding,
                    query_filter=collection_filter,
                    limit=n_results * 2  # Get more for fusion
                )
                
                for result in search_results:
                    # Ensure payload is a dictionary
                    if not isinstance(result.payload, dict):
                        logger.error(f"Invalid payload type: {type(result.payload)} - {result.payload}")
                        continue
                    
                    # Safely access payload fields
                    file_path = result.payload.get('file_path', '')
                    chunk_index = result.payload.get('chunk_index', 0)
                    
                    if not file_path:
                        logger.warning("Search result missing file_path")
                        continue
                    
                    doc_id = f"{file_path}_{chunk_index}"
                    score = float(result.score)
                    vector_results.append((doc_id, score))
                    vector_scores_map[doc_id] = score
                    result_objects_map[doc_id] = result  # Store the original result
                
                # Get BM25 results
                hybrid_searcher = get_hybrid_searcher()
                if not hybrid_searcher:
                    logger.warning("Hybrid searcher not available, falling back to vector search")
                    # Just use vector results
                    for result in search_results[:n_results]:
                        # Ensure payload is a dictionary
                        if not isinstance(result.payload, dict):
                            logger.error(f"Invalid payload type in hybrid fallback: {type(result.payload)} - {result.payload}")
                            continue
                        
                        base_result = {
                            "score": float(result.score),
                            "collection": collection,
                            "search_mode": "vector",  # Indicate fallback
                            "file_path": result.payload.get("file_path", ""),
                            "content": result.payload.get("content", ""),
                            "chunk_index": result.payload.get("chunk_index", 0),
                            "project": result.payload.get("project", "unknown")
                        }
                        
                        if metadata_extractor:
                            base_result.update(metadata_extractor(result.payload))
                        if result_processor:
                            base_result = result_processor(base_result)
                        
                        all_results.append(base_result)
                    continue
                
                bm25_results = hybrid_searcher.bm25_manager.search(
                    collection_name=collection,
                    query=query,
                    k=n_results * 2,
                    qdrant_client=qdrant_client
                )
                bm25_scores_map = {doc_id: score for doc_id, score in bm25_results}
                
                # Determine search type from collection name
                search_type = "code" if "_code" in collection else "documentation" if "_documentation" in collection else "config" if "_config" in collection else "general"
                vector_weight, bm25_weight = hybrid_searcher.get_weights_for_search_type(search_type)
                
                # Fuse results using linear combination with exact match bonus
                fused_results = hybrid_searcher.linear_combination_with_exact_match(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    query=query,
                    result_objects_map=result_objects_map,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    exact_match_bonus=0.2
                )
                
                # Use original results or fetch BM25-only results
                for result in fused_results[:n_results]:
                    doc_id = result.content  # doc_id is stored in content field
                    
                    if doc_id in result_objects_map:
                        # Use the original result object
                        original_result = result_objects_map[doc_id]
                        payload = original_result.payload
                        
                        # Ensure payload is a dictionary
                        if not isinstance(payload, dict):
                            logger.error(f"Invalid payload type in hybrid search: {type(payload)} - {payload}")
                            continue
                        
                        base_result = {
                            "score": result.combined_score,
                            "vector_score": vector_scores_map.get(doc_id, None),
                            "bm25_score": bm25_scores_map.get(doc_id, None),
                            "collection": collection,
                            "search_mode": search_mode,
                            "file_path": payload.get("file_path", ""),
                            "content": payload.get("content", ""),
                            "chunk_index": payload.get("chunk_index", 0),
                            "project": payload.get("project", "unknown")
                        }
                    else:
                        # This is a BM25-only result, need to fetch it
                        parts = doc_id.rsplit('_', 1)
                        if len(parts) == 2:
                            file_path = parts[0]
                            chunk_index = int(parts[1])
                            
                            # Fetch from Qdrant with actual query vector
                            filter_conditions = {
                                "must": [
                                    {"key": "file_path", "match": {"value": file_path}},
                                    {"key": "chunk_index", "match": {"value": chunk_index}}
                                ]
                            }
                            
                            search_result = qdrant_client.search(
                                collection_name=collection,
                                query_vector=query_embedding,  # Use actual query vector
                                query_filter=filter_conditions,
                                limit=1
                            )
                            
                            if search_result:
                                payload = search_result[0].payload
                                
                                # Ensure payload is a dictionary
                                if not isinstance(payload, dict):
                                    logger.error(f"Invalid payload type in BM25 fetch: {type(payload)} - {payload}")
                                    continue
                                
                                base_result = {
                                    "score": result.combined_score,
                                    "vector_score": vector_scores_map.get(doc_id, None),
                                    "bm25_score": bm25_scores_map.get(doc_id, None),
                                    "collection": collection,
                                    "search_mode": search_mode,
                                    "file_path": payload.get("file_path", ""),
                                    "content": payload.get("content", ""),
                                    "chunk_index": payload.get("chunk_index", 0),
                                    "project": payload.get("project", "unknown")
                                }
                            else:
                                continue
                        else:
                            continue
                    
                    # Apply type-specific metadata extraction
                    if metadata_extractor:
                        base_result.update(metadata_extractor(payload))
                    
                    # Apply result processing
                    if result_processor:
                        base_result = result_processor(base_result)
                    
                    all_results.append(base_result)
                    
        except Exception as e:
            # Skip if collection doesn't exist
            logger.debug(f"Error searching collection {collection}: {e}")
            pass
    
    return all_results

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
    
    WHEN TO USE THIS TOOL:
    - User asks to "search for X" or "find X" in the codebase
    - User asks "where is X defined/used/implemented?"
    - General queries about code, configs, or documentation
    - When you need to understand how something works
    - PREFER specialized tools (search_code, search_docs, search_config) for specific file types
    
    This tool automatically:
    - Searches across all indexed content types (code, config, docs)
    - Uses hybrid search (vector + keyword) for best results
    - Includes surrounding context chunks
    - Supports progressive context for token efficiency
    
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
            embeddings_manager = get_embeddings_manager_instance()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embeddings_manager,
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
        embeddings_manager = get_embeddings_manager_instance()
        qdrant_client = get_qdrant_client()
        
        # Generate query embedding using general content type for cross-collection search
        query_embedding_array = embeddings_manager.encode(query, content_type="general")
        # Handle both 1D and 2D arrays - if 2D with single row, extract it
        if query_embedding_array.ndim == 2 and query_embedding_array.shape[0] == 1:
            query_embedding = query_embedding_array[0].tolist()
        else:
            query_embedding = query_embedding_array.tolist()
        
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
        
        # Use the new hybrid search helper
        def general_metadata_extractor(payload):
            # Extract type-specific metadata for general search
            collection = payload.get("collection", "")
            return {
                "display_path": payload.get("display_path", payload.get("file_path", "")),
                "type": "code" if "_code" in collection else ("config" if "_config" in collection else "docs"),
                "language": payload.get("language", "") if "_code" in collection else payload.get("format", ""),
                "line_start": payload.get("line_start", 0),
                "line_end": payload.get("line_end", 0)
            }
        
        all_results = _perform_hybrid_search(
            qdrant_client=qdrant_client,
            embedding_model=embeddings_manager,
            query=query,
            query_embedding=query_embedding,
            search_collections=search_collections,
            n_results=n_results,
            search_mode=search_mode,
            metadata_extractor=general_metadata_extractor
        )
        
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
            # Get embedding dimension from the embeddings manager
            embedding_dimension = embeddings_manager.get_sentence_embedding_dimension()
            all_results = _expand_search_context(all_results, qdrant_client, search_collections, context_chunks, embedding_dimension)
        
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
        import traceback
        duration_ms = (time.time() - start_time) * 1000
        tb_str = traceback.format_exc()
        console_logger.error(f"Failed search: {query[:50]}... - {str(e)}", extra={
            "operation": "search",
            "query_length": len(query),
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error",
            "traceback": tb_str
        })
        return {
            "results": [],
            "query": query,
            "error": str(e),
            "error_code": "SEARCH_ERROR",
            "total": 0,
            "message": f"Search failed: {str(e)}",
            "debug_info": {
                "error_type": type(e).__name__,
                "error_location": tb_str.split('\n')[-3:-1] if '\n' in tb_str else tb_str,
                "full_traceback": tb_str
            }
        }

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
    
    WHEN TO USE THIS TOOL:
    - User asks about functions, classes, methods, or code structure
    - User asks "how does X work?" or "show me the implementation of X"
    - Looking for specific programming constructs or patterns
    - Debugging or understanding code behavior
    - Finding usage examples of APIs or functions
    
    This tool automatically:
    - Searches only in code files (.py, .js, .ts, .go, etc.)
    - Uses code-specific embeddings for better accuracy
    - Preserves code structure (functions, classes) in results
    - Includes line numbers and language information
    
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
            embeddings_manager = get_embeddings_manager_instance()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embeddings_manager,
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
                semantic_cache=semantic_cache,
                collection_suffix="_code"  # Only search code collections
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
        embeddings_manager = get_embeddings_manager_instance()
        qdrant_client = get_qdrant_client()
        
        # Generate query embedding using code content type
        query_embedding_array = embeddings_manager.encode(query, content_type="code")
        # Handle both 1D and 2D arrays - if 2D with single row, extract it
        if query_embedding_array.ndim == 2 and query_embedding_array.shape[0] == 1:
            query_embedding = query_embedding_array[0].tolist()
        else:
            query_embedding = query_embedding_array.tolist()
        
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
        
        # Use the new hybrid search helper
        def code_metadata_extractor(payload):
            return {
                "display_path": payload.get("display_path", payload.get("file_path", "")),
                "language": payload.get("language", ""),
                "line_range": {
                    "start": payload.get("line_start", 0),
                    "end": payload.get("line_end", 0)
                },
                "chunk_type": payload.get("chunk_type", "general"),
                "complexity": payload.get("complexity", 0),
                "imports": payload.get("imports", [])
            }
        
        all_results = _perform_hybrid_search(
            qdrant_client=qdrant_client,
            embedding_model=embeddings_manager,
            query=query,
            query_embedding=query_embedding,
            search_collections=search_collections,
            n_results=n_results,
            search_mode=search_mode,  # Now search_code supports hybrid!
            collection_filter=filter_dict,
            metadata_extractor=code_metadata_extractor
        )
        
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
            # Get embedding dimension from the embeddings manager
            embedding_dimension = embeddings_manager.get_sentence_embedding_dimension()
            expanded_results = _expand_search_context(formatted_results, qdrant_client, search_collections, context_chunks, embedding_dimension)
            
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
        import traceback
        tb_str = traceback.format_exc()
        console_logger.error(f"Code search failed", extra={
            "operation": "search_code",
            "query": query,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": tb_str
        })
        return {
            "results": [],
            "query": query,
            "error": str(e),
            "error_code": "SEARCH_ERROR",
            "total": 0,
            "message": f"Code search failed: {str(e)}",
            "debug_info": {
                "error_type": type(e).__name__,
                "error_location": tb_str.split('\n')[-3:-1] if '\n' in tb_str else tb_str,
                "full_traceback": tb_str
            }
        }

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
    
    WHEN TO USE THIS TOOL:
    - User asks about documentation, guides, or tutorials
    - Looking for setup instructions or configuration guides
    - User asks "how to use X" or "what does the README say"
    - Finding API documentation or usage examples
    - Understanding project structure or architecture docs
    
    This tool automatically:
    - Searches only in documentation files (.md, .rst, .txt)
    - Preserves document structure (headings, sections)
    - Uses documentation-optimized embeddings
    - Returns section context with headings
    
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
            embeddings_manager = get_embeddings_manager_instance()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embeddings_manager,
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
                semantic_cache=semantic_cache,
                collection_suffix="_documentation"  # Only search documentation collections
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
        embeddings_manager = get_embeddings_manager_instance()
        
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
        
        # Generate query embedding for vector search using documentation content type
        query_embedding_array = embeddings_manager.encode(query, content_type="documentation")
        # Handle both 1D and 2D arrays - if 2D with single row, extract it
        if query_embedding_array.ndim == 2 and query_embedding_array.shape[0] == 1:
            query_embedding = query_embedding_array[0].tolist()
        else:
            query_embedding = query_embedding_array.tolist()
        
        # Build filter if doc_type specified
        filter_dict = None
        if doc_type:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            filter_dict = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type.lower()))]
            )
        
        # Use the new hybrid search helper
        def docs_metadata_extractor(payload):
            # Ensure file_name is set (fallback to extracting from file_path if needed)
            file_name = payload.get("file_name", "")
            if not file_name and payload.get("file_path"):
                file_name = Path(payload.get("file_path", "")).name
            
            return {
                "file_name": file_name,
                "doc_type": payload.get("doc_type", "markdown"),
                "title": payload.get("title", ""),
                "heading": payload.get("heading", ""),
                "heading_hierarchy": payload.get("heading_hierarchy", []),
                "heading_level": payload.get("heading_level", 0),
                "chunk_type": payload.get("chunk_type", "section"),
                "has_code_blocks": payload.get("has_code_blocks", False),
                "code_languages": payload.get("code_languages", [])
            }
        
        def docs_result_processor(result):
            # Truncate content for docs
            if "content" in result:
                result["content"] = _truncate_content(result["content"])
            return result
        
        all_results = _perform_hybrid_search(
            qdrant_client=qdrant_client,
            embedding_model=embeddings_manager,
            query=query,
            query_embedding=query_embedding,
            search_collections=search_collections,
            n_results=n_results,
            search_mode=search_mode,
            collection_filter=filter_dict,
            metadata_extractor=docs_metadata_extractor,
            result_processor=docs_result_processor
        )
        
        
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
            # Get embedding dimension from the embeddings manager
            embedding_dimension = embeddings_manager.get_dimension("documentation") if hasattr(embeddings_manager, 'get_dimension') else 384
            expanded_results = _expand_search_context(formatted_results, qdrant_client, search_collections, context_chunks, embedding_dimension)
            
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
        import traceback
        duration_ms = (time.time() - start_time) * 1000
        tb_str = traceback.format_exc()
        console_logger.error(f"Documentation search failed", extra={
            "operation": "search_docs",
            "query": query,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error",
            "traceback": tb_str
        })
        return {
            "results": [],
            "query": query,
            "error": str(e),
            "error_code": "SEARCH_ERROR",
            "total": 0,
            "search_mode": search_mode,
            "message": f"Documentation search failed: {str(e)}",
            "debug_info": {
                "error_type": type(e).__name__,
                "error_location": tb_str.split('\n')[-3:-1] if '\n' in tb_str else tb_str,
                "full_traceback": tb_str
            }
        }

@mcp.tool()
def search_config(
    query: str, 
    file_type: Optional[str] = None, 
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
    Search specifically in configuration files
    
    WHEN TO USE THIS TOOL:
    - User asks about configuration settings or options
    - Looking for environment variables or API keys
    - User asks "how is X configured?" or "what are the settings for Y?"
    - Finding database connections or service endpoints
    - Understanding deployment or build configurations
    
    This tool automatically:
    - Searches only in config files (.json, .yaml, .xml, .env, etc.)
    - Preserves configuration structure and paths
    - Uses config-optimized embeddings
    - Returns hierarchical context (nested config paths)
    
    Args:
        query: Search query
        file_type: Filter by config file type (e.g., 'json', 'yaml', 'toml', 'xml', 'ini')
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
            embeddings_manager = get_embeddings_manager_instance()
            qdrant_client = get_qdrant_client()
            progressive_manager = get_progressive_manager(
                qdrant_client, 
                embeddings_manager,
                config.get("progressive_context", {})
            )
            
            # Use progressive context for config search
            progressive_result = progressive_manager.get_progressive_context(
                query=query,
                level=context_level,
                n_results=n_results,
                cross_project=cross_project,
                search_mode=search_mode,
                include_dependencies=False,  # Not applicable for config
                semantic_cache=semantic_cache,
                collection_suffix="_config"  # Only search config collections
            )
            
            # Filter results by file type if specified
            if file_type:
                progressive_result.results = [
                    r for r in progressive_result.results 
                    if r.get("file_type", "").lower() == file_type.lower() or r.get("file_path", "").endswith(f".{file_type}")
                ]
            
            # Convert to standard response format with progressive metadata
            response = {
                "results": progressive_result.results,
                "query": query,
                "file_type_filter": file_type,
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
                tracker.track_search(query, response["results"], search_type="config_progressive")
            
            # Log completion
            console_logger.info(f"Progressive config search completed", extra={
                "operation": "search_config_progressive",
                "duration": time.time() - start_time,
                "results_count": len(response["results"]),
                "level": progressive_result.level,
                "cache_hit": progressive_result.from_cache,
                "file_type_filter": file_type
            })
            
            return response
            
        except Exception as e:
            # Log error but fall back to regular search
            console_logger.warning(f"Progressive config search failed, falling back to regular search: {e}", extra={
                "operation": "search_config_progressive_fallback",
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
    
    console_logger.info(f"Starting config search: {query[:50]}...", extra={
        "operation": "search_config",
        "query_length": len(query),
        "n_results": n_results,
        "cross_project": cross_project,
        "search_mode": search_mode,
        "file_type_filter": file_type
    })
    
    try:
        # Get collections to search
        search_collections = []
        current_project = get_current_project()
        embeddings_manager = get_embeddings_manager_instance()
        qdrant_client = get_qdrant_client()
        
        if cross_project:
            # Search all config collections
            all_collections = qdrant_client.get_collections().collections
            search_collections = [
                c.name for c in all_collections 
                if "_config" in c.name
            ]
        else:
            # Search only current project's config collection
            if current_project:
                collection_name = f"project_{current_project['name']}_config"
                # Check if collection exists
                all_collections = qdrant_client.get_collections().collections
                existing_collections = [c.name for c in all_collections]
                if collection_name in existing_collections:
                    search_collections = [collection_name]
        
        if not search_collections:
            return {
                "results": [],
                "query": query,
                "total": 0,
                "search_mode": search_mode,
                "project_context": current_project["name"] if current_project else "no project",
                "search_scope": "all projects" if cross_project else "current project",
                "file_type_filter": file_type,
                "message": "No config collections found to search"
            }
        
        # Generate query embedding for vector search using config content type
        query_embedding_array = embeddings_manager.encode(query, content_type="config")
        # Handle both 1D and 2D arrays - if 2D with single row, extract it
        if query_embedding_array.ndim == 2 and query_embedding_array.shape[0] == 1:
            query_embedding = query_embedding_array[0].tolist()
        else:
            query_embedding = query_embedding_array.tolist()
        
        # Get config settings
        config_settings = get_config()
        
        # Define metadata extraction for config files
        def extract_config_metadata(payload):
            return {
                "file_type": payload.get("file_type", ""),
                "path": payload.get("path", ""),
                "value": payload.get("value", "")
            }
        
        # Define result processing
        def process_config_result(result):
            result["type"] = "config"
            return result
        
        # Perform search across collections
        all_results = _perform_hybrid_search(
            qdrant_client=qdrant_client,
            embedding_model=embeddings_manager,
            query=query,
            query_embedding=query_embedding,
            search_collections=search_collections,
            n_results=n_results * 2,  # Get more results for filtering
            search_mode=search_mode,
            collection_filter=None,  # No specific filter for config search
            result_processor=process_config_result,
            metadata_extractor=extract_config_metadata
        )
        
        # Filter by file type if specified
        if file_type:
            all_results = [
                r for r in all_results 
                if r.get("file_type", "").lower() == file_type.lower() or 
                   r.get("file_path", "").endswith(f".{file_type}")
            ]
        
        # Apply enhanced ranking
        if all_results and search_mode == "hybrid":
            # For config files, we don't need dependency graph
            dependency_graph = {}
            
            # Get query context
            query_context = {}
            if current_project and "root_path" in current_project:
                pass
            
            # Apply enhanced ranking
            ranking_config = config_settings.get_section("search").get("enhanced_ranking", {})
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
            # Get embedding dimension from the embeddings manager
            if hasattr(embeddings_manager, 'get_dimension'):
                embedding_dimension = embeddings_manager.get_dimension("config")
            elif hasattr(embeddings_manager, 'get_sentence_embedding_dimension'):
                embedding_dimension = embeddings_manager.get_sentence_embedding_dimension()
            else:
                embedding_dimension = None
            all_results = _expand_search_context(all_results, qdrant_client, search_collections, context_chunks, embedding_dimension)
        
        # Truncate content in results to prevent token limit issues
        for result in all_results:
            if "content" in result:
                result["content"] = _truncate_content(result["content"], max_length=1500)
            if "expanded_content" in result:
                result["expanded_content"] = _truncate_content(result["expanded_content"], max_length=2000)
        
        # Track the search in context
        tracker = get_context_tracker()
        tracker.track_search(query, all_results, search_type="config")
        
        duration_ms = (time.time() - start_time) * 1000
        console_logger.info(f"Completed config search: {query[:50]}...", extra={
            "operation": "search_config",
            "query_length": len(query),
            "duration_ms": duration_ms,
            "results_found": len(all_results),
            "collections_searched": len(search_collections),
            "search_mode": search_mode,
            "file_type_filter": file_type,
            "status": "success"
        })
        
        return {
            "results": all_results,
            "query": query,
            "file_type_filter": file_type,
            "total": len(all_results),
            "search_mode": search_mode,
            "project_context": current_project["name"] if current_project else None,
            "search_scope": "all projects" if cross_project else "current project"
        }
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"Config search failed", extra={
            "operation": "search_config",
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
            "message": f"Config search failed: {str(e)}"
        }

@mcp.tool()
def switch_project(project_path: str) -> Dict[str, Any]:
    """
    Switch to a different project context
    
    WHEN TO USE THIS TOOL:
    - User asks to "switch to project X" or "work on project Y"
    - Moving between different codebases
    - After cloning a new repository
    - When searches return results from wrong project
    - Organizing work across multiple projects
    
    This tool automatically:
    - Changes the working directory
    - Updates project context for searches
    - Isolates collections by project
    - Resets project-specific caches
    - Returns project statistics
    """
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
    """
    Index a configuration file
    
    WHEN TO USE THIS TOOL:
    - User asks to "index this config file"
    - After creating or updating configuration files
    - When config files aren't showing up in searches
    - Adding individual JSON/YAML/XML files
    - PREFER index_directory for multiple files
    
    This tool automatically:
    - Detects config format (JSON, YAML, XML, INI, etc.)
    - Preserves hierarchical structure
    - Extracts nested configuration paths
    - Uses config-specific embeddings
    - Handles environment variables
    """
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
        embeddings_manager = get_embeddings_manager_instance()
        qdrant_client = get_qdrant_client()
        
        # Determine collection
        if force_global:
            collection_name = "global_config"
        else:
            collection_name = get_collection_name(str(abs_path), "config")
        
        # Get embedding dimension for config content type
        embedding_dimension = embeddings_manager.get_dimension("config") if hasattr(embeddings_manager, 'get_dimension') else None
        model_name = embeddings_manager.get_model_name("config") if hasattr(embeddings_manager, 'get_model_name') else None
        
        # Ensure collection exists with config content type
        ensure_collection(collection_name, embedding_dimension, model_name, content_type="config")
        
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
        
        # Process chunks one by one to avoid batch processing issues with specialized embeddings
        for idx, chunk in enumerate(chunks):
            # Use embeddings manager with config content type
            embedding_array = embeddings_manager.encode(chunk.content, content_type="config")
            
            # Handle both 1D and 2D arrays - if 2D with single row, extract it
            if embedding_array.ndim == 2 and embedding_array.shape[0] == 1:
                embedding = embedding_array[0].tolist()
            else:
                embedding = embedding_array.tolist()
            
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
def rebuild_bm25_indices(collections: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Rebuild BM25 indices for specified collections or all collections.
    
    WHEN TO USE THIS TOOL:
    - After server restart when keyword search isn't working
    - User reports "keyword search not finding results"
    - When hybrid search returns only vector results
    - After major reindexing operations
    - Fixing search quality issues
    
    This tool automatically:
    - Rebuilds keyword search indices from Qdrant data
    - Processes all chunks in collections
    - Restores hybrid search functionality
    - Reports chunk counts per collection
    
    Args:
        collections: Optional list of collection names to rebuild. If None, rebuilds all.
    
    Returns:
        Status of rebuild operation with chunk counts (not file counts)
    """
    start_time = time.time()
    console_logger.info("Starting BM25 index rebuild", extra={
        "operation": "rebuild_bm25_indices",
        "collections": collections
    })
    
    try:
        client = get_qdrant_client()
        hybrid_searcher = get_hybrid_searcher()
        
        # Get collections to rebuild
        if collections:
            target_collections = collections
        else:
            # Get all collections
            all_collections = client.get_collections().collections
            target_collections = [c.name for c in all_collections]
        
        results = {
            "rebuilt": [],
            "failed": [],
            "total_chunks": 0
        }
        
        for collection_name in target_collections:
            try:
                console_logger.info(f"Rebuilding BM25 index for {collection_name}")
                
                # Use the build_from_qdrant method
                success = hybrid_searcher.bm25_manager.build_from_qdrant(collection_name, client)
                
                if success:
                    # Get chunk count (documents in BM25 are actually chunks)
                    chunk_count = len(hybrid_searcher.bm25_manager.documents.get(collection_name, []))
                    results["rebuilt"].append({
                        "collection": collection_name,
                        "chunks": chunk_count
                    })
                    results["total_chunks"] += chunk_count
                else:
                    results["failed"].append({
                        "collection": collection_name,
                        "reason": "No chunks found or build failed"
                    })
                    
            except Exception as e:
                results["failed"].append({
                    "collection": collection_name,
                    "reason": str(e)
                })
                console_logger.error(f"Failed to rebuild BM25 index for {collection_name}: {e}")
        
        duration_ms = (time.time() - start_time) * 1000
        
        console_logger.info("BM25 index rebuild completed", extra={
            "operation": "rebuild_bm25_indices",
            "duration_ms": duration_ms,
            "rebuilt_count": len(results["rebuilt"]),
            "failed_count": len(results["failed"]),
            "total_chunks": results["total_chunks"]
        })
        
        return {
            "status": "success" if not results["failed"] else "partial",
            "rebuilt": results["rebuilt"],
            "failed": results["failed"],
            "total_chunks": results["total_chunks"],
            "duration_ms": duration_ms,
            "message": f"Rebuilt {len(results['rebuilt'])} collections with {results['total_chunks']} total chunks"
        }
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        console_logger.error(f"BM25 rebuild failed: {e}", extra={
            "operation": "rebuild_bm25_indices",
            "duration_ms": duration_ms,
            "error": str(e)
        })
        return {
            "status": "error",
            "error": str(e),
            "duration_ms": duration_ms
        }

@mcp.tool()
def get_memory_status() -> Dict[str, Any]:
    """
    Get detailed memory status of the MCP server.
    
    WHEN TO USE THIS TOOL:
    - User asks about memory usage or performance
    - Debugging slow performance or crashes
    - Before performing memory-intensive operations
    - Monitoring server resource consumption
    - When getting out-of-memory errors
    
    This tool returns:
    - Current memory usage by component
    - Embedding model memory consumption
    - Cache sizes and limits
    - Cleanup statistics
    - Memory pressure indicators
    
    Returns current memory usage, component breakdown, and cleanup statistics.
    """
    try:
        memory_manager = get_memory_manager()
        memory_report = memory_manager.get_memory_report()
        
        # Format for easy reading
        return {
            "timestamp": memory_report["timestamp"],
            "process": {
                "rss_mb": round(memory_report["system"].get("process_rss_mb", 0), 1),
                "vms_mb": round(memory_report["system"].get("process_vms_mb", 0), 1),
                "percent_of_limit": round(
                    (memory_report["system"].get("process_rss_mb", 0) / 
                     memory_report["limits"]["total_mb"]) * 100, 1
                )
            },
            "components": {
                name: {
                    "memory_mb": round(stats.get("memory_mb", 0), 1),
                    "items": stats.get("items_count", 0),
                    "last_cleanup": stats.get("last_cleanup"),
                    "cleanup_count": stats.get("cleanup_count", 0)
                }
                for name, stats in memory_report["components"].items()
                if "error" not in stats
            },
            "system": {
                "available_mb": round(memory_report["system"].get("system_available_mb", 0), 1),
                "system_percent": round(memory_report["system"].get("system_percent", 0), 1)
            },
            "limits": memory_report["limits"],
            "status": "critical" if memory_report["system"].get("process_rss_mb", 0) > memory_report["limits"]["aggressive_threshold_mb"]
                     else "high" if memory_report["system"].get("process_rss_mb", 0) > memory_report["limits"]["cleanup_threshold_mb"]
                     else "normal"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def trigger_memory_cleanup(aggressive: bool = False) -> Dict[str, Any]:
    """
    Manually trigger memory cleanup.
    
    WHEN TO USE THIS TOOL:
    - Server is running slow or using too much memory
    - Before performing memory-intensive operations
    - User reports performance issues
    - After indexing large projects
    - When memory_status shows high usage
    
    This tool automatically:
    - Clears unused caches
    - Evicts least-recently-used models
    - Runs garbage collection
    - Frees up system memory
    - Reports memory freed
    
    Args:
        aggressive: If True, performs aggressive cleanup (removes more items)
        
    Returns:
        Cleanup results and new memory status
    """
    try:
        memory_manager = get_memory_manager()
        
        # Get memory before cleanup
        before_report = memory_manager.get_memory_report()
        before_mb = before_report["system"].get("process_rss_mb", 0)
        
        # Trigger cleanup on all components
        total_removed = 0
        component_results = {}
        
        for name, component in memory_manager.registry.components.items():
            try:
                removed = component.cleanup(aggressive=aggressive)
                component_results[name] = {
                    "removed_items": removed,
                    "status": "success"
                }
                total_removed += removed
            except Exception as e:
                component_results[name] = {
                    "removed_items": 0,
                    "status": "error",
                    "error": str(e)
                }
        
        # Force garbage collection
        import gc
        gc.collect()
        gc.collect()  # Run twice
        
        # Get memory after cleanup
        after_report = memory_manager.get_memory_report()
        after_mb = after_report["system"].get("process_rss_mb", 0)
        
        return {
            "cleanup_type": "aggressive" if aggressive else "normal",
            "total_items_removed": total_removed,
            "component_results": component_results,
            "memory_freed_mb": round(before_mb - after_mb, 1),
            "memory_before_mb": round(before_mb, 1),
            "memory_after_mb": round(after_mb, 1),
            "current_status": "critical" if after_mb > after_report["limits"]["aggressive_threshold_mb"]
                           else "high" if after_mb > after_report["limits"]["cleanup_threshold_mb"]
                           else "normal"
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def health_check() -> Dict[str, Any]:
    """
    Check the health status of all services.
    
    WHEN TO USE THIS TOOL:
    - User asks "is everything working?"
    - Debugging connection or performance issues
    - Before starting important operations
    - When searches return errors
    - Regular system health monitoring
    
    This tool checks:
    - Qdrant connection and collections
    - Embedding model availability
    - Disk space availability
    - Memory usage (if psutil available)
    - Current project context
    - GitHub integration status
    
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
    
    # Check memory manager status
    try:
        memory_manager = get_memory_manager()
        memory_report = memory_manager.get_memory_report()
        
        process_mb = memory_report["system"].get("process_rss_mb", 0)
        memory_limits = memory_report["limits"]
        
        # Determine memory health
        memory_status = "healthy"
        if process_mb > memory_limits["aggressive_threshold_mb"]:
            memory_status = "critical"
            health_status["status"] = "warning"
        elif process_mb > memory_limits["cleanup_threshold_mb"]:
            memory_status = "high"
        
        health_status["services"]["memory_manager"] = {
            "status": memory_status,
            "process_memory_mb": round(process_mb, 1),
            "component_memory_mb": round(memory_report["component_total_mb"], 1),
            "components": {
                name: {
                    "memory_mb": round(stats.get("memory_mb", 0), 1),
                    "items": stats.get("items_count", 0)
                }
                for name, stats in memory_report["components"].items()
                if "error" not in stats
            },
            "thresholds": {
                "cleanup_mb": memory_limits["cleanup_threshold_mb"],
                "aggressive_mb": memory_limits["aggressive_threshold_mb"],
                "total_limit_mb": memory_limits["total_mb"]
            }
        }
    except Exception as e:
        health_status["services"]["memory_manager"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    # Log health check result
    console_logger.info(
        f"Health check completed: {health_status['status']}",
        extra={
            "operation": "health_check",
            "status": health_status["status"],
            "qdrant_status": health_status["services"].get("qdrant", {}).get("status", "unknown"),
            "model_status": health_status["services"].get("embedding_model", {}).get("status", "unknown"),
            "memory_status": health_status["services"].get("memory_manager", {}).get("status", "unknown")
        }
    )
    
    return health_status


# GitHub Integration - Import modules with graceful fallback
try:
    from github_integration.client import get_github_client
    from github_integration.issue_analyzer import get_issue_analyzer
    from github_integration.code_generator import get_code_generator
    from github_integration.workflows import get_github_workflows
    from github_integration import PROJECTS_AVAILABLE, get_projects_manager
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    PROJECTS_AVAILABLE = False
    console_logger.warning("GitHub integration not available. Install with: pip install PyGithub GitPython")


# Global GitHub instances (lazy initialization)
_github_client = None
_issue_analyzer = None
_code_generator = None
_github_workflows = None
_projects_manager = None


def get_github_instances():
    """Get or create GitHub service instances."""
    global _github_client, _issue_analyzer, _code_generator, _github_workflows, _projects_manager
    
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
    
    # Initialize Projects manager if available
    if PROJECTS_AVAILABLE and _projects_manager is None:
        try:
            _projects_manager = get_projects_manager(_github_client)
        except Exception as e:
            console_logger.warning(f"Failed to initialize GitHub Projects manager: {e}")
            _projects_manager = None
    
    return _github_client, _issue_analyzer, _code_generator, _github_workflows, _projects_manager


def run_async_in_thread(coro):
    """
    Helper function to run async coroutines in MCP context.
    Handles the case where an event loop is already running by using a thread pool.
    
    Args:
        coro: The coroutine to run
        
    Returns:
        The result of the coroutine
    """
    import asyncio
    import concurrent.futures
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, use thread to avoid event loop issues
            def run_async():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_async)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up the loop when done
            loop.close()
            asyncio.set_event_loop(None)


def check_github_available():
    """Check if GitHub integration is available."""
    if not GITHUB_AVAILABLE:
        return {
            "error": "GitHub integration not available",
            "message": "Install with: pip install PyGithub GitPython"
        }
    return None


def check_projects_available():
    """Check if GitHub Projects integration is available."""
    if not PROJECTS_AVAILABLE:
        return {
            "error": "GitHub Projects integration not available",
            "message": "Install with: pip install 'gql[aiohttp]'"
        }
    return None


def check_repository_context(github_client):
    """Check if a repository context is set."""
    if not github_client.get_current_repository():
        return {
            "error": "No repository context set",
            "message": "Use github_switch_repository first"
        }
    return None


def check_projects_manager(projects_manager):
    """Check if projects manager is available."""
    if not projects_manager:
        return {
            "error": "Projects manager not available",
            "message": "Failed to initialize GitHub Projects manager"
        }
    return None


def validate_github_prerequisites(require_projects=False, require_repo=False):
    """
    Validate common GitHub prerequisites and return appropriate error if any check fails.
    
    Args:
        require_projects: Whether GitHub Projects integration is required
        require_repo: Whether a repository context is required
        
    Returns:
        Tuple of (error_dict or None, github_instances)
    """
    # Check basic GitHub availability
    error = check_github_available()
    if error:
        return error, None
    
    # Check projects availability if required
    if require_projects:
        error = check_projects_available()
        if error:
            return error, None
    
    # Get GitHub instances
    github_client, issue_analyzer, code_generator, workflows, projects_manager = get_github_instances()
    
    # Check repository context if required
    if require_repo:
        error = check_repository_context(github_client)
        if error:
            return error, None
    
    # Check projects manager if projects are required
    if require_projects:
        error = check_projects_manager(projects_manager)
        if error:
            return error, None
    
    return None, (github_client, issue_analyzer, code_generator, workflows, projects_manager)


# GitHub MCP Tools
@mcp.tool()
def github_list_repositories(owner: Optional[str] = None) -> Dict[str, Any]:
    """
    List GitHub repositories for a user/organization.
    
    WHEN TO USE THIS TOOL:
    - User asks to "list my repositories" or "show repos"
    - Need to find a specific repository
    - Before switching to a repository
    - Exploring available projects
    - User asks "what repos do I have access to?"
    
    This tool automatically:
    - Fetches repositories from GitHub API
    - Shows public and private repos (based on auth)
    - Returns repository names, descriptions, and URLs
    - Lists recent activity information
    
    Args:
        owner: Repository owner (defaults to authenticated user)
        
    Returns:
        List of repository information
    """
    try:
        error, instances = validate_github_prerequisites()
        if error:
            return error
        github_client, _, _, _, _ = instances
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "switch to repo X" or "work on repository Y"
    - Before working with GitHub issues or PRs
    - Changing GitHub project context
    - After listing repositories
    - User provides "owner/repo" format
    
    This tool automatically:
    - Sets the active GitHub repository
    - Verifies repository access
    - Updates context for issue/PR operations
    - Returns repository details
    - Enables GitHub-specific features
    
    Args:
        owner: Repository owner
        repo: Repository name
        
    Returns:
        Repository information and switch status
    """
    try:
        error, instances = validate_github_prerequisites()
        if error:
            return error
        github_client, _, _, _, _ = instances
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "show issues" or "list bugs"
    - Looking for work items or tasks
    - User asks "what issues are open?"
    - Filtering issues by label or state
    - Before analyzing or working on issues
    
    This tool automatically:
    - Fetches issues from current repository
    - Filters by state (open/closed/all)
    - Filters by labels if specified
    - Returns issue titles, numbers, and metadata
    - Includes assignees and timestamps
    
    Args:
        state: Issue state (open, closed, all)
        labels: Filter by labels
        limit: Maximum number of issues
        
    Returns:
        List of issue information
    """
    try:
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, _, _ = instances
        
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
    
    WHEN TO USE THIS TOOL:
    - User asks about "issue #123" specifically
    - Need full details about an issue
    - Before analyzing or fixing an issue
    - Getting issue description and comments
    - User asks "what is issue X about?"
    
    This tool automatically:
    - Fetches complete issue details
    - Includes issue body and comments
    - Shows labels, assignees, and status
    - Returns creation and update timestamps
    - Provides full context for analysis
    
    Args:
        issue_number: Issue number
        
    Returns:
        Detailed issue information
    """
    try:
        # Validate prerequisites
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        
        github_client, _, _, _, _ = instances
        
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "create an issue" or "file a bug"
    - Documenting a problem or feature request
    - Creating work items or tasks
    - User provides issue title and description
    - Tracking TODOs or improvements
    
    This tool automatically:
    - Creates issue in current repository
    - Applies specified labels
    - Assigns to specified users
    - Returns issue number and URL
    - Enables tracking and collaboration
    
    Args:
        title: Issue title
        body: Issue description/body (optional)
        labels: List of label names to apply (optional)
        assignees: List of usernames to assign (optional)
        
    Returns:
        Created issue information
    """
    try:
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, _, _ = instances
        
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
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, _, _ = instances
        
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "analyze issue #X"
    - User asks "what is issue #X about?"
    - User asks for "issue analysis", "investigate issue", "understand issue"
    - User wants to know what code/files are related to an issue
    - ALWAYS use this instead of manual search when analyzing GitHub issues
    
    This tool automatically:
    - Fetches issue details and comments
    - Extracts errors, code references, and keywords
    - Performs optimized RAG searches with progressive context
    - Returns summarized analysis with recommendations
    
    Args:
        issue_number: Issue number to analyze
        
    Returns:
        Analysis results with search results and recommendations
    """
    try:
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, workflows, _ = instances
        
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
    
    WHEN TO USE THIS TOOL:
    - User asks to "suggest a fix for issue #X"
    - User asks "how to fix issue #X"
    - User asks for "fix suggestions", "solution", "implementation plan"
    - User wants code changes to resolve an issue
    - Use AFTER github_analyze_issue for best results
    
    This tool automatically:
    - Analyzes the issue with RAG search
    - Generates concrete fix suggestions
    - Provides implementation steps
    - Suggests code changes with context
    
    Args:
        issue_number: Issue number
        
    Returns:
        Fix suggestions and implementation plan
    """
    try:
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, workflows, _ = instances
        
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
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, _, _ = instances
        
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
        error, instances = validate_github_prerequisites(require_repo=True)
        if error:
            return error
        github_client, _, _, workflows, _ = instances
        
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


# GitHub Projects V2 tools (v0.3.4)
@mcp.tool()
def github_list_projects(owner: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """
    List GitHub Projects V2 for a user or organization.
    
    WHEN TO USE THIS TOOL:
    - User asks to "list projects" or "show projects"
    - User wants to see all available projects
    - Before working with a specific project
    - User asks "what projects do I have?"
    - Exploring project management capabilities
    
    This tool lists all GitHub Projects V2 for the specified owner,
    showing project titles, descriptions, item counts, and IDs.
    
    Args:
        owner: Username or organization (defaults to current repo owner)
        limit: Maximum number of projects to return (default: 20, max: 100)
        
    Returns:
        List of projects with details
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Use current repo owner if not specified
        if not owner:
            current_repo = github_client.get_current_repository()
            if current_repo:
                owner = current_repo.owner.login
            else:
                # Try to get authenticated user
                try:
                    user = github_client._github.get_user()
                    owner = user.login
                except:
                    return {
                        "error": "No owner specified",
                        "message": "Specify an owner or switch to a repository"
                    }
        
        # List projects
        projects = run_async_in_thread(
            projects_manager.list_projects(owner, limit)
        )
        
        console_logger.info(f"Listed {len(projects)} projects for {owner}")
        
        return {
            "success": True,
            "owner": owner,
            "count": len(projects),
            "projects": projects
        }
        
    except Exception as e:
        error_msg = f"Failed to list projects: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_create_project(title: str, body: Optional[str] = None, owner: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a new GitHub Project V2.
    
    Args:
        title: Project title
        body: Optional project description (Note: GitHub API currently doesn't support descriptions at creation)
        owner: Repository owner (defaults to current repo owner)
        
    Returns:
        Project information including ID and URL
        
    Note:
        The body parameter is accepted but not used due to GitHub API limitations.
        Consider adding a custom field after creation if descriptions are needed.
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Use current repo owner if not specified
        if not owner:
            current_repo = github_client.get_current_repository()
            if not current_repo:
                return {
                    "error": "No repository context set", 
                    "message": "Use github_switch_repository first or specify owner"
                }
            owner = current_repo.owner.login
        
        # Create project (async function needs to be run in event loop)
        project = run_async_in_thread(
            projects_manager.create_project(owner, title, body)
        )
        
        console_logger.info(f"Created GitHub project '{title}' for {owner}")
        
        response = {
            "success": True,
            "project": {
                "id": project["id"],
                "number": project["number"],
                "title": project["title"],
                "description": None,  # GitHub Projects V2 doesn't support descriptions at creation
                "url": project["url"],
                "owner": owner,
                "created_at": project["createdAt"]
            }
        }
        
        # Add note if description was requested
        if body:
            response["note"] = "GitHub Projects V2 API doesn't support descriptions at creation time. Consider adding a custom field after creation."
            
        return response
        
    except Exception as e:
        error_msg = f"Failed to create project: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_get_project(number: int, owner: Optional[str] = None) -> Dict[str, Any]:
    """
    Get GitHub Project V2 details.
    
    Args:
        number: Project number
        owner: Repository owner (defaults to current repo owner)
        
    Returns:
        Project details including fields and item counts
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Use current repo owner if not specified
        if not owner:
            current_repo = github_client.get_current_repository()
            if not current_repo:
                return {
                    "error": "No repository context set",
                    "message": "Use github_switch_repository first or specify owner"
                }
            owner = current_repo.owner.login
        
        # Get project details
        project = run_async_in_thread(
            projects_manager.get_project(owner, number)
        )
        
        console_logger.info(f"Retrieved project #{number} for {owner}")
        
        return {
            "success": True,
            "project": {
                "id": project["id"],
                "number": project["number"],
                "title": project["title"],
                "description": project.get("shortDescription", ""),
                "url": project["url"],
                "item_count": project["items"]["totalCount"],
                "fields": [
                    {
                        "id": field["id"],
                        "name": field["name"],
                        "type": field["dataType"],
                        "options": field.get("options", [])
                    }
                    for field in project["fields"]["nodes"]
                ],
                "created_at": project["createdAt"],
                "updated_at": project["updatedAt"]
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to get project: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_add_project_item(project_id: str, issue_number: int) -> Dict[str, Any]:
    """
    Add an issue or PR to a GitHub Project V2.
    
    Args:
        project_id: Project node ID
        issue_number: Issue or PR number from current repository
        
    Returns:
        Added item information
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True, require_repo=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        current_repo = github_client.get_current_repository()
        
        # Get the issue/PR to add
        try:
            issue = current_repo.get_issue(issue_number)
            content_id = issue.node_id
        except Exception:
            try:
                pr = current_repo.get_pull(issue_number)
                content_id = pr.node_id
            except Exception:
                return {
                    "error": f"Issue/PR #{issue_number} not found",
                    "message": "Check the issue/PR number"
                }
        
        # Add to project
        item = run_async_in_thread(
            projects_manager.add_item_to_project(project_id, content_id)
        )
        
        console_logger.info(f"Added {item['content']['title']} to project")
        
        return {
            "success": True,
            "item": {
                "id": item["id"],
                "type": item["type"],
                "content": {
                    "id": item["content"]["id"],
                    "number": item["content"]["number"],
                    "title": item["content"]["title"],
                    "url": item["content"]["url"]
                },
                "created_at": item["createdAt"]
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to add item to project: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_update_project_item(project_id: str, item_id: str, field_id: str, value: str) -> Dict[str, Any]:
    """
    Update a field value for a project item.
    
    Args:
        project_id: Project node ID
        item_id: Item node ID
        field_id: Field node ID
        value: New field value
        
    Returns:
        Update confirmation
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Update the field
        result = run_async_in_thread(
            projects_manager.update_item_field(project_id, item_id, field_id, value)
        )
        
        console_logger.info(f"Updated project item field {field_id} to '{value}'")
        
        return {
            "success": True,
            "item_id": result["id"],
            "field_id": field_id,
            "value": value
        }
        
    except Exception as e:
        error_msg = f"Failed to update item field: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_create_project_field(project_id: str, name: str, data_type: str, 
                               options: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Create a custom field in a GitHub Project V2.
    
    Args:
        project_id: Project node ID
        name: Field name
        data_type: Field type (TEXT, NUMBER, DATE, SINGLE_SELECT)
        options: For SINGLE_SELECT, list of {name, color} options
        
    Returns:
        Created field information
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Create the field
        field = run_async_in_thread(
            projects_manager.create_field(project_id, name, data_type, options)
        )
        
        console_logger.info(f"Created project field '{name}' with type {data_type}")
        
        return {
            "success": True,
            "field": {
                "id": field["id"],
                "name": field["name"],
                "type": field["dataType"],
                "options": field.get("options", [])
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to create project field: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_create_project_from_template(title: str, template: str, body: Optional[str] = None, 
                                        owner: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a GitHub Project V2 from a predefined template.
    
    Args:
        title: Project title
        template: Template name ('roadmap', 'bugs', 'features')
        body: Optional project description
        owner: Repository owner (defaults to current repo owner)
        
    Returns:
        Created project with configured fields
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Use current repo owner if not specified
        if not owner:
            current_repo = github_client.get_current_repository()
            if not current_repo:
                return {
                    "error": "No repository context set",
                    "message": "Use github_switch_repository first or specify owner"
                }
            owner = current_repo.owner.login
        
        # Create project from template
        project = run_async_in_thread(
            projects_manager.create_project_from_template(owner, title, template, body)
        )
        
        console_logger.info(f"Created project '{title}' from template '{template}'")
        
        return {
            "success": True,
            "project": {
                "id": project["id"],
                "number": project["number"],
                "title": project["title"],
                "description": project.get("shortDescription", ""),
                "url": project["url"],
                "owner": owner,
                "template": template,
                "fields": [
                    {
                        "id": field["id"],
                        "name": field["name"],
                        "type": field["dataType"],
                        "options": field.get("options", [])
                    }
                    for field in project.get("fields", [])
                ]
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to create project from template: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_get_project_status(project_id: str) -> Dict[str, Any]:
    """
    Get project status overview with item counts and progress.
    
    Args:
        project_id: Project node ID
        
    Returns:
        Project status dashboard with metrics
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Get detailed project status via GraphQL
        # Enhanced query for project status
        status_query = """
        query($projectId: ID!) {
            node(id: $projectId) {
                ... on ProjectV2 {
                    id
                    number
                    title
                    shortDescription
                    url
                    items(first: 100) {
                        totalCount
                        nodes {
                            type
                            content {
                                ... on Issue {
                                    state
                                    closed
                                }
                                ... on PullRequest {
                                    state
                                    merged
                                    closed
                                }
                            }
                        }
                    }
                    fields(first: 20) {
                        nodes {
                            ... on ProjectV2Field {
                                id
                                name
                                dataType
                            }
                        }
                    }
                }
            }
        }
        """
        
        # Handle async execution
        data = run_async_in_thread(
            projects_manager._execute_query(status_query, {"projectId": project_id})
        )
        
        project = data["node"]
        if not project:
            return {"error": "Project not found"}
        
        # Calculate statistics
        items = project["items"]["nodes"]
        total_items = project["items"]["totalCount"]
        
        issue_stats = {"open": 0, "closed": 0}
        pr_stats = {"open": 0, "closed": 0, "merged": 0}
        
        for item in items:
            if item["type"] == "ISSUE":
                if item["content"]["closed"]:
                    issue_stats["closed"] += 1
                else:
                    issue_stats["open"] += 1
            elif item["type"] == "PULL_REQUEST":
                if item["content"]["merged"]:
                    pr_stats["merged"] += 1
                elif item["content"]["closed"]:
                    pr_stats["closed"] += 1
                else:
                    pr_stats["open"] += 1
        
        console_logger.info(f"Retrieved status for project #{project['number']}")
        
        return {
            "success": True,
            "project": {
                "id": project["id"],
                "number": project["number"],
                "title": project["title"],
                "description": project.get("shortDescription", ""),
                "url": project["url"]
            },
            "statistics": {
                "total_items": total_items,
                "issues": issue_stats,
                "pull_requests": pr_stats,
                "completion_rate": round(
                    (issue_stats["closed"] + pr_stats["merged"]) / max(total_items, 1) * 100, 1
                ) if total_items > 0 else 0
            },
            "fields": [
                {
                    "id": field["id"],
                    "name": field["name"],
                    "type": field["dataType"]
                }
                for field in project["fields"]["nodes"]
                if field and "id" in field  # Skip empty field objects
            ]
        }
        
    except Exception as e:
        error_msg = f"Failed to get project status: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_delete_project(project_id: str) -> Dict[str, Any]:
    """
    Delete a GitHub Project V2.
    
    WHEN TO USE THIS TOOL:
    - User asks to "delete project" with a project ID
    - User wants to remove an entire project
    - Cleaning up test or temporary projects
    - Project is no longer needed
    
    This tool permanently deletes a GitHub Project V2. This action cannot be undone.
    Requires the project node ID (starts with PVT_).
    
    Args:
        project_id: Project node ID (must start with PVT_)
        
    Returns:
        Deletion status with project details
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        # Validate project ID format
        if not project_id.startswith("PVT_"):
            return {
                "error": "Projects manager not available",
                "message": "Failed to initialize GitHub Projects manager"
            }
        
        # Delete the project
        result = run_async_in_thread(
            projects_manager.delete_project(project_id)
        )
        
        console_logger.info(f"Deleted project {project_id}")
        
        return {
            "success": True,
            "deleted": result.get("deleted", True),
            "project_id": result.get("project_id"),
            "title": result.get("title", "Unknown"),
            "message": result.get("message", "Project deleted successfully")
        }
        
    except ValueError as e:
        error_msg = f"Invalid input: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Failed to delete project: {str(e)}"
        console_logger.error(error_msg)
        return {"error": error_msg}


@mcp.tool()
def github_smart_add_project_item(project_id: str, issue_number: int) -> Dict[str, Any]:
    """
    Add an issue to a project with intelligent field assignment.
    
    This tool analyzes the issue content and automatically assigns appropriate
    field values based on the issue title, body, and labels.
    
    Args:
        project_id: Project node ID
        issue_number: Issue number from current repository
        
    Returns:
        Item details with applied field assignments
    """
    try:
        error, instances = validate_github_prerequisites(require_projects=True, require_repo=True)
        if error:
            return error
        github_client, _, _, _, projects_manager = instances
        
        current_repo = github_client.get_current_repository()
        
        # Execute smart add with async handling
        result = run_async_in_thread(
            projects_manager.smart_add_issue_to_project(
                project_id, issue_number, current_repo
            )
        )
        
        console_logger.info(f"Smart added issue #{issue_number} to project with {len(result['applied_fields'])} fields set")
        
        return {
            "success": True,
            "item": {
                "id": result["item"]["id"],
                "type": result["item"]["type"],
                "issue_number": issue_number,
                "created_at": result["item"]["createdAt"]
            },
            "applied_fields": result["applied_fields"],
            "all_suggestions": result["suggestions"],
            "message": f"Added issue #{issue_number} with {len(result['applied_fields'])} fields automatically set"
        }
        
    except Exception as e:
        error_msg = f"Failed to smart add item to project: {str(e)}"
        console_logger.error(error_msg)
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


@mcp.tool()
def get_apple_silicon_status() -> Dict[str, Any]:
    """
    Get Apple Silicon optimization status and memory information
    
    Returns information about:
    - Whether running on Apple Silicon
    - MPS availability
    - Memory pressure status
    - Current memory limits
    """
    try:
        memory_manager = get_memory_manager()
        
        # Get basic Apple Silicon status
        status = memory_manager.get_apple_silicon_memory_status()
        
        # Add current memory limits
        status['memory_limits'] = {
            'total_mb': memory_manager.total_memory_limit_mb,
            'cleanup_threshold_mb': memory_manager.cleanup_threshold_mb,
            'aggressive_threshold_mb': memory_manager.aggressive_threshold_mb
        }
        
        # Add embeddings manager status if available
        try:
            embeddings_manager = get_embeddings_manager_instance()
            status['embeddings'] = {
                'device': embeddings_manager.device,
                'max_models_in_memory': embeddings_manager.max_models_in_memory,
                'memory_limit_gb': embeddings_manager.memory_limit_gb,
                'models_loaded': len(embeddings_manager.loaded_models),
                'total_memory_used_gb': embeddings_manager.total_memory_used_gb
            }
        except:
            pass
        
        # Add system memory info
        import psutil
        memory = psutil.virtual_memory()
        status['system_memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
        
        return status
        
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def trigger_apple_silicon_cleanup(level: str = "standard") -> Dict[str, Any]:
    """
    Manually trigger Apple Silicon memory cleanup
    
    Args:
        level: Cleanup level - "standard" or "aggressive"
        
    Returns:
        Cleanup results including before/after memory status
    """
    try:
        memory_manager = get_memory_manager()
        
        if not memory_manager.is_apple_silicon:
            return {"error": "Not running on Apple Silicon"}
        
        # Get before status
        import psutil
        before_memory = psutil.virtual_memory()
        before_process = psutil.Process().memory_info()
        
        # Perform cleanup
        memory_manager.perform_apple_silicon_cleanup(level)
        
        # Get after status
        after_memory = psutil.virtual_memory()
        after_process = psutil.Process().memory_info()
        
        return {
            "cleanup_level": level,
            "before": {
                "available_gb": before_memory.available / (1024**3),
                "process_gb": before_process.rss / (1024**3),
                "system_percent": before_memory.percent
            },
            "after": {
                "available_gb": after_memory.available / (1024**3),
                "process_gb": after_process.rss / (1024**3),
                "system_percent": after_memory.percent
            },
            "memory_freed": {
                "system_gb": (after_memory.available - before_memory.available) / (1024**3),
                "process_gb": (before_process.rss - after_process.rss) / (1024**3)
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


# Run the server
def initialize_bm25_indices():
    """Initialize BM25 indices for all existing collections on startup"""
    try:
        console_logger.info("Initializing BM25 indices...")
        client = get_qdrant_client()
        hybrid_searcher = get_hybrid_searcher()
        
        # Get all collections
        collections = client.get_collections().collections
        
        for collection in collections:
            collection_name = collection.name
            console_logger.info(f"Building BM25 index for {collection_name}...")
            
            # Fetch all documents from the collection
            all_docs = []
            offset = None
            limit = 100
            
            while True:
                points, next_offset = client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not points:
                    break
                    
                for point in points:
                    doc = {
                        "id": point.id,
                        "content": point.payload.get("content", ""),
                        "file_path": point.payload.get("file_path", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "language": point.payload.get("language", ""),
                        "doc_type": point.payload.get("doc_type", ""),
                        "heading": point.payload.get("heading", ""),
                        "chunk_type": point.payload.get("chunk_type", "")
                    }
                    all_docs.append(doc)
                
                offset = next_offset
                if not offset:
                    break
            
            # Update BM25 index
            if all_docs:
                hybrid_searcher.bm25_manager.update_index(collection_name, all_docs)
                console_logger.info(f"  Indexed {len(all_docs)} documents for {collection_name}")
            else:
                console_logger.info(f"  No documents found in {collection_name}")
                
        console_logger.info("BM25 initialization complete")
    except Exception as e:
        console_logger.error(f"Failed to initialize BM25 indices: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant RAG MCP Server")
    parser.add_argument("--watch", action="store_true", help="Enable file watching for auto-reindexing")
    parser.add_argument("--watch-dir", default=".", help="Directory to watch (default: current)")
    parser.add_argument("--debounce", type=float, default=3.0, help="Debounce seconds for file watching")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial index on startup")
    parser.add_argument("--client-cwd", help="Client's working directory (overrides MCP_CLIENT_CWD env var)")
    args = parser.parse_args()
    
    # Initialize memory manager early
    console_logger.info("🧠 Initializing memory manager...")
    memory_manager = get_memory_manager()
    memory_manager.start()
    
    # Check for Apple Silicon
    if memory_manager.is_apple_silicon:
        console_logger.info("🍎 Apple Silicon detected - memory optimizations enabled")
        console_logger.info(f"   Memory limits: {memory_manager.total_memory_limit_mb}MB total, "
                          f"{memory_manager.cleanup_threshold_mb}MB cleanup threshold")
        
        # Set MPS environment variables if not already set
        if not os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'):
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        if not os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO'):
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        if not os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO'):
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
    
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