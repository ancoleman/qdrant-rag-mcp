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
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import version
try:
    from . import __version__
except ImportError:
    __version__ = "0.2.2"  # Fallback version

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
# Import configuration
from config import get_config

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
_current_project = None  # Cache current project detection

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
    for collection_type in ['code', 'config']:
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
        
        # Determine collection
        collection_name = get_collection_name(str(abs_path), "code")
        
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
            "total_chunks": len(formatted_chunks),
            "chunks": formatted_chunks,
            "full_content": full_content,
            "collection": collection_name
        }
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

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
        
        logger.info(f"Starting index_code for {file_path}", extra={
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
        
        # Get file modification time
        try:
            mod_time = abs_path.stat().st_mtime
        except:
            mod_time = time.time()  # Use current time as fallback
        
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
                "modified_at": mod_time
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
            for i, chunk in enumerate(chunks):
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
        
        logger.info(f"Completed index_code for {display_path}", extra={
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
        logger.error(f"Connection error during index_code for {file_path}: {str(e)}", extra={
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
        
        logger.error(f"Failed index_code for {file_path}: {error_msg}", extra={
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
                       ".gitignore", ".dockerignore", ".prettierrc*", ".eslintrc*", 
                       ".editorconfig", ".npmrc", ".yarnrc", ".ragignore"]
        
        # Define exclusion patterns
        exclude_dirs = {
            'node_modules', '__pycache__', '.git', '.venv', 'venv', 
            'env', '.env', 'dist', 'build', 'target', '.pytest_cache',
            '.mypy_cache', '.coverage', 'htmlcov', '.tox', 'data',
            'logs', 'tmp', 'temp', '.idea', '.vscode', '.vs',
            'qdrant_storage', 'models', '.cache'
        }
        
        exclude_patterns = {
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so', '*.dylib',
            '*.dll', '*.class', '*.log', '*.lock', '*.swp', '*.swo',
            '*.bak', '*.tmp', '*.temp', '*.old', '*.orig', '*.rej',
            '.env*', '*.sqlite', '*.db', '*.pid'
        }
        
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
        
        # Create a progress callback
        start_time = time.time()
        processed_count = 0
        
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
            logger.info(
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
                        if any(file_name.endswith(ext) for ext in ['.pyc', '.pyo', '.pyd', '.so', '.dylib', 
                                                                   '.dll', '.class', '.log', '.swp', '.swo',
                                                                   '.bak', '.tmp', '.temp', '.old', '.orig', 
                                                                   '.rej', '.sqlite', '.db', '.pid']):
                            continue
                        
                        if file_name.startswith('.env') or file_name == '.DS_Store':
                            continue
                        
                        # Skip lock files (but not .lock extension for other purposes)
                        if file_name in ['package-lock.json', 'yarn.lock', 'poetry.lock', 'Pipfile.lock', 
                                        'composer.lock', 'Gemfile.lock', 'Cargo.lock', 'uv.lock']:
                            continue
                        
                        files_to_process.append(file_path)
                    except Exception as e:
                        errors.append({"file": str(file_path), "error": str(e)})
        
        # Now process files with progress reporting
        total_files = len(files_to_process)
        if total_files > 0:
            logger = get_logger()
            logger.info(f"Starting to index {total_files} files from {directory}")
            
            # Report initial progress
            report_progress(0, total_files)
            
            for idx, file_path in enumerate(files_to_process):
                try:
                    # Determine file type
                    ext = file_path.suffix.lower()
                    if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                        result = index_config(str(file_path))
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
    force: bool = False
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
    
    Returns:
        Reindex results including what was cleared and indexed
    """
    logger = get_logger()
    start_time = time.time()
    
    logger.info(f"Starting reindex_directory for {directory}", extra={
        "operation": "reindex_directory",
        "directory": directory,
        "recursive": recursive,
        "force": force
    })
    
    try:
        # Get current project context
        current_project = get_current_project()
        if not current_project and not force:
            return {
                "error": "No project context found. Use force=True to reindex anyway.",
                "directory": directory
            }
        
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
            "cleared_collections": clear_result.get("cleared_collections", []),
            "indexed_files": index_result.get("indexed_files", []),
            "total_indexed": index_result.get("total", 0),
            "collections": index_result.get("collections", []),
            "project_context": current_project["name"] if current_project else "no project",
            "errors": index_result.get("errors"),
            "message": f"Reindexed {index_result.get('total', 0)} files after clearing {len(clear_result.get('cleared_collections', []))} collections"
        }
        
        logger.info(f"Completed reindex_directory for {directory}", extra={
            "operation": "reindex_directory",
            "directory": directory,
            "duration_ms": duration_ms,
            "files_indexed": index_result.get('total', 0),
            "collections_cleared": len(clear_result.get('cleared_collections', [])),
            "status": "success"
        })
        
        return result
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Failed reindex_directory for {directory}: {str(e)}", extra={
            "operation": "reindex_directory",
            "directory": directory,
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "directory": directory}

@mcp.tool()
def search(query: str, n_results: int = 5, cross_project: bool = False, search_mode: str = "hybrid", include_dependencies: bool = False, include_context: bool = True, context_chunks: int = 1) -> Dict[str, Any]:
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
    
    logger.info(f"Starting search: {query[:50]}...", extra={
        "operation": "search",
        "query_length": len(query),
        "n_results": n_results,
        "cross_project": cross_project,
        "search_mode": search_mode
    })
    
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
                    
                    # Fuse results using RRF
                    fused_results = hybrid_searcher.reciprocal_rank_fusion(
                        vector_results=vector_results,
                        bm25_results=bm25_results,
                        k=60,
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
        
        logger.info(f"Completed search: {query[:50]}...", extra={
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
        logger.error(f"Failed search: {query[:50]}... - {str(e)}", extra={
            "operation": "search",
            "query_length": len(query),
            "duration_ms": duration_ms,
            "error": str(e),
            "error_type": type(e).__name__,
            "status": "error"
        })
        return {"error": str(e), "query": query}

@mcp.tool()
def search_code(query: str, language: Optional[str] = None, n_results: int = 5, cross_project: bool = False, search_mode: str = "hybrid", include_dependencies: bool = False, include_context: bool = True, context_chunks: int = 1) -> Dict[str, Any]:
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
    """
    logger = get_logger()
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
                    "project": collection_name.rsplit('_', 1)[0]
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
            for i, chunk in enumerate(chunks):
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
    logger.info("Running health check")
    
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
    logger.info(
        f"Health check completed: {health_status['status']}",
        extra={
            "operation": "health_check",
            "status": health_status["status"],
            "qdrant_status": health_status["services"].get("qdrant", {}).get("status", "unknown"),
            "model_status": health_status["services"].get("embedding_model", {}).get("status", "unknown")
        }
    )
    
    return health_status

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