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

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import watchdog for file watching
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logger.info("Watchdog not installed. File watching disabled. Install with: pip install watchdog")

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
                
            logger.info(f"🔄 Auto-reindexing {len(files)} changed files...")
            
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
                        logger.warning(f"Failed to index {file_path}: {result['error']}")
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    
            logger.info(f"✅ Auto-indexed {success}/{len(files)} files")

def get_current_project() -> Optional[Dict[str, str]]:
    """Detect current project based on working directory"""
    global _current_project
    
    # Check if we've already detected the project
    cwd = Path.cwd()
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
                return _current_project
    
    # No project found - use directory name as fallback
    _current_project = {
        "name": cwd.name,
        "root": str(cwd),
        "collection_prefix": f"dir_{cwd.name.replace(' ', '_').replace('-', '_')}"
    }
    return _current_project

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

def ensure_collection(collection_name: str):
    """Ensure a collection exists"""
    client = get_qdrant_client()
    
    from qdrant_client.http.models import Distance, VectorParams
    
    existing = [c.name for c in client.get_collections().collections]
    
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )

def get_qdrant_client():
    """Get or create Qdrant client"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        _qdrant_client = QdrantClient(host=host, port=port)
    
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
                logger.info(f"Cleared collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to clear {collection_name}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
    
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
def index_code(file_path: str, force_global: bool = False) -> Dict[str, Any]:
    """
    Index a source code file
    
    Args:
        file_path: Path to the file to index
        force_global: If True, index to global collection instead of project
    """
    try:
        from qdrant_client.http.models import PointStruct
        
        # Resolve to absolute path
        abs_path = Path(file_path).resolve()
        if not abs_path.exists():
            return {"error": f"File not found: {file_path}"}
        
        # Get services
        code_indexer = get_code_indexer()
        embedding_model = get_embedding_model()
        qdrant_client = get_qdrant_client()
        
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
            
            point = PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "file_path": str(abs_path),
                    "display_path": display_path,
                    "chunk_index": chunk.chunk_index,
                    "line_start": chunk.line_start,
                    "line_end": chunk.line_end,
                    "language": chunk.metadata.get("language", ""),
                    "content": chunk.content,
                    "chunk_type": chunk.metadata.get("chunk_type", "general"),
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
        
        return {
            "indexed": len(chunks),
            "file_path": display_path,
            "collection": collection_name,
            "language": chunks[0].metadata.get("language", "unknown") if chunks else "unknown",
            "project_context": current_project["name"] if current_project else "global"
        }
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

@mcp.tool()
def index_directory(directory: str = ".", patterns: List[str] = None, recursive: bool = True) -> Dict[str, Any]:
    """
    Index files in a directory (defaults to current directory)
    
    Args:
        directory: Directory to index (default: current directory)
        patterns: File patterns to include
        recursive: Whether to search recursively
    """
    try:
        if patterns is None:
            patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go", "*.rs", 
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
        
        # Resolve directory
        if directory == ".":
            dir_path = Path.cwd()
        else:
            dir_path = Path(directory).resolve()
        
        if not dir_path.exists():
            return {"error": f"Directory not found: {directory}"}
        
        # Get current project info
        current_project = get_current_project()
        
        indexed_files = []
        errors = []
        collections_used = set()
        
        # Process files
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
        return {
            "directory": directory,
            "cleared_collections": clear_result.get("cleared_collections", []),
            "indexed_files": index_result.get("indexed_files", []),
            "total_indexed": index_result.get("total", 0),
            "collections": index_result.get("collections", []),
            "project_context": current_project["name"] if current_project else "no project",
            "errors": index_result.get("errors"),
            "message": f"Reindexed {index_result.get('total', 0)} files after clearing {len(clear_result.get('cleared_collections', []))} collections"
        }
        
    except Exception as e:
        return {"error": str(e), "directory": directory}

@mcp.tool()
def search(query: str, n_results: int = 5, cross_project: bool = False) -> Dict[str, Any]:
    """
    Search indexed content (defaults to current project only)
    
    Args:
        query: Search query
        n_results: Number of results
        cross_project: If True, search across all projects (default: False)
    """
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
        
        for collection in search_collections:
            try:
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
            except Exception as e:
                # Skip if collection doesn't exist
                pass
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:n_results]
        
        # Get context info
        current_project = get_current_project()
        
        return {
            "results": all_results,
            "query": query,
            "total": len(all_results),
            "project_context": current_project["name"] if current_project else "no project",
            "search_scope": "all projects" if cross_project else "current project",
            "collections_searched": search_collections
        }
        
    except Exception as e:
        return {"error": str(e), "query": query}

@mcp.tool()
def search_code(query: str, language: Optional[str] = None, n_results: int = 5, cross_project: bool = False) -> Dict[str, Any]:
    """
    Search specifically in code files (defaults to current project)
    
    Args:
        query: Search query
        language: Filter by programming language
        n_results: Number of results
        cross_project: If True, search across all projects
    """
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
                        "file_path": payload.get("display_path", payload.get("file_path", "")),
                        "language": payload.get("language", ""),
                        "line_range": {
                            "start": payload.get("line_start", 0),
                            "end": payload.get("line_end", 0)
                        },
                        "content": payload.get("content", ""),
                        "chunk_type": payload.get("chunk_type", "general"),
                        "project": payload.get("project", "unknown")
                    })
            except:
                pass
        
        # Sort and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:n_results]
        
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
        
        return {
            "indexed": len(chunks),
            "file_path": display_path,
            "collection": collection_name,
            "file_type": chunks[0].metadata.get("file_type", "unknown") if chunks else "unknown",
            "project_context": current_project["name"] if current_project else "global"
        }
        
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

# Run the server
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant RAG MCP Server")
    parser.add_argument("--watch", action="store_true", help="Enable file watching for auto-reindexing")
    parser.add_argument("--watch-dir", default=".", help="Directory to watch (default: current)")
    parser.add_argument("--debounce", type=float, default=3.0, help="Debounce seconds for file watching")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial index on startup")
    args = parser.parse_args()
    
    # Start file watcher if requested
    observer = None
    if args.watch:
        if not WATCHDOG_AVAILABLE:
            logger.error("❌ Watchdog not installed. Install with: pip install watchdog")
            sys.exit(1)
            
        # Perform initial index if requested
        if args.initial_index:
            logger.info(f"📦 Performing initial index of {args.watch_dir}...")
            result = index_directory(args.watch_dir)
            logger.info(f"✅ Initial index complete: {result.get('total', 0)} files indexed")
        
        # Start file watcher
        event_handler = RagFileWatcher(debounce_seconds=args.debounce)
        observer = Observer()
        observer.schedule(event_handler, args.watch_dir, recursive=True)
        observer.start()
        logger.info(f"👀 File watcher started for {args.watch_dir}")
        logger.info(f"⏱️  Debounce: {args.debounce}s")
        
    try:
        # Run MCP server
        logger.info("🚀 Starting Qdrant RAG MCP Server...")
        if args.watch:
            logger.info("✅ Auto-reindexing enabled")
        mcp.run()
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
    finally:
        if observer:
            observer.stop()
            observer.join()
            logger.info("👋 File watcher stopped")