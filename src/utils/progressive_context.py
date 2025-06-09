"""Progressive Context Management for token-efficient search.

This module implements multi-level context retrieval with semantic caching
to reduce token consumption by 50-70% for high-level queries.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict
import time
import json
from pathlib import Path
import hashlib
from sklearn.metrics.pairwise import cosine_similarity

from utils.logging import get_project_logger
from utils.hybrid_search import get_hybrid_searcher
from utils.enhanced_ranker import get_enhanced_ranker
from utils.memory_manager import LRUMemoryCache, get_memory_manager


# Data structures
@dataclass
class QueryIntent:
    """Query intent classification result."""
    level: str  # "file", "class", "method"
    exploration_type: str  # "understanding", "debugging", "navigation"
    confidence: float  # 0.0-1.0


@dataclass
class ExpansionOption:
    """Options for expanding context to more detail."""
    target_level: str  # Next level down
    target_path: str  # Specific file/class/method to expand
    estimated_tokens: int  # Additional tokens if expanded
    relevance_score: float  # How relevant this expansion might be


@dataclass
class CodeHierarchy:
    """Hierarchical representation of code structure."""
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_file(self, file_path: str, summary: str = "", structure: Dict = None):
        """Add a file to the hierarchy."""
        self.files[file_path] = {
            "summary": summary,
            "structure": structure or {},
            "classes": {},
            "functions": []
        }
    
    def add_class(self, file_path: str, class_name: str, summary: str = "", methods: List[str] = None):
        """Add a class to a file in the hierarchy."""
        if file_path not in self.files:
            self.add_file(file_path)
        
        self.files[file_path]["classes"][class_name] = {
            "summary": summary,
            "methods": methods or []
        }
    
    def add_method(self, file_path: str, class_name: Optional[str], method_name: str, 
                   signature: str = "", summary: str = ""):
        """Add a method to the hierarchy."""
        if file_path not in self.files:
            self.add_file(file_path)
        
        method_info = {
            "name": method_name,
            "signature": signature,
            "summary": summary
        }
        
        if class_name and class_name in self.files[file_path]["classes"]:
            # Add to class methods
            if method_name not in self.files[file_path]["classes"][class_name]["methods"]:
                self.files[file_path]["classes"][class_name]["methods"].append(method_name)
        else:
            # Add as standalone function
            self.files[file_path]["functions"].append(method_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert hierarchy to dictionary."""
        return {"files": self.files}


@dataclass
class ProgressiveResult:
    """Result structure for progressive context retrieval."""
    level: str  # "file", "class", "method"
    results: List[Dict[str, Any]]  # Search results at this level
    summary: str  # High-level summary of results
    hierarchy: CodeHierarchy  # Structured code hierarchy
    expansion_options: List[ExpansionOption]  # Available drill-downs
    token_estimate: int  # Estimated tokens for this result
    token_reduction_percent: str  # e.g., "70%"
    from_cache: bool = False  # Whether result came from cache
    query_intent: Optional[QueryIntent] = None  # Classified query intent


class QueryIntentClassifier:
    """Classifies query intent to determine appropriate context level."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, collection_type: Optional[str] = None):
        """Initialize the query intent classifier."""
        self.config = config or {}
        self.collection_type = collection_type  # "code", "documentation", or None for general
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.fallback_level = self.config.get("fallback_level", "class")
        self.logger = get_project_logger()
        
        # Pattern definitions for different intent types
        self.patterns = {
            "file": {
                "keywords": [
                    "what does", "explain", "overview", "structure",
                    "architecture", "how does", "purpose of", "understand",
                    "describe", "summary", "high level", "big picture",
                    # Documentation-specific patterns
                    "changelog", "readme", "guide", "tutorial", "documentation",
                    "setup", "installation", "configuration", "usage"
                ],
                "exploration_type": "understanding"
            },
            "method": {
                "keywords": [
                    "implementation", "bug in", "error in", "fix",
                    "line", "specific", "exact", "debug", "issue",
                    "problem", "broken", "failing", "trace",
                    # Code-specific detailed patterns
                    "function", "method", "parameter", "return value"
                ],
                "exploration_type": "debugging"
            },
            "class": {
                "keywords": [
                    "find", "where is", "show me", "locate",
                    "search for", "looking for", "need", "want",
                    # Navigation patterns
                    "class", "module", "component", "service"
                ],
                "exploration_type": "navigation"
            }
        }
    
    def classify(self, query: str) -> QueryIntent:
        """Classify the query intent."""
        query_lower = query.lower()
        
        # Score each level based on keyword matches
        scores = {}
        for level, pattern_info in self.patterns.items():
            score = 0
            matched_keywords = []
            
            for keyword in pattern_info["keywords"]:
                if keyword in query_lower:
                    score += 1
                    matched_keywords.append(keyword)
            
            # Normalize score by number of keywords
            normalized_score = score / len(pattern_info["keywords"]) if pattern_info["keywords"] else 0
            scores[level] = (normalized_score, matched_keywords, pattern_info["exploration_type"])
        
        # Find the best match
        best_level = max(scores.keys(), key=lambda k: scores[k][0])
        best_score, matched_keywords, exploration_type = scores[best_level]
        
        # Calculate confidence based on score and keyword matches
        confidence = min(0.9, best_score * 2)  # Scale up but cap at 0.9
        if len(matched_keywords) > 2:
            confidence = 0.9
        elif len(matched_keywords) == 0:
            confidence = 0.6
            best_level = self.fallback_level
            exploration_type = "navigation"
        
        self.logger.debug(f"Query intent classification: {best_level} ({confidence:.2f})", extra={
            "query": query[:50],
            "level": best_level,
            "confidence": confidence,
            "matched_keywords": matched_keywords
        })
        
        return QueryIntent(
            level=best_level,
            exploration_type=exploration_type,
            confidence=confidence
        )


class SemanticCache(LRUMemoryCache):
    """Caches query results with semantic similarity matching."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, embedding_model=None):
        """Initialize the semantic cache."""
        # Get config from server config if not provided
        if config is None:
            from config import get_config
            server_config = get_config()
            progressive_config = server_config.get('progressive_context', {})
            config = progressive_config.get('cache', {})
        
        self.config = config
        
        # Get limits from memory management config
        memory_config = get_memory_manager().config
        component_limits = memory_config.get('component_limits', {}).get('progressive_cache', {})
        
        max_memory = float(component_limits.get('max_memory_mb', config.get('max_memory_mb', 200)))
        max_items = int(component_limits.get('max_items', config.get('max_cache_size', 100)))
        
        # Initialize parent LRUMemoryCache
        super().__init__(
            name="progressive_semantic_cache",
            max_memory_mb=max_memory,
            max_items=max_items
        )
        
        # Register with memory manager
        memory_manager = get_memory_manager()
        memory_manager.register_component("progressive_cache", self)
        
        # Semantic cache specific settings
        self.similarity_threshold = float(config.get("similarity_threshold", 0.85))
        self.ttl_seconds = int(config.get("ttl_seconds", 1800))
        self.persistence_enabled = config.get("persistence_enabled", True)
        self.persistence_path = config.get("persistence_path", "~/.mcp-servers/qdrant-rag/progressive_cache")
        
        # Embeddings cache for semantic matching
        self.embeddings_cache = OrderedDict()
        self.embedding_model = embedding_model
        self.logger = get_project_logger()
        
        # Load persisted cache if enabled
        if self.persistence_enabled:
            self._load_cache()
    
    def set_embedding_model(self, embedding_model):
        """Set the embedding model."""
        self.embedding_model = embedding_model
    
    def _get_embedding_model(self):
        """Get the embedding model."""
        if self.embedding_model is None:
            raise ValueError("Embedding model not set. Call set_embedding_model() first.")
        return self.embedding_model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        # Create a hash of the text for caching
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # Generate embedding
        model = self._get_embedding_model()
        
        # Determine content type from the cache key format (level:query)
        content_type = 'general'
        if text.startswith('file:') or text.startswith('class:') or text.startswith('method:'):
            # Extract potential collection suffix hints from the query
            if '_code' in text or 'function' in text.lower() or 'class' in text.lower():
                content_type = 'code'
            elif '_config' in text or 'config' in text.lower() or 'yaml' in text.lower() or 'json' in text.lower():
                content_type = 'config'
            elif '_documentation' in text or 'doc' in text.lower() or 'readme' in text.lower():
                content_type = 'documentation'
        
        # Use appropriate content type for encoding
        if hasattr(model, 'encode') and callable(getattr(model, 'encode')):
            try:
                embedding = model.encode(text, content_type=content_type)
            except TypeError:
                # Fallback if the model doesn't support content_type parameter
                embedding = model.encode(text)
        else:
            embedding = model.encode(text)
        
        # Cache the embedding
        self.embeddings_cache[text_hash] = embedding
        
        # Limit embeddings cache size (more aggressive limit)
        if len(self.embeddings_cache) > self.max_cache_size:
            # Remove oldest quarter
            items_to_remove = list(self.embeddings_cache.keys())[:len(self.embeddings_cache)//4]
            for key in items_to_remove:
                del self.embeddings_cache[key]
        
        return embedding
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry is expired."""
        return time.time() - timestamp > self.ttl_seconds
    
    def get_memory_usage(self) -> float:
        """Override parent to include embeddings cache memory"""
        # Get base cache memory from parent
        base_memory = super().get_memory_usage()
        
        # Add embeddings cache memory (768 or 1024 dims * 4 bytes * count)
        avg_embedding_size = 768 * 4 / (1024**2)  # ~3KB per embedding
        embeddings_size = len(self.embeddings_cache) * avg_embedding_size
        
        return base_memory + embeddings_size
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        expired_keys = []
        
        with self.lock:
            for key, value in self.cache.items():
                if isinstance(value, tuple) and len(value) >= 3:
                    _, _, timestamp = value
                    if self._is_expired(timestamp):
                        expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            if key in self.memory_estimates:
                del self.memory_estimates[key]
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def cleanup(self, aggressive: bool = False) -> int:
        """Override parent cleanup to handle embeddings cache and TTL"""
        removed = 0
        
        # First clean up expired entries
        self._cleanup_expired()
        
        # Then do standard LRU cleanup
        removed = super().cleanup(aggressive)
        
        # Also clean embeddings cache
        if aggressive and len(self.embeddings_cache) > 20:
            # Remove half of embeddings
            items_to_remove = len(self.embeddings_cache) // 2
            keys_to_remove = list(self.embeddings_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.embeddings_cache[key]
            removed += items_to_remove
        elif len(self.embeddings_cache) > 50:
            # Remove 20% of embeddings
            items_to_remove = len(self.embeddings_cache) // 5
            keys_to_remove = list(self.embeddings_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.embeddings_cache[key]
            removed += items_to_remove
        
        return removed
    
    def get_similar(self, query: str, level: str) -> Optional[ProgressiveResult]:
        """Find semantically similar cached queries."""
        # Create cache key that includes level
        cache_key = f"{level}:{query}"
        query_embedding = self._get_embedding(cache_key)
        
        best_match = None
        best_similarity = 0.0
        best_key = None
        
        with self.lock:
            for cached_key in list(self.cache.keys()):
                # Get the stored data
                cache_data = self.cache[cached_key]
                if not isinstance(cache_data, tuple) or len(cache_data) < 3:
                    continue
                    
                result, embedding, timestamp = cache_data
                
                # Skip expired entries
                if self._is_expired(timestamp):
                    continue
                
                # Only match same level
                if not cached_key.startswith(f"{level}:"):
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_match = result
                    best_similarity = similarity
                    best_key = cached_key
        
        if best_match and best_key:
            # Update access time using parent's get method
            _ = self.get(best_key)  # This moves it to end
            self.logger.info(f"Cache hit for query with similarity {best_similarity:.3f}", extra={
                "query": query[:50],
                "level": level,
                "similarity": best_similarity
            })
            # Mark result as from cache
            best_match.from_cache = True
            return best_match
        
        return None
    
    def store(self, query: str, level: str, result: ProgressiveResult):
        """Store a query result in the cache."""
        # Create cache key that includes level
        cache_key = f"{level}:{query}"
        embedding = self._get_embedding(cache_key)
        
        # Calculate memory estimate based on result size
        import sys
        memory_estimate_mb = (
            sys.getsizeof(result.results) / (1024**2) +
            sys.getsizeof(result.summary) / (1024**2) +
            0.001  # Small overhead for other fields
        )
        
        # Store using parent's put method with tuple value
        cache_value = (result, embedding, time.time())
        self.put(cache_key, cache_value, memory_estimate_mb)
        
        # Store embedding separately
        self.embeddings_cache[cache_key] = embedding
        
        # Persist if enabled
        if self.persistence_enabled:
            self._save_cache()
        
        self.logger.debug(f"Cached query result", extra={
            "query": query[:50],
            "level": level,
            "cache_size": self.get_item_count()
        })
    
    def _load_cache(self):
        """Load persisted cache from disk."""
        cache_path = Path(self.persistence_path).expanduser()
        cache_file = cache_path / "semantic_cache.json"
        
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Reconstruct cache with proper objects
            for key, value in cache_data.items():
                # Skip expired entries
                if self._is_expired(value["timestamp"]):
                    continue
                
                # Reconstruct ProgressiveResult
                result = ProgressiveResult(
                    level=value["result"]["level"],
                    results=value["result"]["results"],
                    summary=value["result"]["summary"],
                    hierarchy=CodeHierarchy(files=value["result"]["hierarchy"]["files"]),
                    expansion_options=[
                        ExpansionOption(**opt) for opt in value["result"]["expansion_options"]
                    ],
                    token_estimate=value["result"]["token_estimate"],
                    token_reduction_percent=value["result"]["token_reduction_percent"],
                    from_cache=True
                )
                
                # Reconstruct embedding
                embedding = np.array(value["embedding"])
                
                self.cache[key] = (result, embedding, value["timestamp"])
            
            self.logger.info(f"Loaded {len(self.cache)} cached entries from disk")
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Persist cache to disk."""
        cache_path = Path(self.persistence_path).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / "semantic_cache.json"
        
        try:
            # Convert cache to JSON-serializable format
            cache_data = {}
            for key, (result, embedding, timestamp) in self.cache.items():
                cache_data[key] = {
                    "result": {
                        "level": result.level,
                        "results": result.results,
                        "summary": result.summary,
                        "hierarchy": result.hierarchy.to_dict(),
                        "expansion_options": [
                            {
                                "target_level": opt.target_level,
                                "target_path": opt.target_path,
                                "estimated_tokens": opt.estimated_tokens,
                                "relevance_score": opt.relevance_score
                            }
                            for opt in result.expansion_options
                        ],
                        "token_estimate": result.token_estimate,
                        "token_reduction_percent": result.token_reduction_percent
                    },
                    "embedding": embedding.tolist(),
                    "timestamp": timestamp
                }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def clear_expired(self):
        """Remove expired entries from cache."""
        keys_to_remove = []
        
        for key, (_, _, timestamp) in self.cache.items():
            if self._is_expired(timestamp):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        if keys_to_remove:
            self.logger.info(f"Cleared {len(keys_to_remove)} expired cache entries")
            if self.persistence_enabled:
                self._save_cache()


class HierarchyBuilder:
    """Builds hierarchical code structure from search results."""
    
    def __init__(self):
        """Initialize the hierarchy builder."""
        self.logger = get_project_logger()
    
    def build(self, search_results: List[Dict[str, Any]]) -> CodeHierarchy:
        """Build file → class → method hierarchy from search results."""
        hierarchy = CodeHierarchy()
        
        # Group results by file
        file_groups = {}
        for result in search_results:
            file_path = result.get("file_path", "")
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)
        
        # Process each file
        for file_path, file_results in file_groups.items():
            # Extract file-level summary
            file_summary = self._generate_file_summary(file_results)
            hierarchy.add_file(file_path, summary=file_summary)
            
            # Process chunks in this file
            for result in file_results:
                chunk_type = result.get("chunk_type", "code")
                metadata = result.get("metadata", {})
                
                if chunk_type == "class":
                    # Add class to hierarchy
                    class_name = metadata.get("name", "UnknownClass")
                    class_summary = self._extract_summary(result)
                    methods = metadata.get("methods", [])
                    
                    hierarchy.add_class(
                        file_path,
                        class_name,
                        summary=class_summary,
                        methods=methods
                    )
                    
                elif chunk_type in ["function", "method"]:
                    # Add method/function to hierarchy
                    method_name = metadata.get("name", "unknown_function")
                    parent_class = metadata.get("parent_class")
                    signature = metadata.get("signature", "")
                    method_summary = self._extract_summary(result)
                    
                    hierarchy.add_method(
                        file_path,
                        parent_class,
                        method_name,
                        signature=signature,
                        summary=method_summary
                    )
        
        return hierarchy
    
    def _generate_file_summary(self, file_results: List[Dict[str, Any]]) -> str:
        """Generate a summary for a file based on its contents."""
        # Count different types of elements
        classes = sum(1 for r in file_results if r.get("chunk_type") == "class")
        functions = sum(1 for r in file_results if r.get("chunk_type") in ["function", "method"])
        
        # Extract main purpose from content
        content_snippets = []
        for result in file_results[:3]:  # Look at first few chunks
            content = result.get("content", "")
            if content:
                # Take first line or docstring
                lines = content.strip().split('\n')
                for line in lines[:5]:
                    if line.strip() and not line.strip().startswith(('#', '//', '/*')):
                        content_snippets.append(line.strip())
                        break
        
        summary_parts = []
        if classes > 0:
            summary_parts.append(f"{classes} class{'es' if classes > 1 else ''}")
        if functions > 0:
            summary_parts.append(f"{functions} function{'s' if functions > 1 else ''}")
        
        summary = f"Contains {', '.join(summary_parts)}" if summary_parts else "Code file"
        
        return summary
    
    def _extract_summary(self, result: Dict[str, Any]) -> str:
        """Extract a summary from a search result."""
        metadata = result.get("metadata", {})
        
        # Try to get docstring
        docstring = metadata.get("docstring", "")
        if docstring:
            # Take first line of docstring
            return docstring.split('\n')[0].strip()
        
        # Try to get from content
        content = result.get("content", "")
        if content:
            lines = content.strip().split('\n')
            # Look for docstring in content
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    # Found docstring start
                    if line.count('"""') >= 2 or line.count("'''") >= 2:
                        # Single line docstring
                        return line.strip().strip('"""').strip("'''").strip()
                    else:
                        # Multi-line docstring
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                # Found end
                                return lines[i + 1].strip() if i + 1 < j else ""
        
        # Default based on type
        chunk_type = result.get("chunk_type", "code")
        name = metadata.get("name", "unknown")
        return f"{chunk_type.title()} {name}"


class ProgressiveContextManager:
    """Manages multi-level context retrieval and caching."""
    
    def __init__(self, qdrant_client, embeddings, config: Optional[Dict[str, Any]] = None):
        """Initialize the progressive context manager."""
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.config = config or {}
        
        # Initialize components
        self.semantic_cache = SemanticCache(self.config.get("cache", {}), embeddings)
        self.hierarchy_builder = HierarchyBuilder()
        self.query_classifier = QueryIntentClassifier(self.config.get("query_classification", {}))
        
        self.logger = get_project_logger()
        
        # Level configurations
        self.level_configs = self.config.get("levels", {})
        self.default_level = self.config.get("default_level", "auto")
        
        # Token estimation factors (rough estimates)
        self.token_factors = {
            "file": 0.3,    # 70% reduction
            "class": 0.5,   # 50% reduction
            "method": 0.8,  # 20% reduction
            "full": 1.0     # No reduction
        }
    
    def get_progressive_context(
        self,
        query: str,
        level: str = "auto",
        n_results: int = 5,
        cross_project: bool = False,
        search_mode: str = "hybrid",
        include_dependencies: bool = False,
        semantic_cache: bool = True,
        collection_suffix: Optional[str] = None
    ) -> ProgressiveResult:
        """
        Get context at specified level with expansion options.
        
        Args:
            query: Search query
            level: Context level ("auto", "file", "class", "method", "full")
            n_results: Number of results to return
            cross_project: Whether to search across all projects
            search_mode: Search mode ("vector", "keyword", "hybrid")
            include_dependencies: Whether to include dependent files
            semantic_cache: Whether to use semantic caching
            collection_suffix: Optional suffix to filter collections (e.g., "_documentation", "_code")
        
        Returns:
            ProgressiveResult with context at requested level
        """
        # Classify query intent if level is auto
        query_intent = None
        if level == "auto":
            # For documentation searches, default to file level since docs don't have classes/methods
            if collection_suffix == "_documentation":
                level = "file"
                query_intent = QueryIntent(
                    level="file",
                    exploration_type="understanding",
                    confidence=0.9
                )
                self.logger.info(f"Auto-detected context level for docs: {level}", extra={
                    "query": query[:50],
                    "level": level,
                    "collection_type": "documentation"
                })
            else:
                query_intent = self.query_classifier.classify(query)
                level = query_intent.level
                self.logger.info(f"Auto-detected context level: {level}", extra={
                    "query": query[:50],
                    "level": level,
                    "confidence": query_intent.confidence
                })
        
        # Check semantic cache if enabled
        if semantic_cache:
            cached_result = self.semantic_cache.get_similar(query, level)
            if cached_result:
                cached_result.query_intent = query_intent
                return cached_result
        
        # Perform search at requested level
        search_results = self._search_at_level(
            query, level, n_results, cross_project, search_mode, include_dependencies, collection_suffix
        )
        
        # Build hierarchical structure
        hierarchy = self.hierarchy_builder.build(search_results)
        
        # Generate summary based on level
        summary = self._generate_summary(search_results, level)
        
        # Calculate token estimates
        token_estimate = self._estimate_tokens(search_results, level)
        base_tokens = self._estimate_tokens(search_results, "full")
        reduction_percent = f"{int((1 - token_estimate / base_tokens) * 100)}%" if base_tokens > 0 else "0%"
        
        # Generate expansion options
        expansion_options = self._get_expansion_options(hierarchy, level)
        
        # Create result
        progressive_result = ProgressiveResult(
            level=level,
            results=search_results,
            summary=summary,
            hierarchy=hierarchy,
            expansion_options=expansion_options,
            token_estimate=token_estimate,
            token_reduction_percent=reduction_percent,
            from_cache=False,
            query_intent=query_intent
        )
        
        # Store in cache if enabled
        if semantic_cache:
            self.semantic_cache.store(query, level, progressive_result)
        
        return progressive_result
    
    def _search_at_level(
        self,
        query: str,
        level: str,
        n_results: int,
        cross_project: bool,
        search_mode: str,
        include_dependencies: bool,
        collection_suffix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform search and filter results based on context level."""
        # Determine content type based on collection suffix
        content_type = 'general'
        if collection_suffix == '_code':
            content_type = 'code'
        elif collection_suffix == '_config':
            content_type = 'config'
        elif collection_suffix == '_documentation':
            content_type = 'documentation'
        
        # Generate query embedding with appropriate content type
        query_embedding_array = self.embeddings.encode(query, content_type=content_type)
        # Handle both 1D and 2D arrays - if 2D with single row, extract it
        if hasattr(query_embedding_array, 'ndim'):
            if query_embedding_array.ndim == 2 and query_embedding_array.shape[0] == 1:
                query_embedding = query_embedding_array[0].tolist()
            else:
                query_embedding = query_embedding_array.tolist()
        else:
            # Fallback for non-numpy arrays
            query_embedding = query_embedding_array if isinstance(query_embedding_array, list) else query_embedding_array.tolist()
        
        # Determine which collections to search
        all_collections = [c.name for c in self.qdrant_client.get_collections().collections]
        
        # Apply collection suffix filter if specified
        if collection_suffix:
            all_collections = [c for c in all_collections if c.endswith(collection_suffix)]
        
        if cross_project:
            search_collections = all_collections
        else:
            # Get current project collections
            from qdrant_mcp_context_aware import get_current_project
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
        
        # Adjust n_results based on level to get enough data for summarization
        search_limit = n_results * 3 if level in ["file", "class"] else n_results
        
        # Search across collections
        all_results = []
        hybrid_searcher = get_hybrid_searcher()
        
        for collection in search_collections:
            try:
                if search_mode == "hybrid" and hybrid_searcher:
                    # Get vector search results first
                    vector_results = []
                    vector_scores_map = {}
                    result_objects_map = {}  # Store original result objects
                    
                    search_results = self.qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        limit=search_limit * 2  # Get more for fusion
                    )
                    
                    for result in search_results:
                        doc_id = f"{result.payload['file_path']}_{result.payload.get('chunk_index', 0)}"
                        score = float(result.score)
                        vector_results.append((doc_id, score))
                        vector_scores_map[doc_id] = score
                        result_objects_map[doc_id] = result  # Store the original result object
                    
                    # Get BM25 results (pass qdrant_client for on-demand building)
                    bm25_results = hybrid_searcher.bm25_manager.search(
                        collection_name=collection,
                        query=query,
                        k=search_limit * 2,
                        qdrant_client=self.qdrant_client
                    )
                    bm25_scores_map = {doc_id: score for doc_id, score in bm25_results}
                    
                    # Determine search type from collection suffix
                    search_type = "code" if collection_suffix == "code" else "documentation" if collection_suffix == "documentation" else "general"
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
                    
                    # Use original result objects with updated scores
                    results = []
                    for fused_result in fused_results[:search_limit]:
                        doc_id = fused_result.content  # doc_id is stored in content field
                        
                        # Check if we have the original result object
                        if doc_id in result_objects_map:
                            result = result_objects_map[doc_id]
                            # Override score with fused score
                            result.score = fused_result.combined_score
                            # Store individual scores in payload for transparency
                            result.payload['vector_score'] = vector_scores_map.get(doc_id, 0.0)
                            result.payload['bm25_score'] = bm25_scores_map.get(doc_id, 0.0)
                            results.append(result)
                        else:
                            # This is a BM25-only result, we need to fetch it
                            parts = doc_id.rsplit('_', 1)
                            if len(parts) == 2:
                                file_path = parts[0]
                                chunk_index = int(parts[1])
                                
                                # Fetch full document from Qdrant
                                from qdrant_client.models import Filter, FieldCondition, MatchValue
                                filter_conditions = Filter(
                                    must=[
                                        FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                                        FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index))
                                    ]
                                )
                                
                                fetch_results = self.qdrant_client.search(
                                    collection_name=collection,
                                    query_vector=query_embedding,  # Use actual query vector
                                    query_filter=filter_conditions,
                                    limit=1
                                )
                                
                                if fetch_results:
                                    result = fetch_results[0]
                                    # Override score with fused score
                                    result.score = fused_result.combined_score
                                    # Store individual scores in payload for transparency
                                    result.payload['vector_score'] = vector_scores_map.get(doc_id, 0.0)
                                    result.payload['bm25_score'] = bm25_scores_map.get(doc_id, 0.0)
                                    results.append(result)
                
                elif search_mode == "keyword":
                    # BM25 keyword search only
                    bm25_results = hybrid_searcher.bm25_manager.search(
                        collection_name=collection,
                        query=query,
                        k=search_limit,
                        qdrant_client=self.qdrant_client
                    )
                    
                    # Fetch full documents from Qdrant
                    results = []
                    for doc_id, score in bm25_results:
                        parts = doc_id.rsplit('_', 1)
                        if len(parts) == 2:
                            file_path = parts[0]
                            chunk_index = int(parts[1])
                            
                            from qdrant_client.models import Filter, FieldCondition, MatchValue
                            filter_conditions = Filter(
                                must=[
                                    FieldCondition(key="file_path", match=MatchValue(value=file_path)),
                                    FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index))
                                ]
                            )
                            
                            fetch_results = self.qdrant_client.search(
                                collection_name=collection,
                                query_vector=query_embedding,  # Use actual query vector
                                query_filter=filter_conditions,
                                limit=1
                            )
                            
                            if fetch_results:
                                result = fetch_results[0]
                                result.score = score
                                results.append(result)
                
                else:
                    # Vector search only
                    results = self.qdrant_client.search(
                        collection_name=collection,
                        query_vector=query_embedding,
                        limit=search_limit
                    )
                
                # Process results based on level
                for result in results:
                    if hasattr(result, 'payload'):
                        payload = result.payload
                        score = result.score
                    else:
                        payload = result
                        score = result.get('score', 0.0)
                    
                    # Apply level-specific filtering
                    if level == "file":
                        # For file level, we want to summarize by file
                        # so we'll include all chunks but mark them for summarization
                        result_dict = {
                            "score": score,
                            "collection": collection,
                            "file_path": payload.get("file_path", ""),
                            "chunk_type": payload.get("chunk_type", "code"),
                            "content": payload.get("content", ""),
                            "metadata": payload.get("metadata", {}),
                            "_summarize": True
                        }
                    elif level == "class":
                        # For class level, include class and method signatures
                        chunk_type = payload.get("chunk_type", "code")
                        if chunk_type in ["class", "function", "method"]:
                            # Include but truncate content
                            content = payload.get("content", "")
                            # Extract signature/definition only
                            lines = content.split('\n')
                            signature_lines = []
                            for line in lines[:10]:  # Look at first 10 lines
                                signature_lines.append(line)
                                if line.strip().endswith(':'):  # Found end of signature
                                    break
                            
                            result_dict = {
                                "score": score,
                                "collection": collection,
                                "file_path": payload.get("file_path", ""),
                                "chunk_type": chunk_type,
                                "content": '\n'.join(signature_lines),
                                "metadata": payload.get("metadata", {}),
                                "_truncated": True
                            }
                        else:
                            continue  # Skip non-class/method chunks
                    else:
                        # Method/full level - include full content
                        result_dict = {
                            "score": score,
                            "collection": collection,
                            "file_path": payload.get("file_path", ""),
                            "chunk_type": payload.get("chunk_type", "code"),
                            "content": payload.get("content", ""),
                            "metadata": payload.get("metadata", {})
                        }
                    
                    all_results.append(result_dict)
                    
            except Exception as e:
                self.logger.warning(f"Search failed for collection {collection}: {e}")
        
        # Apply enhanced ranking
        enhanced_ranker = get_enhanced_ranker()
        if enhanced_ranker and all_results:
            # Get current project context for ranking
            from qdrant_mcp_context_aware import get_current_project
            query_context = get_current_project()
            
            # Get dependency graph if available
            dependency_graph = None
            if include_dependencies:
                try:
                    from utils.dependency_resolver import DependencyResolver
                    # Use the first collection to get dependency graph
                    if search_collections:
                        resolver = DependencyResolver(self.qdrant_client, search_collections[0])
                        resolver.load_dependencies_from_collection()
                        dependency_graph = resolver.graph
                except:
                    pass
            
            # Apply enhanced ranking with full context
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
            # For non-enhanced results, just sort by existing score
            all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Post-process results based on level
        if level == "file":
            # Group by file and create summaries
            file_groups = {}
            for result in all_results:
                file_path = result.get("file_path", "")
                if file_path not in file_groups:
                    file_groups[file_path] = []
                file_groups[file_path].append(result)
            
            # Create one result per file with summary
            final_results = []
            for file_path, file_results in list(file_groups.items())[:n_results]:
                # Use the highest scoring chunk as the representative
                best_result = file_results[0]
                best_result["_file_chunks"] = len(file_results)
                best_result["content"] = f"File contains {len(file_results)} relevant sections"
                final_results.append(best_result)
            
            return final_results
        else:
            # Return top n_results
            return all_results[:n_results]
    
    def _generate_summary(self, results: List[Dict[str, Any]], level: str) -> str:
        """Generate a summary of results appropriate for the context level."""
        if not results:
            return "No results found."
        
        if level == "file":
            # File-level summary
            files = set(r.get("file_path", "") for r in results)
            return f"Found relevant content in {len(files)} file{'s' if len(files) != 1 else ''}: " + \
                   ", ".join(list(files)[:3]) + ("..." if len(files) > 3 else "")
        
        elif level == "class":
            # Class-level summary
            classes = []
            for r in results:
                if r.get("chunk_type") == "class":
                    class_name = r.get("metadata", {}).get("name", "Unknown")
                    file_path = r.get("file_path", "")
                    classes.append(f"{class_name} ({file_path})")
            
            if classes:
                return f"Found {len(classes)} relevant class{'es' if len(classes) != 1 else ''}: " + \
                       ", ".join(classes[:3]) + ("..." if len(classes) > 3 else "")
            else:
                return "Found relevant code sections."
        
        else:
            # Method/full level summary
            return f"Found {len(results)} relevant code section{'s' if len(results) != 1 else ''}."
    
    def _estimate_tokens(self, results: List[Dict[str, Any]], level: str) -> int:
        """Estimate token count for results at given level."""
        if not results:
            return 0
        
        # Base estimation on content length
        total_chars = sum(len(r.get("content", "")) for r in results)
        
        # Rough token estimation (1 token ≈ 4 characters)
        base_tokens = total_chars // 4
        
        # Apply level factor
        factor = self.token_factors.get(level, 1.0)
        estimated_tokens = int(base_tokens * factor)
        
        return estimated_tokens
    
    def _get_expansion_options(
        self,
        hierarchy: CodeHierarchy,
        current_level: str
    ) -> List[ExpansionOption]:
        """Generate expansion options for drilling down to more detail."""
        options = []
        
        if current_level == "file":
            # Can expand to class level
            for file_path, file_info in hierarchy.files.items():
                for class_name in file_info["classes"]:
                    options.append(ExpansionOption(
                        target_level="class",
                        target_path=f"{file_path}::{class_name}",
                        estimated_tokens=800,  # Rough estimate
                        relevance_score=0.8  # Would be calculated based on search scores
                    ))
        
        elif current_level == "class":
            # Can expand to method level
            for file_path, file_info in hierarchy.files.items():
                for class_name, class_info in file_info["classes"].items():
                    for method_name in class_info["methods"]:
                        options.append(ExpansionOption(
                            target_level="method",
                            target_path=f"{file_path}::{class_name}::{method_name}",
                            estimated_tokens=400,  # Rough estimate
                            relevance_score=0.7  # Would be calculated based on search scores
                        ))
        
        # Sort by relevance and limit
        options.sort(key=lambda x: x.relevance_score, reverse=True)
        return options[:10]  # Limit to top 10 expansion options


# Module-level instance management
_progressive_manager = None
_query_classifier = None


def get_progressive_manager(qdrant_client, embeddings, config: Optional[Dict[str, Any]] = None) -> ProgressiveContextManager:
    """Get or create the progressive context manager instance."""
    global _progressive_manager
    if _progressive_manager is None:
        _progressive_manager = ProgressiveContextManager(qdrant_client, embeddings, config)
    return _progressive_manager


def get_query_classifier(config: Optional[Dict[str, Any]] = None) -> QueryIntentClassifier:
    """Get or create the query intent classifier instance."""
    global _query_classifier
    if _query_classifier is None:
        _query_classifier = QueryIntentClassifier(config)
    return _query_classifier