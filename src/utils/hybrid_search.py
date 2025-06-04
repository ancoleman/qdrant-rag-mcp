"""
Hybrid search implementation combining BM25 keyword search with vector search.

This module provides functionality to perform hybrid search by combining
traditional BM25 keyword-based search with semantic vector search for
improved retrieval precision.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import re
from typing import Callable

# Use project-aware logging
try:
    from utils.logging import get_project_logger
    logger = get_project_logger()
except ImportError:
    # Fallback to standard logging if not available
    logger = logging.getLogger(__name__)

# Import scoring pipeline
try:
    from .scoring_pipeline import (
        ScoringPipeline, 
        create_hybrid_pipeline, 
        create_code_search_pipeline, 
        create_documentation_pipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


def code_preprocessor(text: str) -> str:
    """
    Preprocess code text for better BM25 tokenization.
    
    Phase 2 improvement: Handle code-specific patterns like:
    - camelCase and snake_case splitting
    - Special characters and operators
    - Common code patterns
    """
    # Split camelCase: BM25Manager -> BM25 Manager
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Split after numbers: BM25Manager -> BM 25 Manager
    text = re.sub(r'([0-9])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([A-Z])([0-9])', r'\1 \2', text)
    
    # Split snake_case: bm25_manager -> bm25 manager
    text = re.sub(r'_', ' ', text)
    
    # Split dots but preserve file extensions
    text = re.sub(r'\.(?![a-zA-Z]{2,4}\b)', ' . ', text)
    
    # Handle common operators and symbols
    text = re.sub(r'([=<>!]+)', r' \1 ', text)
    text = re.sub(r'([(){}\[\],:;])', r' \1 ', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text.lower()


@dataclass
class SearchResult:
    """Represents a search result with scores from different methods"""
    content: str
    file_path: str
    chunk_index: int
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "chunk_index": self.chunk_index,
            "scores": {
                "vector": self.vector_score,
                "bm25": self.bm25_score,
                "combined": self.combined_score
            },
            "metadata": self.metadata or {}
        }


class BM25Manager:
    """Manages BM25 indices for collections"""
    
    def __init__(self):
        self.indices = {}  # collection_name -> BM25Retriever
        self.documents = {}  # collection_name -> List[Document]
        self.preprocessors = {}  # collection_name -> preprocessor function
        
    def update_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        """Update BM25 index for a collection - replaces entire index with chunks"""
        # Determine if this is a code collection and needs preprocessing
        is_code_collection = "_code" in collection_name
        
        # Set preprocessor for code collections (Phase 2 improvement)
        if is_code_collection:
            self.preprocessors[collection_name] = code_preprocessor
        
        # Convert to Langchain documents
        langchain_docs = []
        for doc in documents:
            content = doc.get("content", "")
            
            # Apply preprocessing if available
            if collection_name in self.preprocessors:
                preprocessed_content = self.preprocessors[collection_name](content)
            else:
                preprocessed_content = content
            
            metadata = {
                "file_path": doc.get("file_path", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "line_start": doc.get("line_start", 0),
                "line_end": doc.get("line_end", 0),
                "language": doc.get("language", ""),
                "chunk_type": doc.get("chunk_type", "general"),
                "id": doc.get("id", ""),  # Store document ID for retrieval
                "original_content": content  # Store original for display
            }
            langchain_docs.append(Document(page_content=preprocessed_content, metadata=metadata))
        
        # Create BM25 retriever
        if langchain_docs:
            retriever = BM25Retriever.from_documents(langchain_docs)
            self.indices[collection_name] = retriever
            self.documents[collection_name] = langchain_docs
            logger.info(f"Updated BM25 index for {collection_name} with {len(langchain_docs)} chunks", extra={
                "operation": "bm25_index_update",
                "collection": collection_name,
                "chunk_count": len(langchain_docs),
                "status": "success"
            })
    
    def append_documents(self, collection_name: str, new_documents: List[Dict[str, Any]]):
        """Append chunks to existing BM25 index - more efficient for incremental updates"""
        # Check if we need preprocessing
        is_code_collection = "_code" in collection_name
        if is_code_collection and collection_name not in self.preprocessors:
            self.preprocessors[collection_name] = code_preprocessor
        
        # Convert new documents to Langchain format
        new_langchain_docs = []
        for doc in new_documents:
            content = doc.get("content", "")
            
            # Apply preprocessing if available
            if collection_name in self.preprocessors:
                preprocessed_content = self.preprocessors[collection_name](content)
            else:
                preprocessed_content = content
            
            metadata = {
                "file_path": doc.get("file_path", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "line_start": doc.get("line_start", 0),
                "line_end": doc.get("line_end", 0),
                "language": doc.get("language", ""),
                "chunk_type": doc.get("chunk_type", "general"),
                "id": doc.get("id", ""),  # Store document ID for retrieval
                "original_content": content  # Store original for display
            }
            new_langchain_docs.append(Document(page_content=preprocessed_content, metadata=metadata))
        
        if not new_langchain_docs:
            return
        
        # If collection doesn't exist yet, just create it
        if collection_name not in self.documents:
            self.update_index(collection_name, new_documents)
            return
        
        # Append to existing documents and rebuild index
        # (BM25Retriever doesn't support incremental updates, so we must rebuild)
        existing_docs = self.documents.get(collection_name, [])
        all_docs = existing_docs + new_langchain_docs
        
        # Rebuild the retriever with all documents
        retriever = BM25Retriever.from_documents(all_docs)
        self.indices[collection_name] = retriever
        self.documents[collection_name] = all_docs
        
        logger.info(f"Appended {len(new_documents)} chunks to BM25 index for {collection_name} (total: {len(all_docs)})", extra={
            "operation": "bm25_index_append",
            "collection": collection_name,
            "new_count": len(new_documents),
            "total_count": len(all_docs),
            "status": "success"
        })
    
    def needs_rebuild(self, collection_name: str) -> bool:
        """Check if BM25 index needs to be rebuilt for a collection"""
        return collection_name not in self.indices
    
    def build_from_qdrant(self, collection_name: str, qdrant_client):
        """Build BM25 index from all chunks in a Qdrant collection"""
        # Set preprocessor for code collections
        is_code_collection = "_code" in collection_name
        if is_code_collection:
            self.preprocessors[collection_name] = code_preprocessor
            
        try:
            documents = []
            offset = None
            
            while True:
                # Scroll through all documents
                scroll_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,  # Batch size
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = scroll_result
                
                if not points:
                    break
                
                # Convert points to documents
                for point in points:
                    doc = {
                        "id": f"{point.payload.get('file_path', '')}_{point.payload.get('chunk_index', 0)}",
                        "content": point.payload.get("content", ""),
                        "file_path": point.payload.get("file_path", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "line_start": point.payload.get("line_start", 0),
                        "line_end": point.payload.get("line_end", 0),
                        "language": point.payload.get("language", ""),
                        "chunk_type": point.payload.get("chunk_type", "general"),
                        "doc_type": point.payload.get("doc_type", ""),
                        "heading": point.payload.get("heading", ""),
                    }
                    documents.append(doc)
                
                # Update offset for next batch
                offset = next_offset
                
                # Break if no more documents
                if next_offset is None:
                    break
            
            # Build BM25 index with all chunks
            if documents:
                self.update_index(collection_name, documents)
                logger.info(f"Built BM25 index for {collection_name} from Qdrant with {len(documents)} chunks")
                return True
            else:
                logger.warning(f"No chunks found in Qdrant collection {collection_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build BM25 index from Qdrant for {collection_name}: {e}")
            return False
        
    def search(self, collection_name: str, query: str, k: int = 5, qdrant_client=None) -> List[Tuple[str, float]]:
        """Search using BM25 in a specific collection"""
        if collection_name not in self.indices:
            # Try to build index on-demand if client provided
            if qdrant_client:
                logger.info(f"BM25 index not found for {collection_name}, building on-demand...")
                self.build_from_qdrant(collection_name, qdrant_client)
                
                # Check if it was built successfully
                if collection_name not in self.indices:
                    logger.warning(f"Failed to build BM25 index for collection {collection_name}")
                    return []
            else:
                logger.warning(f"BM25 index not found for collection {collection_name} and no client provided")
                return []
            
        retriever = self.indices[collection_name]
        retriever.k = k
        
        # Preprocess query if we have a preprocessor for this collection
        if collection_name in self.preprocessors:
            processed_query = self.preprocessors[collection_name](query)
        else:
            processed_query = query
        
        # Get results
        results = retriever.invoke(processed_query)
        
        # Convert to (doc_id, score) tuples
        # Note: BM25Retriever doesn't provide scores, so we use normalized rank-based scoring
        scored_results = []
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id", f"{doc.metadata['file_path']}_{doc.metadata['chunk_index']}")
            # Use normalized score instead of reciprocal rank for better alignment with vector scores
            # This gives 0.667, 0.75, 0.8 for ranks 1, 2, 3 instead of 1.0, 0.5, 0.333
            score = 1.0 - (1.0 / (rank + 2))
            scored_results.append((doc_id, score))
            
        return scored_results
    
    def clear_index(self, collection_name: str):
        """Clear BM25 index for a collection"""
        if collection_name in self.indices:
            del self.indices[collection_name]
            del self.documents[collection_name]
            

class HybridSearcher:
    """
    Implements hybrid search combining BM25 and vector search.
    
    This class manages the coordination between keyword-based BM25 search
    and semantic vector search, providing various fusion strategies to
    combine results.
    """
    
    def __init__(self):
        self.bm25_manager = BM25Manager()
        
        # Load weight configuration from config file
        try:
            from ..config import get_config
            config = get_config()
        except:
            # Fallback if import fails
            config = {}
        
        # Default weights for different search types
        self.default_weights = {
            "code": {"vector": 0.5, "bm25": 0.5},
            "documentation": {"vector": 0.8, "bm25": 0.2},
            "config": {"vector": 0.6, "bm25": 0.4},
            "general": {"vector": 0.7, "bm25": 0.3}
        }
        
        # Override with config if available
        if "hybrid_search" in config and "weights" in config["hybrid_search"]:
            self.default_weights.update(config["hybrid_search"]["weights"])
        
    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF is a simple but effective fusion method that combines rankings
        from different retrieval methods.
        
        Args:
            vector_results: List of (doc_id, score) from vector search
            bm25_results: List of (doc_id, score) from BM25 search
            k: RRF parameter (default 60)
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 results
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Create score dictionaries
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        
        # Add vector search contributions
        for rank, (doc_id, _) in enumerate(vector_results):
            rrf_scores[doc_id] += vector_weight * (1.0 / (k + rank + 1))
            
        # Add BM25 contributions  
        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] += bm25_weight * (1.0 / (k + rank + 1))
            
        # Create final results
        results = []
        for doc_id, combined_score in rrf_scores.items():
            result = SearchResult(
                content=doc_id,  # Will be replaced with actual content
                file_path="",    # Will be populated from metadata
                chunk_index=0,   # Will be populated from metadata
                vector_score=vector_scores.get(doc_id),
                bm25_score=bm25_scores.get(doc_id),
                combined_score=combined_score
            )
            results.append(result)
            
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def linear_combination(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Combine results using simple linear combination of scores.
        
        Args:
            vector_results: List of (doc_id, score) from vector search
            bm25_results: List of (doc_id, score) from BM25 search
            vector_weight: Weight for vector scores (should sum to 1 with bm25_weight)
            bm25_weight: Weight for BM25 scores
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Normalize weights
        total_weight = vector_weight + bm25_weight
        vector_weight = vector_weight / total_weight
        bm25_weight = bm25_weight / total_weight
        
        # Create score dictionaries
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # Get all unique document IDs
        all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # Calculate combined scores
        results = []
        for doc_id in all_doc_ids:
            v_score = vector_scores.get(doc_id, 0.0)
            b_score = bm25_scores.get(doc_id, 0.0)
            
            # Normalize scores to [0, 1] range if needed
            combined = (vector_weight * v_score) + (bm25_weight * b_score)
            
            result = SearchResult(
                content=doc_id,
                file_path="",
                chunk_index=0,
                vector_score=v_score,
                bm25_score=b_score,
                combined_score=combined
            )
            results.append(result)
            
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def linear_combination_with_exact_match(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]], 
        query: str,
        result_objects_map: Dict[str, Any],  # doc_id -> original result object with content
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        exact_match_bonus: float = 0.2
    ) -> List[SearchResult]:
        """
        Combine results using linear combination with exact match bonus.
        
        Args:
            vector_results: List of (doc_id, score) from vector search
            bm25_results: List of (doc_id, score) from BM25 search
            query: Original search query
            result_objects_map: Map of doc_id to result objects containing content
            vector_weight: Weight for vector scores
            bm25_weight: Weight for BM25 scores
            exact_match_bonus: Bonus to add for exact matches (default 0.2)
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # First do normal linear combination
        results = self.linear_combination(
            vector_results, bm25_results, vector_weight, bm25_weight
        )
        
        # Add exact match bonus
        query_terms = query.lower().split()
        
        for result in results:
            doc_id = result.content
            if doc_id in result_objects_map:
                content = result_objects_map[doc_id].payload.get("content", "").lower()
                
                # Check if all query terms appear in content
                if all(term in content for term in query_terms):
                    # Add bonus, capping at 1.0
                    result.combined_score = min(1.0, result.combined_score + exact_match_bonus)
                    result.metadata = result.metadata or {}
                    result.metadata["exact_match"] = True
        
        # Re-sort by updated scores
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def get_weights_for_search_type(self, search_type: str = "general") -> Tuple[float, float]:
        """
        Get vector and BM25 weights based on search type.
        
        Args:
            search_type: Type of search ("code", "documentation", "config", "general")
            
        Returns:
            Tuple of (vector_weight, bm25_weight)
        """
        weights = self.default_weights.get(search_type, self.default_weights["general"])
        return weights["vector"], weights["bm25"]
    
    def get_fusion_method(self, method: str = "rrf"):
        """
        Get the fusion method function by name.
        
        Args:
            method: Fusion method name ("rrf" or "linear")
            
        Returns:
            Fusion function
        """
        methods = {
            "rrf": self.reciprocal_rank_fusion,
            "linear": self.linear_combination
        }
        
        if method not in methods:
            logger.warning(f"Unknown fusion method: {method}, using RRF")
            return self.reciprocal_rank_fusion
            
        return methods[method]
    
    def search_with_pipeline(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        search_type: str = "general",
        enhanced_ranker=None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search using the configurable scoring pipeline.
        
        Args:
            query: Search query
            vector_results: Results from vector search with scores
            bm25_results: Results from BM25 search with scores
            search_type: Type of search ("code", "documentation", "config", "general")
            enhanced_ranker: Optional enhanced ranker instance
            context: Optional context for scoring
            
        Returns:
            List of results with pipeline scores
        """
        if not PIPELINE_AVAILABLE:
            # Fallback to linear combination if pipeline not available
            logger.warning("Scoring pipeline not available, using linear combination")
            v_weight, b_weight = self.get_weights_for_search_type(search_type)
            
            # Convert to tuples for legacy method
            v_tuples = [(r.get("id", f"{r['file_path']}_{r['chunk_index']}"), r.get("score", 0)) 
                       for r in vector_results]
            b_tuples = [(r.get("id", f"{r['file_path']}_{r['chunk_index']}"), r.get("score", 0)) 
                       for r in bm25_results]
            
            results = self.linear_combination(v_tuples, b_tuples, v_weight, b_weight)
            
            # Convert back to dict format
            return [{"id": r.content, "score": r.combined_score, "vector_score": r.vector_score, 
                    "bm25_score": r.bm25_score} for r in results]
        
        # Merge results from both searches
        all_docs = {}
        
        # Add vector results
        for result in vector_results:
            doc_id = result.get("id", f"{result['file_path']}_{result['chunk_index']}")
            all_docs[doc_id] = result.copy()
            all_docs[doc_id]["vector_score"] = result.get("score", 0.0)
            all_docs[doc_id]["id"] = doc_id
        
        # Add/update with BM25 results
        for result in bm25_results:
            doc_id = result.get("id", f"{result['file_path']}_{result['chunk_index']}")
            if doc_id not in all_docs:
                all_docs[doc_id] = result.copy()
                all_docs[doc_id]["vector_score"] = 0.0
            all_docs[doc_id]["bm25_score"] = result.get("score", 0.0)
            all_docs[doc_id]["id"] = doc_id
        
        # Convert to list
        documents = list(all_docs.values())
        
        # Create appropriate pipeline based on search type
        if search_type == "code":
            pipeline = create_code_search_pipeline(enhanced_ranker)
        elif search_type == "documentation":
            pipeline = create_documentation_pipeline(enhanced_ranker)
        else:
            v_weight, b_weight = self.get_weights_for_search_type(search_type)
            pipeline = create_hybrid_pipeline(v_weight, b_weight, enhanced_ranker=enhanced_ranker)
        
        # Run pipeline
        return pipeline.score(query, documents, context)


# Singleton instance
_hybrid_searcher = None

def get_hybrid_searcher() -> HybridSearcher:
    """Get or create the singleton hybrid searcher instance"""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher