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

# Use project-aware logging
try:
    from utils.logging import get_project_logger
    logger = get_project_logger()
except ImportError:
    # Fallback to standard logging if not available
    logger = logging.getLogger(__name__)


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
        
    def update_index(self, collection_name: str, documents: List[Dict[str, Any]]):
        """Update BM25 index for a collection"""
        # Convert to Langchain documents
        langchain_docs = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = {
                "file_path": doc.get("file_path", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "line_start": doc.get("line_start", 0),
                "line_end": doc.get("line_end", 0),
                "language": doc.get("language", ""),
                "chunk_type": doc.get("chunk_type", "general"),
                "id": doc.get("id", "")  # Store document ID for retrieval
            }
            langchain_docs.append(Document(page_content=content, metadata=metadata))
        
        # Create BM25 retriever
        if langchain_docs:
            retriever = BM25Retriever.from_documents(langchain_docs)
            self.indices[collection_name] = retriever
            self.documents[collection_name] = langchain_docs
            logger.info(f"Updated BM25 index for {collection_name} with {len(langchain_docs)} documents", extra={
                "operation": "bm25_index_update",
                "collection": collection_name,
                "document_count": len(langchain_docs),
                "status": "success"
            })
        
    def search(self, collection_name: str, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search using BM25 in a specific collection"""
        if collection_name not in self.indices:
            return []
            
        retriever = self.indices[collection_name]
        retriever.k = k
        
        # Get results
        results = retriever.invoke(query)
        
        # Convert to (doc_id, score) tuples
        # Note: BM25Retriever doesn't provide scores, so we use rank-based scoring
        scored_results = []
        for rank, doc in enumerate(results):
            doc_id = doc.metadata.get("id", f"{doc.metadata['file_path']}_{doc.metadata['chunk_index']}")
            # Use reciprocal rank as score (1/rank)
            score = 1.0 / (rank + 1)
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


# Singleton instance
_hybrid_searcher = None

def get_hybrid_searcher() -> HybridSearcher:
    """Get or create the singleton hybrid searcher instance"""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher