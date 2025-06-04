"""
Configurable scoring pipeline for combining multiple scoring stages.

This module implements a flexible pipeline architecture for search result scoring,
allowing different scoring strategies to be composed and configured dynamically.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

# Use project-aware logging
try:
    from utils.logging import get_project_logger
    logger = get_project_logger()
except ImportError:
    logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result from a scoring stage with detailed score breakdown"""
    doc_id: str
    stage_name: str
    score: float
    metadata: Dict[str, Any] = None
    

class ScoringStage(ABC):
    """Abstract base class for scoring pipeline stages"""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Score documents and return results"""
        pass
        
    def get_config(self) -> Dict[str, Any]:
        """Get stage configuration"""
        return {
            "name": self.name,
            "weight": self.weight,
            "type": self.__class__.__name__
        }


class VectorScoringStage(ScoringStage):
    """Stage for vector similarity scoring"""
    
    def __init__(self, weight: float = 0.7):
        super().__init__("vector", weight)
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Extract and normalize vector scores from documents"""
        results = []
        for doc in documents:
            # Vector scores are already computed during search
            vector_score = doc.get("vector_score", 0.0)
            
            results.append(ScoringResult(
                doc_id=doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}"),
                stage_name=self.name,
                score=vector_score,
                metadata={"similarity_type": "cosine"}
            ))
        
        return results


class BM25ScoringStage(ScoringStage):
    """Stage for BM25 keyword scoring"""
    
    def __init__(self, weight: float = 0.3):
        super().__init__("bm25", weight)
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Extract and normalize BM25 scores from documents"""
        results = []
        for doc in documents:
            # BM25 scores are already computed during search
            bm25_score = doc.get("bm25_score", 0.0)
            
            results.append(ScoringResult(
                doc_id=doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}"),
                stage_name=self.name,
                score=bm25_score,
                metadata={"tokenization": "code_aware" if "_code" in context.get("collection", "") else "standard"}
            ))
        
        return results


class ExactMatchStage(ScoringStage):
    """Stage for exact match boosting"""
    
    def __init__(self, bonus: float = 0.2):
        super().__init__("exact_match", 1.0)
        self.bonus = bonus
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Add bonus for exact query matches"""
        results = []
        query_lower = query.lower()
        query_terms = query_lower.split()
        
        for doc in documents:
            content = doc.get("content", "").lower()
            
            # Check for exact phrase match
            exact_phrase = query_lower in content
            
            # Check if all terms are present
            all_terms = all(term in content for term in query_terms)
            
            # Calculate bonus
            if exact_phrase:
                score = self.bonus
                match_type = "exact_phrase"
            elif all_terms:
                score = self.bonus * 0.5
                match_type = "all_terms"
            else:
                score = 0.0
                match_type = "none"
            
            results.append(ScoringResult(
                doc_id=doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}"),
                stage_name=self.name,
                score=score,
                metadata={"match_type": match_type}
            ))
        
        return results


class FusionStage(ScoringStage):
    """Stage that combines scores from multiple previous stages"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("fusion", 1.0)
        self.weights = weights or {"vector": 0.7, "bm25": 0.3}
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Combine scores from previous stages using configured weights"""
        # This stage expects pre-computed scores in context
        stage_scores = context.get("stage_scores", {})
        results = []
        
        for doc in documents:
            doc_id = doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}")
            
            # Collect scores from different stages
            combined_score = 0.0
            score_breakdown = {}
            
            for stage_name, weight in self.weights.items():
                if stage_name in stage_scores and doc_id in stage_scores[stage_name]:
                    stage_score = stage_scores[stage_name][doc_id]
                    weighted_score = weight * stage_score
                    combined_score += weighted_score
                    score_breakdown[stage_name] = {
                        "raw": stage_score,
                        "weighted": weighted_score,
                        "weight": weight
                    }
            
            # Normalize if weights don't sum to 1
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                combined_score /= total_weight
            
            results.append(ScoringResult(
                doc_id=doc_id,
                stage_name=self.name,
                score=combined_score,
                metadata={"breakdown": score_breakdown}
            ))
        
        return results


class EnhancedRankingStage(ScoringStage):
    """Stage that applies enhanced ranking signals"""
    
    def __init__(self, ranker=None):
        super().__init__("enhanced_ranking", 1.0)
        self.ranker = ranker
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        """Apply enhanced ranking signals"""
        if not self.ranker:
            # Just pass through if no ranker configured
            return [
                ScoringResult(
                    doc_id=doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}"),
                    stage_name=self.name,
                    score=doc.get("score", 0.0)
                )
                for doc in documents
            ]
        
        # Apply enhanced ranking
        enhanced_docs = self.ranker.rank_results(
            documents,
            query_context=context.get("query_context"),
            dependency_graph=context.get("dependency_graph")
        )
        
        results = []
        for doc in enhanced_docs:
            results.append(ScoringResult(
                doc_id=doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}"),
                stage_name=self.name,
                score=doc.get("enhanced_score", doc.get("score", 0.0)),
                metadata={"ranking_signals": doc.get("ranking_signals", {})}
            ))
        
        return results


class ScoringPipeline:
    """
    Configurable pipeline for document scoring.
    
    This class orchestrates multiple scoring stages to produce final document scores.
    Stages are executed in order, with results from each stage available to subsequent stages.
    """
    
    def __init__(self, stages: List[ScoringStage], config: Optional[Dict[str, Any]] = None):
        """
        Initialize the scoring pipeline.
        
        Args:
            stages: List of scoring stages to execute in order
            config: Optional configuration for the pipeline
        """
        self.stages = stages
        self.config = config or {}
        self.debug = self.config.get("debug", False)
        
        logger.info(f"Scoring pipeline initialized with {len(stages)} stages: {[s.name for s in stages]}")
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run the scoring pipeline on documents.
        
        Args:
            query: Search query
            documents: List of documents to score
            context: Optional context for scoring (e.g., current file, dependencies)
            
        Returns:
            Documents with final scores and stage metadata
        """
        if not documents:
            return []
            
        start_time = time.time()
        context = context or {}
        
        # Track scores from each stage
        stage_scores = {}
        stage_metadata = {}
        
        # Execute each stage
        for stage in self.stages:
            stage_start = time.time()
            
            # Add previous stage scores to context
            context["stage_scores"] = stage_scores
            
            # Run the stage
            try:
                stage_results = stage.score(query, documents, context)
                
                # Store results by document ID
                stage_scores[stage.name] = {}
                stage_metadata[stage.name] = {}
                
                for result in stage_results:
                    stage_scores[stage.name][result.doc_id] = result.score
                    if result.metadata:
                        stage_metadata[stage.name][result.doc_id] = result.metadata
                
                stage_duration = (time.time() - stage_start) * 1000
                logger.debug(f"Stage '{stage.name}' completed in {stage_duration:.1f}ms")
                
            except Exception as e:
                logger.error(f"Error in scoring stage '{stage.name}': {e}")
                # Continue with other stages
                continue
        
        # Build final results
        final_results = []
        for doc in documents:
            doc_id = doc.get("id", f"{doc.get('file_path', '')}_{doc.get('chunk_index', 0)}")
            
            # Get final score (from last stage or fusion stage)
            final_score = 0.0
            for stage in reversed(self.stages):
                if stage.name in stage_scores and doc_id in stage_scores[stage.name]:
                    final_score = stage_scores[stage.name][doc_id]
                    break
            
            # Build result with all metadata
            result = doc.copy()
            result["score"] = final_score
            result["pipeline_scores"] = {
                stage_name: scores.get(doc_id, 0.0)
                for stage_name, scores in stage_scores.items()
            }
            
            if self.debug:
                result["pipeline_metadata"] = {
                    stage_name: metadata.get(doc_id, {})
                    for stage_name, metadata in stage_metadata.items()
                    if doc_id in metadata
                }
            
            final_results.append(result)
        
        # Sort by final score
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        pipeline_duration = (time.time() - start_time) * 1000
        logger.info(f"Scoring pipeline completed in {pipeline_duration:.1f}ms for {len(documents)} documents")
        
        return final_results
    
    def get_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return {
            "stages": [stage.get_config() for stage in self.stages],
            "config": self.config
        }


# Factory functions for common pipeline configurations

def create_hybrid_pipeline(vector_weight: float = 0.7, bm25_weight: float = 0.3, 
                          exact_match_bonus: float = 0.2, enhanced_ranker=None) -> ScoringPipeline:
    """Create a standard hybrid search scoring pipeline"""
    stages = [
        VectorScoringStage(weight=vector_weight),
        BM25ScoringStage(weight=bm25_weight),
        FusionStage(weights={"vector": vector_weight, "bm25": bm25_weight}),
        ExactMatchStage(bonus=exact_match_bonus)
    ]
    
    if enhanced_ranker:
        stages.append(EnhancedRankingStage(ranker=enhanced_ranker))
    
    return ScoringPipeline(stages)


def create_code_search_pipeline(enhanced_ranker=None) -> ScoringPipeline:
    """Create a pipeline optimized for code search"""
    stages = [
        VectorScoringStage(weight=0.5),
        BM25ScoringStage(weight=0.5),  # Higher weight for code keyword matching
        FusionStage(weights={"vector": 0.5, "bm25": 0.5}),
        ExactMatchStage(bonus=0.3)  # Higher bonus for exact code matches
    ]
    
    if enhanced_ranker:
        stages.append(EnhancedRankingStage(ranker=enhanced_ranker))
    
    return ScoringPipeline(stages, config={"debug": False})


def create_documentation_pipeline(enhanced_ranker=None) -> ScoringPipeline:
    """Create a pipeline optimized for documentation search"""
    stages = [
        VectorScoringStage(weight=0.8),
        BM25ScoringStage(weight=0.2),  # Lower weight for docs
        FusionStage(weights={"vector": 0.8, "bm25": 0.2}),
        ExactMatchStage(bonus=0.1)  # Lower bonus for docs
    ]
    
    if enhanced_ranker:
        stages.append(EnhancedRankingStage(ranker=enhanced_ranker))
    
    return ScoringPipeline(stages)