"""
Enhanced Ranking System for v0.2.1

This module implements multi-signal ranking to improve search precision by combining:
- File proximity scoring (same directory boost)
- Dependency distance ranking (direct imports scored higher)
- Code structure similarity metrics
- Recency weighting for recent changes
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import time
from datetime import datetime
from .logging import get_project_logger

logger = get_project_logger()


class EnhancedRanker:
    """
    Implements enhanced ranking signals for search results.
    
    This class adds multiple ranking signals on top of the base hybrid search
    to improve result relevance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the enhanced ranker with configurable weights.
        
        Args:
            config: Optional configuration with weight settings
        """
        self.config = config or {}
        
        # Default weights for different signals
        self.weights = {
            "base_score": self.config.get("base_score_weight", 0.4),
            "file_proximity": self.config.get("file_proximity_weight", 0.2),
            "dependency_distance": self.config.get("dependency_distance_weight", 0.2),
            "code_structure": self.config.get("code_structure_weight", 0.1),
            "recency": self.config.get("recency_weight", 0.1)
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
        logger.info(f"Enhanced ranker initialized with weights: {self.weights}")
    
    def rank_results(
        self,
        results: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]] = None,
        dependency_graph: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Apply enhanced ranking to search results.
        
        Args:
            results: List of search results with base scores
            query_context: Optional context about the query (e.g., current file)
            dependency_graph: Optional dependency information
            
        Returns:
            Re-ranked results with enhanced scores
        """
        if not results:
            return results
            
        start_time = time.time()
        
        # Calculate individual ranking signals
        proximity_scores = self._calculate_file_proximity_scores(results, query_context)
        dependency_scores = self._calculate_dependency_scores(results, dependency_graph)
        structure_scores = self._calculate_structure_similarity_scores(results)
        recency_scores = self._calculate_recency_scores(results)
        
        # Combine scores
        enhanced_results = []
        for i, result in enumerate(results):
            # Get base score (already normalized 0-1 from RRF)
            base_score = float(result.get("score", 0.0))
            
            # Calculate enhanced score
            enhanced_score = (
                self.weights["base_score"] * base_score +
                self.weights["file_proximity"] * proximity_scores[i] +
                self.weights["dependency_distance"] * dependency_scores[i] +
                self.weights["code_structure"] * structure_scores[i] +
                self.weights["recency"] * recency_scores[i]
            )
            
            # Add enhanced result
            enhanced_result = result.copy()
            enhanced_result["enhanced_score"] = enhanced_score
            enhanced_result["ranking_signals"] = {
                "base_score": base_score,
                "file_proximity": proximity_scores[i],
                "dependency_distance": dependency_scores[i],
                "code_structure": structure_scores[i],
                "recency": recency_scores[i]
            }
            enhanced_results.append(enhanced_result)
        
        # Sort by enhanced score (ensure it's a float)
        enhanced_results.sort(key=lambda x: float(x["enhanced_score"]), reverse=True)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.debug(f"Enhanced ranking completed in {duration_ms:.1f}ms for {len(results)} results")
        
        return enhanced_results
    
    def _calculate_file_proximity_scores(
        self,
        results: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Calculate file proximity scores based on directory distance.
        
        Files in the same directory get higher scores, with decreasing
        scores for files further away in the directory tree.
        """
        scores = []
        
        # Determine reference path (from query context or top result)
        reference_path = None
        if query_context and "current_file" in query_context:
            reference_path = query_context["current_file"]
        elif results and "file_path" in results[0]:
            # Use the top result as reference
            reference_path = results[0].get("file_path", "")
            
        if not reference_path:
            # No reference, all get neutral score
            return [0.5] * len(results)
            
        reference_dir = os.path.dirname(reference_path)
        reference_parts = reference_dir.split(os.sep)
        
        for result in results:
            file_path = result.get("file_path", result.get("display_path", ""))
            if not file_path:
                scores.append(0.0)
                continue
                
            file_dir = os.path.dirname(file_path)
            file_parts = file_dir.split(os.sep)
            
            # Calculate common path length
            common_parts = 0
            for ref_part, file_part in zip(reference_parts, file_parts):
                if ref_part == file_part:
                    common_parts += 1
                else:
                    break
            
            # Calculate proximity score
            if file_dir == reference_dir:
                # Same directory
                score = 1.0
            else:
                # Score based on directory distance
                ref_depth = len(reference_parts)
                file_depth = len(file_parts)
                max_depth = max(ref_depth, file_depth)
                
                if max_depth > 0:
                    # Higher score for more common parts
                    score = common_parts / max_depth
                else:
                    score = 0.5
                    
            scores.append(score)
            
        return scores
    
    def _calculate_dependency_scores(
        self,
        results: List[Dict[str, Any]],
        dependency_graph: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Calculate dependency distance scores.
        
        Direct dependencies get higher scores than indirect dependencies.
        """
        if not dependency_graph:
            # No dependency info, all get neutral score
            return [0.5] * len(results)
            
        scores = []
        
        # Get all result file paths
        result_files = set()
        for result in results:
            file_path = result.get("file_path", "")
            if file_path:
                result_files.add(file_path)
        
        for result in results:
            file_path = result.get("file_path", "")
            if not file_path:
                scores.append(0.0)
                continue
                
            # Check if this file has dependencies with other results
            score = 0.0
            file_deps = dependency_graph.get(file_path, {})
            
            # Direct imports (this file imports others)
            imports = file_deps.get("imports", [])
            direct_imports = sum(1 for imp in imports if imp in result_files)
            
            # Direct importers (other files import this)
            importers = file_deps.get("imported_by", [])
            direct_importers = sum(1 for imp in importers if imp in result_files)
            
            # Calculate score based on dependency relationships
            if direct_imports > 0 or direct_importers > 0:
                # Has direct dependencies with other results
                score = 1.0
            elif file_path in dependency_graph:
                # Has dependencies but not with current results
                score = 0.3
            else:
                # No dependencies
                score = 0.0
                
            scores.append(score)
            
        return scores
    
    def _calculate_structure_similarity_scores(
        self,
        results: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate code structure similarity scores.
        
        Similar code structures (e.g., all functions, all classes) get higher scores.
        """
        scores = []
        
        # Count chunk types
        chunk_types = defaultdict(int)
        for result in results:
            chunk_type = result.get("chunk_type", "general")
            chunk_types[chunk_type] += 1
            
        # Find dominant chunk type
        if chunk_types:
            dominant_type = max(chunk_types.items(), key=lambda x: x[1])[0]
            dominant_count = chunk_types[dominant_type]
            total_count = len(results)
            
            for result in results:
                chunk_type = result.get("chunk_type", "general")
                
                if chunk_type == dominant_type and dominant_count > 1:
                    # Part of dominant structure type
                    score = 0.8 + (0.2 * (dominant_count / total_count))
                else:
                    # Different structure type
                    score = 0.3
                    
                scores.append(score)
        else:
            # No structure info
            scores = [0.5] * len(results)
            
        return scores
    
    def _calculate_recency_scores(
        self,
        results: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Calculate recency scores based on file modification times.
        
        More recently modified files get higher scores.
        """
        scores = []
        
        # Get modification times
        mod_times = []
        current_time = time.time()
        
        for result in results:
            # Try to get modification time from metadata
            mod_time = result.get("modified_at", None)
            if mod_time is not None:
                try:
                    # Try to convert to float (Unix timestamp)
                    mod_time = float(mod_time)
                except (ValueError, TypeError):
                    # If it's a string timestamp, try to parse it
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(mod_time.replace('Z', '+00:00'))
                        mod_time = dt.timestamp()
                    except:
                        mod_time = 0
            else:
                # Try to get from file system
                file_path = result.get("file_path", "")
                if file_path and os.path.exists(file_path):
                    try:
                        mod_time = os.path.getmtime(file_path)
                    except:
                        mod_time = 0
                else:
                    mod_time = 0
                    
            mod_times.append(mod_time)
        
        # Calculate scores based on recency
        if any(mod_times):
            max_age = current_time - min(t for t in mod_times if t > 0)
            
            for mod_time in mod_times:
                if mod_time > 0:
                    age = current_time - mod_time
                    # Exponential decay: recent files get higher scores
                    score = max(0.0, 1.0 - (age / max_age))
                else:
                    score = 0.0
                    
                scores.append(score)
        else:
            # No modification time info
            scores = [0.5] * len(results)
            
        return scores
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update the ranking weights dynamically.
        
        Args:
            new_weights: Dictionary of signal names to weight values
        """
        for signal, weight in new_weights.items():
            if signal in self.weights:
                self.weights[signal] = weight
                
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
        logger.info(f"Updated ranking weights: {self.weights}")


# Singleton instance
_enhanced_ranker = None


def get_enhanced_ranker(config: Optional[Dict[str, Any]] = None) -> EnhancedRanker:
    """Get or create the singleton enhanced ranker instance"""
    global _enhanced_ranker
    if _enhanced_ranker is None:
        _enhanced_ranker = EnhancedRanker(config)
    return _enhanced_ranker