#!/usr/bin/env python3
"""Test the configurable scoring pipeline"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from typing import List, Dict, Any
from utils.scoring_pipeline import (
    ScoringPipeline, ScoringStage, ScoringResult,
    VectorScoringStage, BM25ScoringStage, ExactMatchStage, 
    FusionStage, EnhancedRankingStage,
    create_hybrid_pipeline, create_code_search_pipeline, create_documentation_pipeline
)
from utils.enhanced_ranker import EnhancedRanker


def test_basic_pipeline():
    """Test basic scoring pipeline functionality"""
    print("Testing Basic Scoring Pipeline")
    print("-" * 50)
    
    # Sample documents with pre-computed scores
    documents = [
        {
            "id": "file1_0",
            "file_path": "/project/src/utils/helper.py",
            "chunk_index": 0,
            "content": "def calculate_score(query, document):",
            "vector_score": 0.85,
            "bm25_score": 0.6
        },
        {
            "id": "file2_0", 
            "file_path": "/project/src/main.py",
            "chunk_index": 0,
            "content": "# Calculate the final score",
            "vector_score": 0.7,
            "bm25_score": 0.8
        },
        {
            "id": "file3_0",
            "file_path": "/project/tests/test_score.py",
            "chunk_index": 0,
            "content": "assert calculate_score(query, doc) > 0.5",
            "vector_score": 0.75,
            "bm25_score": 0.9
        }
    ]
    
    # Create a simple pipeline
    pipeline = ScoringPipeline([
        VectorScoringStage(weight=0.7),
        BM25ScoringStage(weight=0.3),
        FusionStage(weights={"vector": 0.7, "bm25": 0.3})
    ])
    
    # Score documents
    query = "calculate score"
    results = pipeline.score(query, documents)
    
    # Display results
    print(f"\nQuery: '{query}'")
    print("\nResults:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['file_path']}")
        print(f"   Final Score: {result['score']:.3f}")
        print(f"   Pipeline Scores: {result['pipeline_scores']}")
        

def test_exact_match_bonus():
    """Test exact match bonus stage"""
    print("\n\nTesting Exact Match Bonus")
    print("-" * 50)
    
    documents = [
        {
            "id": "file1_0",
            "file_path": "/project/src/search.py",
            "content": "def exact_match_bonus(query, text):",  # Exact match
            "vector_score": 0.7,
            "bm25_score": 0.8
        },
        {
            "id": "file2_0",
            "file_path": "/project/src/scorer.py", 
            "content": "# Apply bonus for exact matches",  # Partial match
            "vector_score": 0.8,
            "bm25_score": 0.7
        },
        {
            "id": "file3_0",
            "file_path": "/project/src/utils.py",
            "content": "return score + bonus",  # No match
            "vector_score": 0.9,
            "bm25_score": 0.6
        }
    ]
    
    # Pipeline with exact match bonus
    pipeline = ScoringPipeline([
        VectorScoringStage(),
        BM25ScoringStage(), 
        FusionStage(),
        ExactMatchStage(bonus=0.2)
    ], config={"debug": True})
    
    query = "exact_match_bonus"
    results = pipeline.score(query, documents)
    
    print(f"\nQuery: '{query}'")
    print("\nResults with exact match bonus:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['file_path']}")
        print(f"   Content: {result['content']}")
        print(f"   Final Score: {result['score']:.3f}")
        if "pipeline_metadata" in result:
            exact_match_meta = result["pipeline_metadata"].get("exact_match", {})
            if exact_match_meta:
                print(f"   Exact Match: {exact_match_meta}")


def test_factory_pipelines():
    """Test factory-created pipelines"""
    print("\n\nTesting Factory Pipelines")
    print("-" * 50)
    
    documents = [
        {
            "id": "code_1",
            "file_path": "/project/src/parser.py",
            "content": "class BM25Manager:\n    def __init__(self):",
            "chunk_type": "class",
            "vector_score": 0.75,
            "bm25_score": 0.85
        },
        {
            "id": "doc_1",
            "file_path": "/project/docs/README.md",
            "content": "# BM25Manager Documentation\nThis class manages BM25 indices",
            "chunk_type": "documentation",
            "vector_score": 0.8,
            "bm25_score": 0.7
        }
    ]
    
    # Test code search pipeline
    print("\n1. Code Search Pipeline (50/50 vector/BM25):")
    code_pipeline = create_code_search_pipeline()
    code_results = code_pipeline.score("BM25Manager", documents)
    
    for result in code_results[:2]:
        print(f"   {result['file_path']}: {result['score']:.3f}")
    
    # Test documentation pipeline  
    print("\n2. Documentation Pipeline (80/20 vector/BM25):")
    doc_pipeline = create_documentation_pipeline()
    doc_results = doc_pipeline.score("BM25Manager", documents)
    
    for result in doc_results[:2]:
        print(f"   {result['file_path']}: {result['score']:.3f}")


def test_with_enhanced_ranker():
    """Test pipeline with enhanced ranker"""
    print("\n\nTesting Pipeline with Enhanced Ranker")
    print("-" * 50)
    
    documents = [
        {
            "id": "file1_0",
            "file_path": "/project/src/utils/helper.py",
            "chunk_type": "function",
            "content": "def helper_function():",
            "vector_score": 0.75,
            "bm25_score": 0.7,
            "modified_at": 1700000000
        },
        {
            "id": "file2_0",
            "file_path": "/project/src/utils/logger.py",
            "chunk_type": "function", 
            "content": "def log_message():",
            "vector_score": 0.7,
            "bm25_score": 0.75,
            "modified_at": 1600000000
        }
    ]
    
    # Create enhanced ranker
    ranker_config = {
        "base_score_weight": 0.4,
        "file_proximity_weight": 0.3,
        "recency_weight": 0.3
    }
    enhanced_ranker = EnhancedRanker(ranker_config)
    
    # Create pipeline with enhanced ranker
    pipeline = create_hybrid_pipeline(enhanced_ranker=enhanced_ranker)
    
    # Add query context
    context = {
        "query_context": {
            "current_file": "/project/src/utils/config.py"
        }
    }
    
    results = pipeline.score("helper", documents, context)
    
    print("\nResults with enhanced ranking:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['file_path']}")
        print(f"   Final Score: {result['score']:.3f}")
        print(f"   Stage Scores: {result['pipeline_scores']}")


def test_custom_stage():
    """Test adding a custom scoring stage"""
    print("\n\nTesting Custom Scoring Stage")
    print("-" * 50)
    
    class FileTypeBoostStage(ScoringStage):
        """Custom stage that boosts Python files"""
        
        def __init__(self, python_boost: float = 0.1):
            super().__init__("file_type_boost", 1.0)
            self.python_boost = python_boost
            
        def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
            results = []
            for doc in documents:
                file_path = doc.get("file_path", "")
                boost = self.python_boost if file_path.endswith(".py") else 0.0
                
                results.append(ScoringResult(
                    doc_id=doc.get("id"),
                    stage_name=self.name,
                    score=doc.get("score", 0.0) + boost,
                    metadata={"file_type": "python" if boost > 0 else "other"}
                ))
            return results
    
    # Create pipeline with custom stage
    pipeline = ScoringPipeline([
        VectorScoringStage(),
        BM25ScoringStage(),
        FusionStage(),
        FileTypeBoostStage(python_boost=0.15)
    ])
    
    documents = [
        {"id": "1", "file_path": "/project/README.md", "vector_score": 0.8, "bm25_score": 0.7, "score": 0.77},
        {"id": "2", "file_path": "/project/main.py", "vector_score": 0.75, "bm25_score": 0.7, "score": 0.735},
        {"id": "3", "file_path": "/project/config.json", "vector_score": 0.85, "bm25_score": 0.6, "score": 0.775}
    ]
    
    results = pipeline.score("config", documents)
    
    print("\nResults with file type boost:")
    for result in results:
        print(f"{result['file_path']}: {result['score']:.3f} (was {result.get('vector_score', 0)*0.7 + result.get('bm25_score', 0)*0.3:.3f})")


if __name__ == "__main__":
    test_basic_pipeline()
    test_exact_match_bonus()
    test_factory_pipelines()
    test_with_enhanced_ranker()
    test_custom_stage()
    
    print("\n\n✅ All scoring pipeline tests completed!")