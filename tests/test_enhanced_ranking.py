"""
Test script for enhanced ranking in v0.2.1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.enhanced_ranker import EnhancedRanker


def test_enhanced_ranker():
    """Test the enhanced ranker with sample results"""
    
    # Sample search results
    results = [
        {
            "score": 0.8,
            "file_path": "/project/src/utils/helper.py",
            "chunk_type": "function",
            "modified_at": 1700000000  # Recent
        },
        {
            "score": 0.75,
            "file_path": "/project/src/utils/logger.py",  # Same directory
            "chunk_type": "function",
            "modified_at": 1600000000  # Older
        },
        {
            "score": 0.7,
            "file_path": "/project/tests/test_helper.py",  # Different directory
            "chunk_type": "function",
            "modified_at": 1650000000
        },
        {
            "score": 0.65,
            "file_path": "/project/src/core/main.py",
            "chunk_type": "class",  # Different structure
            "modified_at": 1690000000
        }
    ]
    
    # Test with query context
    query_context = {
        "current_file": "/project/src/utils/config.py"
    }
    
    # Create ranker with test weights
    config = {
        "base_score_weight": 0.4,
        "file_proximity_weight": 0.3,
        "dependency_distance_weight": 0.0,  # No dependencies in test
        "code_structure_weight": 0.2,
        "recency_weight": 0.1
    }
    
    ranker = EnhancedRanker(config)
    
    # Apply ranking
    enhanced_results = ranker.rank_results(results, query_context)
    
    # Print results
    print("Enhanced Ranking Results:")
    print("-" * 80)
    
    for i, result in enumerate(enhanced_results):
        print(f"\nRank {i+1}:")
        print(f"  File: {result['file_path']}")
        print(f"  Base Score: {result['score']:.3f}")
        print(f"  Enhanced Score: {result['enhanced_score']:.3f}")
        print(f"  Ranking Signals:")
        signals = result.get('ranking_signals', {})
        for signal, value in signals.items():
            print(f"    {signal}: {value:.3f}")
    
    # Verify that file proximity boosted same-directory files
    utils_files = [r for r in enhanced_results if "/utils/" in r["file_path"]]
    assert len(utils_files) >= 2, "Should have at least 2 utils files"
    
    # Check that utils files are ranked higher
    utils_positions = [i for i, r in enumerate(enhanced_results) if "/utils/" in r["file_path"]]
    avg_position = sum(utils_positions) / len(utils_positions)
    print(f"\nAverage position of utils files: {avg_position:.1f}")
    assert avg_position < 2.0, "Utils files should be ranked in top positions due to proximity"
    
    print("\n✅ Enhanced ranking test passed!")


if __name__ == "__main__":
    test_enhanced_ranker()