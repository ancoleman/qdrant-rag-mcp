# Enhanced Ranking Guide

This guide explains how the enhanced ranking system works in v0.2.1+ and how to configure it for optimal search results.

## Overview

The enhanced ranking system combines 5 different signals to provide more relevant search results. Instead of relying solely on semantic similarity, it considers multiple factors that affect code relevance.

## The 5 Ranking Signals

### 1. Base Score (40% default weight)
The original hybrid search score combining:
- **Vector similarity**: Semantic meaning of the code
- **BM25 keyword match**: Exact term matching

This remains the foundation but is now enhanced with additional context.

### 2. File Proximity (20% default weight)
Files in the same or nearby directories get boosted scores:
- **Same directory**: 100% boost
- **Parent/child directory**: 75% boost
- **Sibling directories**: 50% boost
- **Unrelated paths**: 0% boost

**Example**: When searching from `src/utils/logger.py`, other files in `src/utils/` will rank higher.

### 3. Dependency Distance (20% default weight)
Files with import relationships score higher:
- **Direct imports**: 100% boost (file A imports file B)
- **Direct importers**: 100% boost (file B imports file A)
- **Indirect relationships**: 30% boost
- **No relationship**: 0% boost

**Example**: Searching for a function will prioritize files that import or are imported by the current context.

### 4. Code Structure Similarity (10% default weight)
Similar code structures group together:
- **Same chunk type**: High boost (function with function, class with class)
- **Different types**: Lower score

**Example**: When searching from a test file, other test files rank higher.

### 5. Recency (10% default weight)
Recently modified files surface first:
- **Very recent**: 90-100% boost
- **Recent**: 50-90% boost
- **Older**: 0-50% boost

Uses exponential decay based on file modification time.

## Configuration

### Default Weights

Configure in `config/server_config.json`:

```json
{
  "search": {
    "enhanced_ranking": {
      "base_score_weight": 0.4,
      "file_proximity_weight": 0.2,
      "dependency_distance_weight": 0.2,
      "code_structure_weight": 0.1,
      "recency_weight": 0.1
    }
  }
}
```

### Customizing Weights

Adjust weights based on your use case:

#### For Refactoring (emphasize structure and dependencies)
```json
{
  "enhanced_ranking": {
    "base_score_weight": 0.3,
    "file_proximity_weight": 0.1,
    "dependency_distance_weight": 0.3,
    "code_structure_weight": 0.2,
    "recency_weight": 0.1
  }
}
```

#### For Bug Fixing (emphasize recency and proximity)
```json
{
  "enhanced_ranking": {
    "base_score_weight": 0.3,
    "file_proximity_weight": 0.3,
    "dependency_distance_weight": 0.1,
    "code_structure_weight": 0.1,
    "recency_weight": 0.2
  }
}
```

#### For Architecture Review (emphasize dependencies)
```json
{
  "enhanced_ranking": {
    "base_score_weight": 0.3,
    "file_proximity_weight": 0.2,
    "dependency_distance_weight": 0.4,
    "code_structure_weight": 0.1,
    "recency_weight": 0.0
  }
}
```

## Understanding Search Results

Each search result now includes ranking signals:

```json
{
  "score": 0.786,  // Final enhanced score
  "ranking_signals": {
    "base_score": 0.65,         // Original search score
    "file_proximity": 1.0,      // Same directory
    "dependency_distance": 0.5, // No direct relationship
    "code_structure": 0.9,      // Similar structure
    "recency": 0.8             // Recently modified
  }
}
```

## When Enhanced Ranking Applies

- **Enabled**: Only in `hybrid` search mode (default)
- **Disabled**: In `vector` or `keyword` only modes
- **Automatic**: No code changes needed, just update config

## Tips for Best Results

1. **Keep dependencies indexed**: Run `reindex_directory` after major refactors
2. **Use with context**: Enhanced ranking works best with `include_dependencies=true`
3. **Monitor signals**: Check `ranking_signals` to understand why results ranked as they did
4. **Adjust gradually**: Small weight changes (±0.1) can significantly affect results

## Performance Considerations

- **Overhead**: <50ms for 100 results
- **Memory**: O(n) where n is result count
- **No indexing changes**: Ranking happens at search time

## Troubleshooting

### Results seem random
- Check if you're using `hybrid` mode
- Verify weights sum close to 1.0 (auto-normalized)
- Ensure files have been recently reindexed

### Old files ranking too high
- Reduce `recency_weight`
- Increase `base_score_weight`

### Missing expected results
- Enhanced ranking doesn't change what's found, only the order
- Try increasing `n_results` to see more matches
- Check if files are properly indexed

## Example Searches

### Finding related utilities
```
Query: "logger utils"
Context: Currently in src/utils/config.py

Results will prioritize:
1. Other files in src/utils/ (proximity boost)
2. Files that import logging utilities (dependency boost)
3. Other utility functions (structure boost)
```

### Finding recent changes
```
Query: "bug fix"
With high recency_weight (0.3)

Results will prioritize:
1. Recently modified files
2. Files near your current location
3. Files with "fix" in content
```

### Architecture exploration
```
Query: "database connection"
With high dependency_weight (0.4)

Results will prioritize:
1. Core database modules
2. Files that import database utilities
3. Configuration files (if indexed)
```

## Advanced Usage

### Programmatic Weight Updates

While the MCP server uses config file weights, the EnhancedRanker class supports dynamic updates:

```python
ranker.update_weights({
    "recency_weight": 0.3,
    "file_proximity_weight": 0.1
})
```

### Custom Ranking Signals

The system is extensible. New signals can be added by:
1. Implementing a `_calculate_[signal]_scores` method
2. Adding the signal to weight configuration
3. Including in the ranking combination

## Summary

Enhanced ranking makes search results more relevant by considering multiple factors beyond just text similarity. The default weights work well for general use, but customizing them for your workflow can significantly improve your search experience.