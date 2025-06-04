# Scoring Pipeline Architecture

## Overview

The scoring pipeline is a configurable, modular system for combining multiple scoring signals in the Qdrant RAG search system. It replaces ad-hoc score calculations with a clean, extensible architecture that makes it easy to experiment with different scoring strategies.

## Architecture

### Core Components

#### 1. ScoringStage (Abstract Base Class)
```python
class ScoringStage(ABC):
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        pass
```

Each scoring stage:
- Has a unique name for identification
- Can have an optional weight (though weights are typically handled by fusion stages)
- Receives the query, documents, and context
- Returns ScoringResult objects with scores and metadata

#### 2. ScoringPipeline
```python
class ScoringPipeline:
    def __init__(self, stages: List[ScoringStage], config: Optional[Dict[str, Any]] = None):
        self.stages = stages
        self.config = config or {}
```

The pipeline:
- Executes stages in order
- Passes results from previous stages via context
- Tracks all stage scores and metadata
- Returns documents with final scores and debugging information

### Built-in Stages

#### VectorScoringStage
- Extracts pre-computed vector similarity scores
- Default weight: 0.7 for general search
- Captures cosine similarity from embedding search

#### BM25ScoringStage
- Extracts pre-computed BM25 keyword scores
- Default weight: 0.3 for general search
- Indicates whether code-aware tokenization was used

#### ExactMatchStage
- Adds bonus for exact query matches
- Checks for exact phrase matches and all-terms matches
- Default bonus: 0.2 for exact phrase, 0.1 for all terms
- Useful for boosting highly relevant results

#### FusionStage
- Combines scores from multiple stages using weighted sum
- Configurable weights for each input stage
- Normalizes weights to sum to 1.0
- Provides score breakdown in metadata

#### EnhancedRankingStage
- Applies advanced ranking signals (file proximity, recency, etc.)
- Integrates with the existing EnhancedRanker
- Adds ranking signals to metadata

## Usage Examples

### Basic Pipeline Creation

```python
# Create a simple hybrid search pipeline
pipeline = ScoringPipeline([
    VectorScoringStage(weight=0.7),
    BM25ScoringStage(weight=0.3),
    FusionStage(weights={"vector": 0.7, "bm25": 0.3})
])

# Score documents
results = pipeline.score(query, documents)
```

### Using Factory Functions

```python
# Code search pipeline (50/50 vector/BM25 + higher exact match bonus)
code_pipeline = create_code_search_pipeline(enhanced_ranker)

# Documentation pipeline (80/20 vector/BM25 + lower exact match bonus)
doc_pipeline = create_documentation_pipeline(enhanced_ranker)

# Custom weights
custom_pipeline = create_hybrid_pipeline(
    vector_weight=0.6,
    bm25_weight=0.4,
    exact_match_bonus=0.25,
    enhanced_ranker=ranker
)
```

### Creating Custom Stages

```python
class FileTypeBoostStage(ScoringStage):
    """Boost scores based on file type"""
    
    def __init__(self, boost_map: Dict[str, float]):
        super().__init__("file_type_boost", 1.0)
        self.boost_map = boost_map
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        results = []
        for doc in documents:
            file_path = doc.get("file_path", "")
            ext = os.path.splitext(file_path)[1]
            boost = self.boost_map.get(ext, 0.0)
            
            # Get current score from previous stages
            current_score = context.get("stage_scores", {}).get("fusion", {}).get(doc["id"], 0.0)
            
            results.append(ScoringResult(
                doc_id=doc["id"],
                stage_name=self.name,
                score=current_score + boost,
                metadata={"file_type": ext, "boost": boost}
            ))
        return results

# Use in pipeline
pipeline = ScoringPipeline([
    VectorScoringStage(),
    BM25ScoringStage(),
    FusionStage(),
    FileTypeBoostStage(boost_map={".py": 0.1, ".md": 0.05})
])
```

## Integration with Hybrid Search

The scoring pipeline is integrated into the HybridSearcher class:

```python
def search_with_pipeline(
    self,
    query: str,
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    search_type: str = "general",
    enhanced_ranker=None,
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Perform search using the configurable scoring pipeline"""
```

This method:
1. Merges results from vector and BM25 searches
2. Selects appropriate pipeline based on search type
3. Runs the pipeline and returns scored results
4. Falls back to legacy scoring if pipeline not available

## Score Flow Example

For a query "BM25Manager search", here's how scores flow through the pipeline:

1. **Input Documents**:
   - Doc1: vector_score=0.85, bm25_score=0.6
   - Doc2: vector_score=0.7, bm25_score=0.9

2. **VectorScoringStage**: Extracts vector scores
   - Doc1: 0.85
   - Doc2: 0.7

3. **BM25ScoringStage**: Extracts BM25 scores
   - Doc1: 0.6
   - Doc2: 0.9

4. **FusionStage** (weights: vector=0.7, bm25=0.3):
   - Doc1: (0.85 × 0.7) + (0.6 × 0.3) = 0.775
   - Doc2: (0.7 × 0.7) + (0.9 × 0.3) = 0.76

5. **ExactMatchStage** (bonus=0.2):
   - Doc1: Contains "BM25Manager" → 0.775 + 0.2 = 0.975
   - Doc2: No exact match → 0.76 + 0 = 0.76

6. **Final Ranking**:
   1. Doc1: 0.975
   2. Doc2: 0.76

## Configuration

### Pipeline Configuration
```python
pipeline_config = {
    "debug": True,  # Include detailed metadata in results
    "max_stages": 10,  # Maximum number of stages
    "timeout_ms": 1000  # Timeout for pipeline execution
}
```

### Stage Weights
Different search types use different default weights:

- **Code Search**: 50/50 vector/BM25 (equal emphasis on semantic and keyword)
- **Documentation**: 80/20 vector/BM25 (semantic understanding preferred)
- **Config**: 60/40 vector/BM25 (balanced approach)
- **General**: 70/30 vector/BM25 (slight semantic preference)

## Performance Considerations

1. **Stage Execution**: Stages run sequentially, so order matters for dependencies
2. **Memory Usage**: Each stage stores scores in memory; consider this for large result sets
3. **Custom Stages**: Keep custom stage logic efficient as it runs for every document
4. **Debugging**: Enable debug mode only during development to reduce memory usage

## Best Practices

1. **Stage Order**: Place independent stages (vector, BM25) before dependent ones (fusion, boosting)
2. **Weight Normalization**: Fusion stages should normalize weights to maintain score ranges
3. **Error Handling**: Stages should handle missing data gracefully (return 0 scores)
4. **Metadata**: Include relevant debugging info in metadata for troubleshooting
5. **Testing**: Test custom stages with edge cases (empty queries, missing fields)

## Extending the Pipeline

### Adding New Scoring Signals

1. Create a new stage class inheriting from ScoringStage
2. Implement the score method
3. Add to pipeline configuration
4. Test with representative queries

### Example: Query Complexity Stage
```python
class QueryComplexityStage(ScoringStage):
    """Adjust scores based on query complexity"""
    
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        # Simple complexity: number of unique terms
        complexity = len(set(query.lower().split()))
        
        # More complex queries might need stricter matching
        adjustment = 1.0 if complexity <= 2 else 0.9
        
        results = []
        for doc in documents:
            current_score = self._get_current_score(doc["id"], context)
            results.append(ScoringResult(
                doc_id=doc["id"],
                stage_name=self.name,
                score=current_score * adjustment,
                metadata={"query_complexity": complexity, "adjustment": adjustment}
            ))
        return results
```

## Debugging

Enable debug mode to see detailed scoring information:

```python
pipeline = ScoringPipeline(stages, config={"debug": True})
results = pipeline.score(query, documents)

# Each result includes:
{
    "id": "file1_0",
    "score": 0.875,
    "pipeline_scores": {
        "vector": 0.8,
        "bm25": 0.7,
        "fusion": 0.77,
        "exact_match": 0.875
    },
    "pipeline_metadata": {
        "fusion": {"breakdown": {...}},
        "exact_match": {"match_type": "all_terms"}
    }
}
```

## Future Enhancements

1. **Parallel Stage Execution**: Run independent stages concurrently
2. **Conditional Stages**: Skip stages based on query or document characteristics
3. **Learning-to-Rank**: Train stage weights based on user feedback
4. **Stage Composition**: Combine multiple stages into reusable components
5. **Async Support**: Async stage execution for I/O-bound operations

## Conclusion

The scoring pipeline provides a clean, extensible architecture for combining multiple scoring signals. By breaking down scoring into discrete stages, it becomes easier to:

- Understand how scores are calculated
- Experiment with new scoring strategies
- Debug scoring issues
- Maintain and extend the scoring system

The modular design ensures that new requirements can be met by adding new stages rather than modifying existing code, following the Open-Closed Principle.