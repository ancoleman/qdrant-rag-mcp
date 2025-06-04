# Scoring Pipeline Quick Reference

## Quick Start

```python
from utils.scoring_pipeline import create_hybrid_pipeline

# Basic usage
pipeline = create_hybrid_pipeline()
results = pipeline.score(query, documents)
```

## Factory Functions

### create_hybrid_pipeline()
```python
pipeline = create_hybrid_pipeline(
    vector_weight=0.7,      # Weight for vector search
    bm25_weight=0.3,        # Weight for BM25 search
    exact_match_bonus=0.2,  # Bonus for exact matches
    enhanced_ranker=ranker  # Optional enhanced ranker
)
```

### create_code_search_pipeline()
```python
# Optimized for code: 50/50 vector/BM25, higher exact match bonus
pipeline = create_code_search_pipeline(enhanced_ranker)
```

### create_documentation_pipeline()
```python
# Optimized for docs: 80/20 vector/BM25, lower exact match bonus  
pipeline = create_documentation_pipeline(enhanced_ranker)
```

## Built-in Stages

| Stage | Purpose | Default Weight | Configuration |
|-------|---------|----------------|---------------|
| VectorScoringStage | Extract vector similarity scores | 0.7 | `weight` |
| BM25ScoringStage | Extract BM25 keyword scores | 0.3 | `weight` |
| FusionStage | Combine multiple scores | 1.0 | `weights` dict |
| ExactMatchStage | Bonus for exact matches | N/A | `bonus` (default 0.2) |
| EnhancedRankingStage | Apply ranking signals | 1.0 | `ranker` instance |

## Creating Custom Stages

```python
from utils.scoring_pipeline import ScoringStage, ScoringResult

class MyCustomStage(ScoringStage):
    def __init__(self, my_param: float = 1.0):
        super().__init__("my_custom_stage", weight=1.0)
        self.my_param = my_param
        
    def score(self, query: str, documents: List[Dict[str, Any]], context: Dict[str, Any]) -> List[ScoringResult]:
        results = []
        for doc in documents:
            # Your scoring logic here
            score = self.calculate_score(doc, query)
            
            results.append(ScoringResult(
                doc_id=doc["id"],
                stage_name=self.name,
                score=score,
                metadata={"my_param": self.my_param}
            ))
        return results
```

## Pipeline Assembly

```python
# Custom pipeline with multiple stages
pipeline = ScoringPipeline([
    VectorScoringStage(weight=0.6),
    BM25ScoringStage(weight=0.4),
    FusionStage(weights={"vector": 0.6, "bm25": 0.4}),
    ExactMatchStage(bonus=0.15),
    MyCustomStage(my_param=0.5),
    EnhancedRankingStage(ranker=enhanced_ranker)
])

# With debug mode
pipeline = ScoringPipeline(stages, config={"debug": True})
```

## Integration with HybridSearcher

```python
from utils.hybrid_search import get_hybrid_searcher

searcher = get_hybrid_searcher()

# Use pipeline-based search
results = searcher.search_with_pipeline(
    query="your search query",
    vector_results=vector_results,
    bm25_results=bm25_results,
    search_type="code",  # or "documentation", "config", "general"
    enhanced_ranker=ranker,
    context={"current_file": "/path/to/file.py"}
)
```

## Result Structure

```python
# Standard result
{
    "id": "file_path_0",
    "score": 0.875,  # Final score
    "pipeline_scores": {
        "vector": 0.8,
        "bm25": 0.7,
        "fusion": 0.77,
        "exact_match": 0.875
    }
}

# With debug mode enabled
{
    "id": "file_path_0",
    "score": 0.875,
    "pipeline_scores": {...},
    "pipeline_metadata": {
        "vector": {"similarity_type": "cosine"},
        "bm25": {"tokenization": "code_aware"},
        "fusion": {
            "breakdown": {
                "vector": {"raw": 0.8, "weighted": 0.56, "weight": 0.7},
                "bm25": {"raw": 0.7, "weighted": 0.21, "weight": 0.3}
            }
        },
        "exact_match": {"match_type": "exact_phrase"}
    }
}
```

## Common Patterns

### Boosting Recent Files
```python
class RecencyBoostStage(ScoringStage):
    def score(self, query, documents, context):
        results = []
        current_time = time.time()
        for doc in documents:
            age_days = (current_time - doc.get("modified_at", 0)) / 86400
            boost = max(0, 0.1 * (1 - age_days / 365))  # Decay over 1 year
            
            current_score = context["stage_scores"].get("fusion", {}).get(doc["id"], 0)
            results.append(ScoringResult(
                doc_id=doc["id"],
                stage_name=self.name,
                score=current_score + boost,
                metadata={"age_days": age_days, "boost": boost}
            ))
        return results
```

### Language-Specific Boosting
```python
class LanguageBoostStage(ScoringStage):
    def __init__(self, preferred_language: str, boost: float = 0.1):
        super().__init__("language_boost")
        self.preferred_language = preferred_language
        self.boost = boost
        
    def score(self, query, documents, context):
        results = []
        for doc in documents:
            is_preferred = doc.get("language") == self.preferred_language
            boost = self.boost if is_preferred else 0.0
            
            current_score = context["stage_scores"].get("fusion", {}).get(doc["id"], 0)
            results.append(ScoringResult(
                doc_id=doc["id"],
                stage_name=self.name,
                score=current_score + boost,
                metadata={"language": doc.get("language"), "boosted": is_preferred}
            ))
        return results
```

### Conditional Stage Execution
```python
class ConditionalExactMatchStage(ExactMatchStage):
    def score(self, query, documents, context):
        # Only apply exact match for short queries
        if len(query.split()) > 5:
            # Pass through scores unchanged
            return [
                ScoringResult(
                    doc_id=doc["id"],
                    stage_name=self.name,
                    score=context["stage_scores"].get("fusion", {}).get(doc["id"], 0),
                    metadata={"skipped": True, "reason": "query_too_long"}
                )
                for doc in documents
            ]
        
        # Normal exact match logic
        return super().score(query, documents, context)
```

## Tips & Tricks

1. **Stage Order Matters**: Dependent stages must come after their dependencies
2. **Access Previous Scores**: Use `context["stage_scores"][stage_name][doc_id]`
3. **Normalize Scores**: Keep scores in [0, 1] range for consistency
4. **Handle Missing Data**: Use `.get()` with defaults for optional fields
5. **Debug Selectively**: Enable debug mode only when needed to save memory

## Troubleshooting

### Common Issues

**Issue**: Scores seem too low/high
- Check weight normalization in FusionStage
- Verify input scores are in expected range
- Enable debug mode to see score breakdown

**Issue**: Custom stage not affecting results
- Ensure stage is added to pipeline
- Check that stage is accessing correct previous scores
- Verify score calculation logic

**Issue**: Memory usage too high
- Disable debug mode in production
- Limit number of documents processed
- Optimize custom stage implementations

### Debugging Commands

```python
# Print pipeline configuration
print(pipeline.get_config())

# Inspect stage scores
for doc in results[:3]:
    print(f"Document: {doc['id']}")
    print(f"Final Score: {doc['score']}")
    print(f"Stage Scores: {doc.get('pipeline_scores', {})}")
    
# Check specific stage metadata
if "pipeline_metadata" in results[0]:
    print(results[0]["pipeline_metadata"].get("fusion", {}))