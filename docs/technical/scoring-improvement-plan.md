# Scoring Improvement Implementation Plan

## Phase 1: Immediate Fixes (1-2 days)

### 1. Normalize BM25 Scores
**File**: `src/utils/hybrid_search.py`

```python
def search(self, collection_name: str, query: str, k: int = 5, qdrant_client=None) -> List[Tuple[str, float]]:
    # ... existing code ...
    
    # Instead of reciprocal rank:
    # score = 1.0 / (rank + 1)
    
    # Use normalized score:
    score = 1.0 - (1.0 / (rank + 2))  # Gives 0.67, 0.75, 0.80 for ranks 1,2,3
```

### 2. Configure Weights by Search Type
**File**: `config/server_config.json`

```json
{
  "hybrid_search": {
    "weights": {
      "code": {
        "vector": 0.5,
        "bm25": 0.5
      },
      "documentation": {
        "vector": 0.8,
        "bm25": 0.2
      },
      "config": {
        "vector": 0.6,
        "bm25": 0.4
      }
    }
  }
}
```

### 3. Add Exact Match Detection
**File**: `src/utils/hybrid_search.py`

```python
def linear_combination_with_exact_match(
    self,
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]], 
    query: str,
    documents: Dict[str, str],  # doc_id -> content
    vector_weight: float = 0.5,
    bm25_weight: float = 0.5
):
    # ... existing combination logic ...
    
    # Add exact match bonus
    query_terms = query.lower().split()
    for result in results:
        content = documents.get(result.content, "").lower()
        if all(term in content for term in query_terms):
            result.combined_score = min(1.0, result.combined_score + 0.2)
```

## Phase 2: Structural Improvements (1 week)

### 1. Improve Chunking Strategy
**Approach**: Keep related code together

- Combine class definitions with their methods
- Keep function signatures with first few lines of implementation
- Maintain import context with code that uses it

### 2. Enhanced BM25 Implementation
**Options**:
- Use Elasticsearch-style BM25 with configurable parameters
- Implement BM25+ or BM25L variants
- Add phrase matching support

### 3. Scoring Pipeline Refactor
Create a configurable scoring pipeline:

```python
class ScoringPipeline:
    def __init__(self, stages: List[ScoringStage]):
        self.stages = stages
    
    def score(self, query, results):
        for stage in self.stages:
            results = stage.apply(query, results)
        return results

# Usage:
pipeline = ScoringPipeline([
    VectorScoringStage(),
    BM25ScoringStage(),
    FusionStage(weights=config),
    ExactMatchStage(bonus=0.2),
    EnhancedRankingStage()
])
```

## Phase 3: Advanced Improvements (2-4 weeks)

### 1. Code-Aware Embeddings
**Options**:
- CodeBERT: Microsoft's code-trained BERT
- GraphCodeBERT: Includes data flow  
- CodeT5: Salesforce's code generation model
- StarCoder: Recent open model for code

**Implementation**:
```python
# config/server_config.json
{
  "embedding_models": {
    "code": "microsoft/codebert-base",
    "documentation": "sentence-transformers/all-mpnet-base-v2",
    "default": "all-MiniLM-L12-v2"
  }
}
```

### 2. Query Understanding
Detect query intent and adjust scoring:

```python
class QueryClassifier:
    def classify(self, query: str) -> QueryType:
        # Detect patterns
        if re.match(r'\w+\.\w+', query):  # method.call
            return QueryType.METHOD_SEARCH
        elif 'class' in query or 'def' in query:
            return QueryType.DEFINITION_SEARCH
        elif 'error' in query or 'bug' in query:
            return QueryType.DEBUG_SEARCH
        else:
            return QueryType.GENERAL_SEARCH
```

### 3. Feedback Loop
Track which results users actually use:

```python
@mcp.tool()
def mark_result_useful(
    query: str,
    result_id: str, 
    useful: bool
):
    """Track search result effectiveness"""
    # Store feedback
    # Adjust future rankings
```

## Testing Strategy

### 1. Benchmark Queries
Create a test set with expected scores:

```python
test_queries = [
    {
        "query": "BM25Manager search",
        "expected_top_result": "src/utils/hybrid_search.py::BM25Manager.search",
        "expected_min_score": 0.85
    },
    {
        "query": "initialize embeddings",
        "expected_top_result": "src/utils/embeddings.py::initialize",
        "expected_min_score": 0.75
    }
]
```

### 2. A/B Testing
Run both scoring methods in parallel:

```python
@mcp.tool()
def search_with_comparison(query: str, ...):
    old_results = search_with_old_scoring(query)
    new_results = search_with_new_scoring(query)
    
    return {
        "results": new_results,
        "scoring_comparison": {
            "old_top_score": old_results[0]["score"],
            "new_top_score": new_results[0]["score"],
            "score_improvement": new_results[0]["score"] - old_results[0]["score"]
        }
    }
```

## Success Metrics

1. **Score Distribution**:
   - Exact matches: 0.85-0.95
   - Strong matches: 0.70-0.85
   - Good matches: 0.50-0.70
   - Weak matches: <0.50

2. **User Satisfaction**:
   - Top result is correct 80% of time
   - Desired result in top 3: 95% of time

3. **Performance**:
   - Scoring adds <10ms latency
   - No significant memory increase

## Rollout Plan

1. **Week 1**: Implement Phase 1 fixes
2. **Week 2**: Test and tune weights
3. **Week 3-4**: Implement Phase 2 improvements
4. **Month 2**: Begin Phase 3 if needed

## Rollback Strategy

All changes should be feature-flagged:

```json
{
  "scoring_features": {
    "use_normalized_bm25": true,
    "use_exact_match_bonus": true,
    "use_new_weights": true,
    "use_code_embeddings": false
  }
}
```

This allows quick rollback if issues arise.