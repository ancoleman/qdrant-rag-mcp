# Search Scoring Analysis

## Executive Summary

Our search system is producing lower-than-expected scores (0.5-0.6 range) when searching for exact terms that should yield high confidence matches (0.8-0.9 range). This analysis examines the root causes and proposes solutions.

## Current Scoring Behavior

### Example Query: "bm25_manager search"

**Expected**: High scores (0.8+) for exact matches of these terms
**Actual**: Scores in 0.5-0.6 range

#### Score Breakdown:
- **Vector Search Only**: 0.60-0.68
- **Keyword Search Only**: 1.0, 0.5, 0.333 (reciprocal rank scores)
- **Hybrid Search**: 0.603 (after linear combination and enhancement)

## Root Causes

### 1. Chunking Strategy Limitations

**Issue**: Code is split into small chunks that separate related concepts
- Classes are separated from their methods
- Methods are separated from their implementations
- Context is lost between chunks

**Example**:
```
Chunk 1: class BM25Manager
Chunk 2: def search(self, ...)
```

These should be together for "bm25_manager search" to match well.

### 2. BM25 Scoring Limitations

**Issue**: BM25 returns reciprocal rank scores (1/rank) not true relevance scores
- Rank 1: 1.0
- Rank 2: 0.5
- Rank 3: 0.333

**Problem**: These don't align well with vector similarity scores (0.0-1.0 continuous)

### 3. Vector Similarity Gaps

**Issue**: Generic embeddings don't understand code semantics well
- "bm25_manager" → "BM25Manager" similarity is only ~0.68
- Camel case, underscores, and code conventions reduce similarity
- Generic models weren't trained on code structure

### 4. Linear Combination Weights

**Current**: 0.7 vector + 0.3 BM25

**Issue**: These weights may not be optimal for code search where exact matches matter more

### 5. Enhanced Ranking Penalties

**Issue**: Additional signals sometimes reduce scores rather than boost them
- Code structure: 0.3 for class definitions (low)
- Recency: 0.0 for older files
- File proximity: Only helps if searching from within same file

## Impact on Search Quality

1. **Exact matches don't surface well**: Terms that should be obvious matches get mediocre scores
2. **User confidence is reduced**: Scores of 0.5-0.6 suggest uncertain matches
3. **Progressive search suffers**: Low base scores mean progressive context shows weak matches
4. **Ranking is less reliable**: Similar scores make it hard to distinguish quality

## Score Pipeline Analysis

```
1. Vector Search: "bm25_manager search" → 0.68
2. BM25 Search: "bm25_manager search" → 0.5 (rank 2)  
3. Linear Combination: (0.7 × 0.68) + (0.3 × 0.5) = 0.626
4. Enhanced Ranking: 0.626 → 0.603 (penalties applied)
```

Each step reduces confidence rather than reinforcing good matches.

## Recommendations

### Immediate Improvements (Low Effort)

1. **Adjust Linear Combination Weights**
   - For code search: 0.5 vector + 0.5 BM25
   - For docs search: 0.8 vector + 0.2 BM25
   - Make weights configurable per search type

2. **Normalize BM25 Scores**
   - Convert reciprocal rank to 0-1 scale: `score = 1 - (1 / (rank + 1))`
   - Or use actual BM25 scores if available from the retriever

3. **Tune Enhanced Ranking**
   - Increase base score weight to 0.6
   - Make enhancement additive (bonus) rather than multiplicative

### Medium-term Improvements (Moderate Effort)

1. **Implement Smarter Chunking**
   - Keep classes with their methods
   - Keep function signatures with bodies
   - Use AST-aware chunking boundaries

2. **Add Exact Match Boosting**
   - If query terms appear exactly in content, boost score by 0.2
   - Implement phrase matching in BM25

3. **Context-Aware Scoring**
   - If searching for code, prioritize code patterns
   - If searching for docs, prioritize natural language

### Long-term Improvements (High Effort)

1. **Code-Specific Embeddings**
   - Use CodeBERT or similar code-trained models
   - Fine-tune on your codebase

2. **Learning to Rank**
   - Collect user feedback on result quality
   - Train a model to predict good matches

3. **Semantic Code Understanding**
   - Parse code structure (imports, calls, definitions)
   - Build a code graph for better relevance

## Proposed Scoring Formula

```python
# Immediate fix
def improved_score(vector_score, bm25_rank, query, content):
    # Normalize BM25
    bm25_score = 1.0 - (1.0 / (bm25_rank + 2))
    
    # Exact match bonus
    exact_match_bonus = 0.2 if all(term in content.lower() for term in query.lower().split()) else 0
    
    # Base combination
    base_score = (0.5 * vector_score) + (0.5 * bm25_score)
    
    # Add bonus
    final_score = min(1.0, base_score + exact_match_bonus)
    
    return final_score
```

## Expected Outcomes

With these improvements:
- Exact matches: 0.8-0.95 scores
- Good matches: 0.6-0.8 scores  
- Weak matches: 0.3-0.6 scores
- Poor matches: <0.3 scores

This aligns better with user expectations and makes scores more interpretable.