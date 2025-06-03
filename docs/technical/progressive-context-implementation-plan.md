# Progressive Context Management Implementation Plan (v0.3.2)

## Overview

Progressive Context Management will deliver 50-70% token reduction for initial queries by implementing a multi-level context API that provides file → class → method hierarchy with semantic caching.

## Goals

1. **Reduce Initial Token Consumption**: 50-70% reduction for high-level queries
2. **Enable Progressive Detail Expansion**: Start with summaries, expand on demand
3. **Implement Semantic Caching**: Reuse similar query results
4. **Provide Hierarchical Navigation**: File → class → method structure

## Architecture

### Core Components

#### 1. Progressive Context Manager
```python
class ProgressiveContextManager:
    """Manages multi-level context retrieval and caching."""
    
    def __init__(self, qdrant_client, embeddings, cache_config):
        self.qdrant_client = qdrant_client
        self.embeddings = embeddings
        self.semantic_cache = SemanticCache(cache_config)
        self.hierarchy_builder = HierarchyBuilder()
        
    def get_progressive_context(self, query: str, level: str = "file") -> ProgressiveResult:
        """Get context at specified level with expansion options."""
        # Check semantic cache first
        cached = self.semantic_cache.get_similar(query)
        if cached:
            return cached
            
        # Perform search at requested level
        results = self._search_at_level(query, level)
        
        # Build hierarchical structure
        hierarchy = self.hierarchy_builder.build(results)
        
        # Cache and return
        progressive_result = ProgressiveResult(
            level=level,
            summary=self._generate_summary(results),
            hierarchy=hierarchy,
            expansion_options=self._get_expansion_options(hierarchy)
        )
        
        self.semantic_cache.store(query, progressive_result)
        return progressive_result
```

#### 2. Semantic Cache
```python
class SemanticCache:
    """Caches query results with semantic similarity matching."""
    
    def __init__(self, config):
        self.similarity_threshold = config.get("similarity_threshold", 0.85)
        self.max_cache_size = config.get("max_cache_size", 1000)
        self.cache_ttl = config.get("cache_ttl_seconds", 3600)
        self.cache = OrderedDict()
        self.embeddings_cache = {}
        
    def get_similar(self, query: str) -> Optional[ProgressiveResult]:
        """Find semantically similar cached queries."""
        query_embedding = self._get_embedding(query)
        
        for cached_query, (result, embedding, timestamp) in self.cache.items():
            if self._is_expired(timestamp):
                continue
                
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= self.similarity_threshold:
                # Update access time
                self.cache.move_to_end(cached_query)
                return result
                
        return None
```

#### 3. Hierarchy Builder
```python
class HierarchyBuilder:
    """Builds hierarchical code structure from search results."""
    
    def build(self, search_results: List[Dict]) -> CodeHierarchy:
        """Build file → class → method hierarchy."""
        hierarchy = CodeHierarchy()
        
        for result in search_results:
            file_path = result["file_path"]
            chunk_type = result.get("chunk_type", "code")
            
            # Extract hierarchical information from metadata
            if chunk_type == "class":
                hierarchy.add_class(
                    file_path,
                    class_name=result["metadata"]["name"],
                    summary=self._extract_summary(result)
                )
            elif chunk_type == "function":
                hierarchy.add_method(
                    file_path,
                    class_name=result["metadata"].get("parent_class"),
                    method_name=result["metadata"]["name"],
                    signature=result["metadata"].get("signature"),
                    summary=self._extract_summary(result)
                )
                
        return hierarchy
```

#### 4. Progressive Result Structure
```python
@dataclass
class ProgressiveResult:
    """Result structure for progressive context retrieval."""
    level: str  # "file", "class", "method"
    summary: str  # High-level summary of results
    hierarchy: CodeHierarchy  # Structured code hierarchy
    expansion_options: List[ExpansionOption]  # Available drill-downs
    token_estimate: int  # Estimated tokens for this result
    
@dataclass
class ExpansionOption:
    """Options for expanding context to more detail."""
    target_level: str  # Next level down
    target_path: str  # Specific file/class/method to expand
    estimated_tokens: int  # Additional tokens if expanded
    relevance_score: float  # How relevant this expansion might be
```

### Context Levels

1. **File Level** (Highest - 70% token reduction)
   - Returns file summaries and main components
   - Lists classes/functions without implementation
   - Provides import/export relationships

2. **Class Level** (Medium - 50% token reduction)
   - Returns class definitions with method signatures
   - Includes docstrings and type hints
   - Excludes method implementations

3. **Method Level** (Detailed - 20% token reduction)
   - Returns full method implementations
   - Includes surrounding context
   - Similar to current search results

### MCP Tool Integration

#### New MCP Tools

1. **search_progressive**
```python
@mcp_tool
async def search_progressive(
    query: str,
    level: str = "file",
    n_results: int = 5,
    expand_paths: Optional[List[str]] = None
) -> ProgressiveSearchResult:
    """
    Search with progressive context management.
    
    Args:
        query: Search query
        level: Context level ("file", "class", "method")
        n_results: Number of results
        expand_paths: Specific paths to expand to next level
    
    Returns:
        Progressive search results with expansion options
    """
```

2. **expand_context**
```python
@mcp_tool
async def expand_context(
    expansion_option: ExpansionOption,
    include_dependencies: bool = False
) -> ProgressiveResult:
    """
    Expand a specific context to more detail.
    
    Args:
        expansion_option: The expansion to perform
        include_dependencies: Include related code
        
    Returns:
        Expanded context at the next level
    """
```

3. **get_code_hierarchy**
```python
@mcp_tool
async def get_code_hierarchy(
    file_paths: List[str],
    max_depth: int = 3
) -> CodeHierarchy:
    """
    Get hierarchical structure of code files.
    
    Args:
        file_paths: Files to analyze
        max_depth: Maximum hierarchy depth
        
    Returns:
        Hierarchical code structure
    """
```

### Query Intent Classification

```python
class QueryIntentClassifier:
    """Classifies query intent to determine appropriate context level."""
    
    def classify(self, query: str) -> QueryIntent:
        """Classify the query intent."""
        query_lower = query.lower()
        
        # High-level exploration patterns
        if any(pattern in query_lower for pattern in [
            "what does", "explain", "overview", "structure",
            "architecture", "how does", "purpose of"
        ]):
            return QueryIntent(
                level="file",
                exploration_type="understanding",
                confidence=0.8
            )
            
        # Implementation-specific patterns
        if any(pattern in query_lower for pattern in [
            "implementation", "bug in", "error in", "fix",
            "line", "specific", "exact"
        ]):
            return QueryIntent(
                level="method",
                exploration_type="debugging",
                confidence=0.9
            )
            
        # Default to class level
        return QueryIntent(
            level="class",
            exploration_type="navigation",
            confidence=0.6
        )
```

## Implementation Steps

### Phase 1: Core Infrastructure (Days 1-3)

1. **Create Progressive Context Manager**
   - Implement basic multi-level search
   - Add level-aware result filtering
   - Create summary generation logic

2. **Build Hierarchy System**
   - Implement CodeHierarchy data structure
   - Add hierarchy extraction from AST metadata
   - Create hierarchy navigation methods

3. **Add Basic MCP Tools**
   - Implement search_progressive tool
   - Add expand_context tool
   - Update existing search to use progressive manager

### Phase 2: Semantic Caching (Days 4-6)

1. **Implement Semantic Cache**
   - Create embedding-based cache lookup
   - Add similarity threshold configuration
   - Implement cache expiration and size limits

2. **Add Cache Warming**
   - Pre-populate cache with common queries
   - Implement background cache updates
   - Add cache hit tracking

3. **Optimize Cache Performance**
   - Use FAISS for fast similarity search
   - Implement cache persistence
   - Add cache metrics

### Phase 3: Query Intelligence (Days 7-8)

1. **Build Query Classifier**
   - Implement pattern-based classification
   - Add confidence scoring
   - Create feedback mechanism

2. **Integrate with Search**
   - Auto-select appropriate level
   - Provide level override options
   - Add explanation for level selection

### Phase 4: Integration & Testing (Days 9-10)

1. **Update Existing Tools**
   - Modify search tools to support progressive mode
   - Add backward compatibility
   - Update documentation

2. **Performance Testing**
   - Measure token reduction
   - Benchmark cache performance
   - Validate hierarchy accuracy

3. **User Experience**
   - Add clear expansion indicators
   - Provide token estimates
   - Create usage examples

## Configuration

Add to `server_config.json`:

```json
{
  "progressive_context": {
    "enabled": true,
    "default_level": "class",
    "cache": {
      "similarity_threshold": 0.85,
      "max_cache_size": 1000,
      "ttl_seconds": 3600,
      "persistence_enabled": true,
      "persistence_path": "~/.mcp-servers/qdrant-rag/cache"
    },
    "levels": {
      "file": {
        "include_summaries": true,
        "max_summary_length": 500,
        "include_structure": true
      },
      "class": {
        "include_signatures": true,
        "include_docstrings": true,
        "exclude_private": false
      },
      "method": {
        "include_implementation": true,
        "context_lines": 10
      }
    },
    "query_classification": {
      "enabled": true,
      "confidence_threshold": 0.7,
      "fallback_level": "class"
    }
  }
}
```

## Usage Examples

### Example 1: High-Level Exploration
```
User: "What does the authentication system do?"

System uses progressive search at file level:
- Returns file summaries for auth-related files
- Shows main classes without implementation
- Estimates: 500 tokens (vs 3000 with full search)

User can expand specific files/classes for more detail.
```

### Example 2: Specific Implementation
```
User: "Show the bug in the login validation"

System uses method level search:
- Returns full implementation of validation methods
- Includes surrounding context
- Similar to current search behavior
```

### Example 3: Progressive Exploration
```
User: "How does the payment processing work?"

1. Initial file-level results (500 tokens)
2. User expands PaymentProcessor class (800 tokens) 
3. User expands processPayment method (400 tokens)

Total: 1700 tokens (vs 5000 tokens if all returned initially)
```

## Success Metrics

1. **Token Reduction**
   - Target: 50-70% for high-level queries
   - Measure: Average tokens per query by type

2. **Cache Performance**
   - Target: 40% cache hit rate
   - Measure: Cache hits vs misses

3. **User Satisfaction**
   - Target: 80% use progressive features
   - Measure: Feature adoption rate

4. **Search Quality**
   - Target: No degradation in result relevance
   - Measure: Result accuracy scores

## Risks & Mitigations

1. **Risk**: Summary quality affects usefulness
   - **Mitigation**: Use AST metadata for accurate summaries

2. **Risk**: Cache invalidation complexity
   - **Mitigation**: Time-based expiry + file change detection

3. **Risk**: User confusion with levels
   - **Mitigation**: Clear UI indicators and auto-selection

4. **Risk**: Performance overhead
   - **Mitigation**: Async operations and efficient caching

## Future Enhancements

1. **Learned Summarization**: Use small LLM for better summaries
2. **Adaptive Levels**: Learn user preferences over time
3. **Cross-Project Cache**: Share cache across projects
4. **Streaming Expansion**: Progressive loading as user scrolls