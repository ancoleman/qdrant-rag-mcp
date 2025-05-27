# Implementation Guide: Multi-Signal Hybrid Search

This guide details how to implement hybrid search combining semantic vectors, BM25 keyword search, and dependency-aware retrieval for 45% better precision.

## Overview

Hybrid search combines multiple retrieval signals to overcome the limitations of pure semantic search:
- **Semantic search**: Good for concepts and similar meanings
- **BM25 keyword search**: Excellent for exact function/variable names
- **Dependency tracking**: Finds related code through imports/calls

## Current Limitations

Our current pure semantic search misses:
1. Exact function name matches (developer searches for `processPayment`)
2. Variable names and specific identifiers
3. Code that's functionally related but semantically different
4. Import relationships and dependencies

## Proposed Architecture

```
Query → Query Analyzer → Multi-Signal Retrieval → Fusion → Re-ranking → Results
           ↓                    ↓
      Classify Intent    ┌──────┴───────┐
           ↓             ↓              ↓
      Weight Signals  Semantic    BM25 Keyword    Dependency
                      Search       Search          Graph
```

## Implementation Plan

### Step 1: Add BM25 Index Alongside Vectors

```python
# src/indexers/bm25_indexer.py
import math
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
import numpy as np

class BM25Index:
    """BM25 implementation for code search"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = defaultdict(int)
        self.doc_lengths = {}
        self.avg_doc_length = 0
        self.corpus_size = 0
        self.doc_vectors = {}  # doc_id -> term frequencies
        self.vocabulary = set()
        
    def tokenize_code(self, text: str) -> List[str]:
        """Tokenize code content for BM25"""
        import re
        
        # Split on word boundaries, keep underscores
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Also split camelCase and snake_case
        expanded_tokens = []
        for token in tokens:
            # Split camelCase
            camel_split = re.sub('([A-Z][a-z]+)', r' \1', 
                                re.sub('([A-Z]+)', r' \1', token)).split()
            expanded_tokens.extend(camel_split)
            
            # Split snake_case
            snake_split = token.split('_')
            expanded_tokens.extend(snake_split)
        
        # Remove empty tokens and duplicates
        tokens = list(set(t.lower() for t in expanded_tokens if t))
        
        return tokens
    
    def add_document(self, doc_id: str, content: str):
        """Add document to BM25 index"""
        tokens = self.tokenize_code(content)
        
        # Update document frequency
        token_freqs = Counter(tokens)
        self.doc_vectors[doc_id] = token_freqs
        
        # Update vocabulary
        self.vocabulary.update(tokens)
        
        # Update document frequency for each unique token
        for token in set(tokens):
            self.doc_freqs[token] += 1
        
        # Update document length
        self.doc_lengths[doc_id] = len(tokens)
        self.corpus_size += 1
        
        # Update average document length
        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
    
    def get_scores(self, query: str, doc_ids: List[str] = None) -> Dict[str, float]:
        """Calculate BM25 scores for query"""
        query_tokens = self.tokenize_code(query)
        scores = {}
        
        if doc_ids is None:
            doc_ids = self.doc_vectors.keys()
        
        for doc_id in doc_ids:
            if doc_id not in self.doc_vectors:
                continue
                
            score = 0.0
            doc_vector = self.doc_vectors[doc_id]
            doc_length = self.doc_lengths[doc_id]
            
            for token in query_tokens:
                if token not in self.vocabulary:
                    continue
                
                # Term frequency in document
                tf = doc_vector.get(token, 0)
                
                # Document frequency
                df = self.doc_freqs.get(token, 0)
                
                # IDF calculation
                idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 formula
                numerator = idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                
                score += numerator / denominator
            
            scores[doc_id] = score
        
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search documents using BM25"""
        scores = self.get_scores(query)
        
        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]
```

### Step 2: Build Dependency Graph

```python
# src/indexers/dependency_graph.py
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import ast
import re

@dataclass
class CodeEntity:
    """Represents a code entity (file, class, function)"""
    id: str
    type: str  # 'file', 'class', 'function', 'variable'
    name: str
    file_path: str
    imports: List[str]
    exports: List[str]
    calls: List[str]
    references: List[str]

class DependencyGraph:
    """Track dependencies between code entities"""
    
    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> imported files
        self.export_graph: Dict[str, Set[str]] = defaultdict(set)  # file -> exported entities
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)    # entity -> called entities
        self.reference_graph: Dict[str, Set[str]] = defaultdict(set)  # entity -> referenced
        
    def add_file(self, file_path: str, content: str, language: str):
        """Analyze file and add to dependency graph"""
        if language == 'python':
            self._analyze_python_file(file_path, content)
        elif language in ['javascript', 'typescript']:
            self._analyze_js_file(file_path, content)
        # Add more languages as needed
    
    def _analyze_python_file(self, file_path: str, content: str):
        """Analyze Python file for dependencies"""
        try:
            tree = ast.parse(content)
        except:
            return
        
        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    self.import_graph[file_path].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    full_import = f"{module}.{alias.name}" if module else alias.name
                    imports.append(full_import)
                    self.import_graph[file_path].add(module)
        
        # Extract exports (top-level definitions)
        exports = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                exports.append(node.name)
                self.export_graph[file_path].add(node.name)
                
                # Create entity
                entity_id = f"{file_path}:{node.name}"
                entity = CodeEntity(
                    id=entity_id,
                    type='function' if isinstance(node, ast.FunctionDef) else 'class',
                    name=node.name,
                    file_path=file_path,
                    imports=imports,
                    exports=[],
                    calls=self._extract_calls(node),
                    references=[]
                )
                self.entities[entity_id] = entity
        
        # File entity
        file_entity = CodeEntity(
            id=file_path,
            type='file',
            name=file_path.split('/')[-1],
            file_path=file_path,
            imports=imports,
            exports=exports,
            calls=[],
            references=[]
        )
        self.entities[file_path] = file_entity
    
    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls from AST node"""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # Handle method calls like obj.method()
                    if isinstance(child.func.value, ast.Name):
                        calls.append(f"{child.func.value.id}.{child.func.attr}")
        return calls
    
    def find_dependencies(self, entity_id: str, max_depth: int = 2) -> Set[str]:
        """Find all dependencies of an entity up to max_depth"""
        visited = set()
        to_visit = [(entity_id, 0)]
        dependencies = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get entity
            entity = self.entities.get(current_id)
            if not entity:
                continue
            
            # Add direct imports
            for imp in entity.imports:
                dependencies.add(imp)
                if depth < max_depth:
                    to_visit.append((imp, depth + 1))
            
            # Add called functions
            for call in entity.calls:
                # Try to resolve call to entity
                possible_entities = [
                    f"{entity.file_path}:{call}",
                    call  # Global reference
                ]
                for possible_id in possible_entities:
                    if possible_id in self.entities:
                        dependencies.add(possible_id)
                        if depth < max_depth:
                            to_visit.append((possible_id, depth + 1))
        
        return dependencies
    
    def find_dependents(self, entity_id: str) -> Set[str]:
        """Find all entities that depend on this entity"""
        dependents = set()
        
        entity = self.entities.get(entity_id)
        if not entity:
            return dependents
        
        # Check all entities for references
        for other_id, other_entity in self.entities.items():
            if entity.name in other_entity.calls:
                dependents.add(other_id)
            if entity.file_path in other_entity.imports:
                dependents.add(other_id)
        
        return dependents
```

### Step 3: Implement Reciprocal Rank Fusion

```python
# src/search/fusion.py
from typing import List, Dict, Tuple, Any
import numpy as np

class ReciprocRankFusion:
    """Combine rankings from multiple sources using RRF"""
    
    def __init__(self, k: int = 60):
        """
        Args:
            k: Constant for RRF formula (typically 60)
        """
        self.k = k
    
    def fuse(self, 
             rankings: Dict[str, List[Tuple[str, float]]],
             weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        Fuse multiple rankings using RRF
        
        Args:
            rankings: Dict of source_name -> [(doc_id, score), ...]
            weights: Optional weights for each source
            
        Returns:
            Fused ranking [(doc_id, fused_score), ...]
        """
        if weights is None:
            weights = {source: 1.0 for source in rankings}
        
        # Calculate RRF scores
        doc_scores = defaultdict(float)
        
        for source, ranking in rankings.items():
            weight = weights.get(source, 1.0)
            
            for rank, (doc_id, score) in enumerate(ranking):
                # RRF formula: 1 / (k + rank)
                rrf_score = weight / (self.k + rank + 1)
                doc_scores[doc_id] += rrf_score
        
        # Sort by fused score
        fused_ranking = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return fused_ranking

class AdaptiveRankFusion:
    """Advanced fusion that adapts weights based on query type"""
    
    def __init__(self):
        self.rrf = ReciprocRankFusion()
        self.query_patterns = {
            'exact_search': re.compile(r'"[^"]+"|\b\w+\(\)|\bclass\s+\w+|\bdef\s+\w+'),
            'concept_search': re.compile(r'\b(how|what|why|implement|design|pattern)\b', re.I),
            'dependency_search': re.compile(r'\b(import|require|depend|use|call)\b', re.I)
        }
    
    def classify_query(self, query: str) -> Dict[str, float]:
        """Classify query type and return weights"""
        weights = {
            'semantic': 0.4,
            'bm25': 0.4,
            'dependency': 0.2
        }
        
        # Adjust weights based on query patterns
        if self.query_patterns['exact_search'].search(query):
            weights['bm25'] = 0.6
            weights['semantic'] = 0.3
            weights['dependency'] = 0.1
        elif self.query_patterns['concept_search'].search(query):
            weights['semantic'] = 0.6
            weights['bm25'] = 0.2
            weights['dependency'] = 0.2
        elif self.query_patterns['dependency_search'].search(query):
            weights['dependency'] = 0.5
            weights['semantic'] = 0.3
            weights['bm25'] = 0.2
        
        return weights
    
    def fuse_adaptive(self, 
                      query: str,
                      rankings: Dict[str, List[Tuple[str, float]]]) -> List[Tuple[str, float]]:
        """Adaptively fuse rankings based on query type"""
        weights = self.classify_query(query)
        return self.rrf.fuse(rankings, weights)
```

### Step 4: Integrate Hybrid Search into MCP Server

```python
# Updates to src/qdrant_mcp_context_aware.py

class HybridSearcher:
    """Combines semantic, BM25, and dependency search"""
    
    def __init__(self, 
                 qdrant_client: QdrantClient,
                 embedding_model: Any,
                 bm25_index: BM25Index,
                 dep_graph: DependencyGraph):
        self.qdrant = qdrant_client
        self.embedder = embedding_model
        self.bm25 = bm25_index
        self.deps = dep_graph
        self.fusion = AdaptiveRankFusion()
    
    def search(self, 
               query: str, 
               collection: str,
               n_results: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search"""
        
        # 1. Semantic search
        query_vector = self.embedder.encode(query).tolist()
        semantic_results = self.qdrant.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=n_results * 3  # Get more for fusion
        )
        
        semantic_ranking = [
            (str(r.id), r.score) 
            for r in semantic_results
        ]
        
        # 2. BM25 search
        # Get all doc IDs from the collection for BM25
        doc_ids = [str(r.id) for r in semantic_results]
        bm25_scores = self.bm25.get_scores(query, doc_ids)
        bm25_ranking = sorted(
            bm25_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_results * 3]
        
        # 3. Dependency expansion
        dependency_scores = {}
        
        # For top semantic results, find dependencies
        for doc_id, score in semantic_ranking[:5]:
            deps = self.deps.find_dependencies(doc_id, max_depth=1)
            for dep_id in deps:
                if dep_id not in dependency_scores:
                    dependency_scores[dep_id] = 0
                # Propagate partial score
                dependency_scores[dep_id] += score * 0.3
        
        dependency_ranking = sorted(
            dependency_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_results]
        
        # 4. Fuse rankings
        rankings = {
            'semantic': semantic_ranking,
            'bm25': bm25_ranking,
            'dependency': dependency_ranking
        }
        
        fused_results = self.fusion.fuse_adaptive(query, rankings)
        
        # 5. Fetch full results
        final_results = []
        for doc_id, fused_score in fused_results[:n_results]:
            # Get document from Qdrant
            doc = self.qdrant.retrieve(
                collection_name=collection,
                ids=[doc_id]
            )[0]
            
            result = {
                'id': doc_id,
                'score': fused_score,
                'content': doc.payload.get('content', ''),
                'metadata': doc.payload,
                'scores': {
                    'semantic': next((s for d, s in semantic_ranking if d == doc_id), 0),
                    'bm25': bm25_scores.get(doc_id, 0),
                    'dependency': dependency_scores.get(doc_id, 0)
                }
            }
            final_results.append(result)
        
        return final_results

# Update the MCP tool
@mcp.tool()
def search_hybrid(
    query: str,
    n_results: int = 5,
    search_mode: str = "auto",  # 'auto', 'semantic', 'keyword', 'hybrid'
    include_dependencies: bool = True,
    cross_project: bool = False
) -> Dict[str, Any]:
    """
    Enhanced hybrid search combining multiple signals
    
    Args:
        query: Search query
        n_results: Number of results to return
        search_mode: Force specific search mode or use auto
        include_dependencies: Include dependency expansion
        cross_project: Search across all projects
    """
    # Get services
    hybrid_searcher = get_hybrid_searcher()
    
    # Determine collections to search
    collections = determine_search_collections(cross_project)
    
    all_results = []
    
    for collection in collections:
        if search_mode == "semantic":
            # Semantic only
            results = semantic_search(query, collection, n_results)
        elif search_mode == "keyword":
            # BM25 only
            results = bm25_search(query, collection, n_results)
        else:
            # Full hybrid
            results = hybrid_searcher.search(query, collection, n_results)
        
        all_results.extend(results)
    
    # Sort by score and limit
    all_results.sort(key=lambda x: x['score'], reverse=True)
    final_results = all_results[:n_results]
    
    return {
        'results': final_results,
        'query': query,
        'mode': search_mode,
        'total': len(final_results),
        'search_signals': {
            'semantic': True,
            'keyword': search_mode in ['keyword', 'hybrid', 'auto'],
            'dependencies': include_dependencies and search_mode != 'keyword'
        }
    }
```

### Step 5: Update Indexing to Build All Indices

```python
# Update indexing functions

def index_code_hybrid(file_path: str, content: str):
    """Index code for hybrid search"""
    
    # 1. Existing vector indexing
    chunks = create_chunks(file_path, content)
    embeddings = embed_chunks(chunks)
    store_in_qdrant(chunks, embeddings)
    
    # 2. Add to BM25 index
    bm25_index = get_bm25_index()
    for chunk in chunks:
        doc_id = chunk['metadata']['chunk_id']
        bm25_index.add_document(doc_id, chunk['content'])
    
    # 3. Build dependency graph
    dep_graph = get_dependency_graph()
    language = detect_language(file_path)
    dep_graph.add_file(file_path, content, language)
    
    # 4. Link chunks to dependency graph
    for chunk in chunks:
        if 'structure_name' in chunk['metadata']:
            entity_id = f"{file_path}:{chunk['metadata']['structure_name']}"
            # Store mapping between chunk and entity
            store_chunk_entity_mapping(
                chunk['metadata']['chunk_id'],
                entity_id
            )
```

## Benefits of Hybrid Search

### 1. Improved Precision

```python
# Example: Search for "processPayment function"

# Pure semantic might return:
# - handleTransaction (semantically similar)
# - chargeCard (related concept)
# - validatePayment (related concept)

# Hybrid search returns:
# - processPayment (exact match via BM25)
# - PaymentProcessor.process (dependency graph)
# - processPaymentAsync (BM25 partial match)
```

### 2. Better Recall

- Semantic search finds conceptually related code
- BM25 finds exact matches users expect
- Dependency graph finds functionally related code

### 3. Query-Adaptive Results

Different query types get optimal handling:
- `"authentication"` → Semantic-heavy search
- `"getUserById()"` → BM25-heavy search  
- `"what calls processOrder"` → Dependency-heavy search

## Performance Considerations

### Storage
- BM25 index: ~20% of vector storage
- Dependency graph: ~10MB per 10K files
- Total overhead: ~30% additional storage

### Search Latency
- Parallel execution of different searches
- Fusion adds ~5-10ms
- Total latency increase: ~20%

### Indexing
- BM25 indexing: Very fast
- Dependency analysis: ~50ms per file
- Can be done in parallel with embedding

## Configuration Options

```python
# Recommended configurations

# For code-heavy repositories
HYBRID_WEIGHTS = {
    'semantic': 0.3,
    'bm25': 0.5,
    'dependency': 0.2
}

# For documentation-heavy repositories  
HYBRID_WEIGHTS = {
    'semantic': 0.6,
    'bm25': 0.3,
    'dependency': 0.1
}

# For framework/library code
HYBRID_WEIGHTS = {
    'semantic': 0.3,
    'bm25': 0.3,
    'dependency': 0.4
}
```

## Next Steps

1. Implement language-specific tokenizers for BM25
2. Add support for more dependency patterns
3. Build query performance tracking
4. Create benchmarks for different query types
5. Add configuration UI for weight tuning

This hybrid search implementation would provide the 45% improvement in retrieval precision mentioned in the research, making the RAG system much more effective for real-world code search scenarios.