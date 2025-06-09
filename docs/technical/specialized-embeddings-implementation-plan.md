# Specialized Embeddings Implementation Plan for v0.3.3

## 🎯 Implementation Status: ✅ COMPLETE

**Last Updated**: 2025-06-04  
**Completed**: 2025-06-04

### ✅ Completed Features
- Created `SpecializedEmbeddingManager` with LRU eviction and memory management
- Built central model registry with persistence
- Updated `embeddings.py` with `UnifiedEmbeddingsManager` for backward compatibility
- Enhanced collection management with model metadata storage
- Updated all indexing functions to use specialized embeddings
- Updated all search functions with content-type aware embeddings
- Implemented memory optimization with configurable limits
- Model compatibility checking with dimension validation
- Comprehensive unit tests (38 tests across 3 test files)
- Enhanced model download scripts with specialized model support
- Updated changelog with v0.3.3 release notes

### 📋 Deferred to Future Releases
- Migration utilities for existing collections (users will recreate collections)
- Performance benchmarking tools

## Overview

This document details the implementation plan for adding specialized embedding models to the Qdrant RAG MCP server. This feature, originally scheduled for v0.6.0, is being fast-tracked to v0.3.3 based on the significant performance improvements it will provide (30-60% better retrieval precision).

## Target Models

Based on extensive research and optimization for Apple M3 Pro with 18GB RAM, we will implement three specialized embedding models:

### 1. Code Embeddings: `nomic-ai/CodeRankEmbed`
- **Dimensions**: 768
- **Memory**: ~2GB RAM
- **Context Length**: 8192 tokens (32x improvement)
- **Performance**: 50-80% better code search accuracy
- **Languages**: Python, JavaScript, Java, Go, PHP, Ruby
- **Fallback**: `CodeBERTa-small-v1` (84MB, ~800MB RAM)

### 2. Configuration Embeddings: `jinaai/jina-embeddings-v3`
- **Dimensions**: 1024
- **Memory**: ~2GB RAM
- **Context Length**: 8192 tokens
- **Performance**: 15-20% improvement for config similarity
- **Formats**: JSON, YAML, TOML, XML
- **Fallback**: `jinaai/jina-embeddings-v2-base-en` (~1GB RAM)

### 3. Documentation Embeddings: `hkunlp/instructor-large`
- **Dimensions**: 768
- **Memory**: ~1.5GB RAM
- **Features**: Instruction-based prompting
- **Performance**: 30% improvement on technical terminology
- **Speed**: 35-45 tokens/second on M3 Pro
- **Fallback**: `sentence-transformers/all-mpnet-base-v2`

### 4. General/Fallback: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Purpose**: Backward compatibility and unknown content types

## Architecture Design

### 1. Enhanced Embedding Manager

```python
# src/utils/specialized_embeddings.py
from collections import OrderedDict
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
import torch

class SpecializedEmbeddingManager:
    """Manages multiple specialized embedding models with lazy loading"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.model_configs = {
            'code': {
                'name': 'nomic-ai/CodeRankEmbed',
                'dimension': 768,
                'fallback': 'CodeBERTa-small-v1',
                'max_tokens': 8192
            },
            'config': {
                'name': 'jinaai/jina-embeddings-v3',
                'dimension': 1024,
                'fallback': 'jinaai/jina-embeddings-v2-base-en',
                'max_tokens': 8192
            },
            'documentation': {
                'name': 'hkunlp/instructor-large',
                'dimension': 768,
                'fallback': 'sentence-transformers/all-mpnet-base-v2',
                'instruction_prefix': "Represent the technical documentation for retrieval:",
                'max_tokens': 512
            },
            'general': {
                'name': 'sentence-transformers/all-MiniLM-L6-v2',
                'dimension': 384,
                'max_tokens': 256
            }
        }
        
        # LRU cache for loaded models
        self.loaded_models = OrderedDict()
        self.max_models_in_memory = 3  # Configurable
        self.memory_limit_gb = 7.0  # Total memory limit
        
    def encode(self, texts: List[str], content_type: str = 'general') -> np.ndarray:
        """Encode texts using the appropriate model"""
        model = self.load_model(content_type)
        return model.encode(texts)
```

### 2. Model Registry

```python
# src/utils/model_registry.py
class ModelRegistry:
    """Central registry for model configurations and mappings"""
    
    def __init__(self):
        self.models = {}
        self.collection_model_mapping = {}
        
    def register_model(self, content_type: str, model_config: Dict[str, Any]):
        """Register a model configuration"""
        self.models[content_type] = model_config
        
    def get_model_for_collection(self, collection_name: str) -> str:
        """Get the model used for a specific collection"""
        return self.collection_model_mapping.get(collection_name)
```

### 3. Unified Embeddings Manager

```python
# src/utils/embeddings.py
class UnifiedEmbeddingsManager:
    """Backward-compatible wrapper that can use single or specialized models"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.use_specialized = config.get('use_specialized_embeddings', True)
        
        if self.use_specialized:
            self.manager = SpecializedEmbeddingManager(config)
        else:
            # Fallback to single model
            self.manager = EmbeddingsManager(config)
    
    def encode(self, texts: List[str], content_type: Optional[str] = None) -> np.ndarray:
        """Encode with optional content type"""
        if self.use_specialized and content_type:
            return self.manager.encode(texts, content_type)
        else:
            return self.manager.encode(texts)
```

## Implementation Phases

### Phase 1: Core Infrastructure ✅
1. Create `SpecializedEmbeddingManager` class
2. Implement LRU eviction and memory management
3. Build model registry system
4. Create unified embeddings wrapper

### Phase 2: Integration ✅
1. Update collection metadata storage
2. Modify indexing functions to use specialized models
3. Update search functions for model-aware queries
4. Add backward compatibility layer

### Phase 3: Testing & Polish ✅
1. Unit tests for all components
2. Integration testing with existing collections
3. Performance validation
4. Documentation updates

## Configuration

### Environment Variables
```bash
# Enable/disable specialized embeddings
QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true

# Model selection
QDRANT_CODE_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
QDRANT_CONFIG_EMBEDDING_MODEL=jinaai/jina-embeddings-v3
QDRANT_DOC_EMBEDDING_MODEL=hkunlp/instructor-large
QDRANT_GENERAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Memory management
QDRANT_MAX_MODELS_IN_MEMORY=3
QDRANT_MEMORY_LIMIT_GB=7.0

# Fallback models
QDRANT_CODE_EMBEDDING_FALLBACK=microsoft/codebert-base
QDRANT_CONFIG_EMBEDDING_FALLBACK=jinaai/jina-embeddings-v2-base-en
QDRANT_DOC_EMBEDDING_FALLBACK=sentence-transformers/all-mpnet-base-v2
```

### server_config.json
```json
{
  "specialized_embeddings": {
    "enabled": true,
    "models": {
      "code": {
        "name": "nomic-ai/CodeRankEmbed",
        "dimension": 768,
        "fallback": "microsoft/codebert-base"
      },
      "config": {
        "name": "jinaai/jina-embeddings-v3",
        "dimension": 1024,
        "fallback": "jinaai/jina-embeddings-v2-base-en"
      },
      "documentation": {
        "name": "hkunlp/instructor-large",
        "dimension": 768,
        "fallback": "sentence-transformers/all-mpnet-base-v2",
        "instruction_prefix": "Represent the technical documentation for retrieval:"
      }
    },
    "memory": {
      "max_models_in_memory": 3,
      "memory_limit_gb": 7.0
    }
  }
}
```

## Key Implementation Files

### Update qdrant_mcp_context_aware.py

```python
# In index_code function
def index_code(file_path: str, force_global: bool = False) -> Dict[str, Any]:
    # ... existing code ...
    
    # Use specialized embeddings
    embeddings = embedding_manager.encode(
        [chunk.page_content for chunk in chunks],
        content_type='code'
    )
    
    # Store metadata about the model used
    ensure_collection(
        collection_name,
        embedding_dimension=embeddings.shape[1],
        metadata={
            'embedding_model': embedding_manager.get_model_name('code'),
            'content_type': 'code'
        }
    )
```

### Update ensure_collection in qdrant_mcp_context_aware.py

```python
def ensure_collection(collection_name: str, 
                     embedding_dimension: int,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
    """Create collection with model metadata"""
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "size": embedding_dimension,
                "distance": "Cosine"
            },
            metadata=metadata or {}
        )
    except Exception:
        # Collection exists, update metadata
        if metadata:
            qdrant_client.update_collection(
                collection_name=collection_name,
                metadata=metadata
            )
```

## Technical Implementation Details

### 1. Lazy Model Loading

```python
def load_model(self, content_type: str) -> SentenceTransformer:
    """Load model with LRU eviction"""
    model_config = self.model_configs.get(content_type, self.model_configs['general'])
    model_name = model_config['name']
    
    # Check if already loaded
    if model_name in self.loaded_models:
        # Move to end (most recently used)
        self.loaded_models.move_to_end(model_name)
        return self.loaded_models[model_name]
    
    # Check memory before loading
    if len(self.loaded_models) >= self.max_models_in_memory:
        # Evict least recently used
        evicted_name = next(iter(self.loaded_models))
        self.loaded_models.pop(evicted_name)
        logger.info(f"Evicted model {evicted_name} from memory")
    
    # Load new model
    try:
        model = SentenceTransformer(model_name, device=self.device)
        self.loaded_models[model_name] = model
        return model
    except Exception as e:
        # Fall back to alternative model
        fallback_name = model_config.get('fallback')
        if fallback_name:
            logger.warning(f"Failed to load {model_name}, using fallback {fallback_name}")
            model = SentenceTransformer(fallback_name, device=self.device)
            self.loaded_models[fallback_name] = model
            return model
        raise
```

### 2. Instruction-Based Prompting for Documentation

```python
def encode_documentation(self, texts: List[str]) -> np.ndarray:
    """Encode documentation with instruction prompting"""
    model = self.load_model('documentation')
    
    # Add instruction prefix if using instructor model
    if 'instructor' in self.model_configs['documentation']['name']:
        instruction = self.model_configs['documentation']['instruction_prefix']
        texts = [f"{instruction} {text}" for text in texts]
    
    return model.encode(texts, batch_size=self.batch_size)
```

### 3. Model-Aware Search

```python
def get_query_embedding(self, query: str, collection_name: str) -> List[float]:
    """Get query embedding using the same model as the collection"""
    # Retrieve collection metadata
    collection_info = qdrant_client.get_collection(collection_name)
    model_name = collection_info.metadata.get('embedding_model', 'all-MiniLM-L6-v2')
    
    # Load appropriate model
    content_type = collection_info.metadata.get('content_type', 'general')
    model = self.embedding_manager.load_model(content_type)
    
    # Encode query
    embedding = model.encode([query])[0]
    return embedding.tolist()
```

## Migration Strategy

### 1. Backward Compatibility
- Keep `all-MiniLM-L6-v2` as default for existing collections
- Auto-detect model from collection metadata
- Support gradual migration per collection type
- Maintain old API signatures

### 2. User Migration Path
```bash
# Users will need to recreate collections (no automatic migration)
# Step 1: Enable specialized embeddings
export QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true

# Step 2: Download new models
./scripts/download_models.sh

# Step 3: Force reindex to use new models
"Force reindex the entire project"
```

## Testing Strategy

### Unit Tests ✅
- `test_specialized_embeddings.py`: Test core manager functionality
- `test_unified_embeddings.py`: Test backward compatibility
- `test_model_compatibility.py`: Test model matching logic

### Integration Tests
- Index files with each content type
- Verify correct model selection
- Test search with mismatched models
- Validate memory management

### Performance Tests
- Memory usage tracking
- Model loading times
- Encoding performance
- Search latency comparison

## Expected Impact

### Search Quality Improvements
- **Code Search**: 30-50% better relevance with CodeRankEmbed
- **Config Search**: 15-20% improvement with Jina v3
- **Documentation**: 30% better technical term understanding

### Resource Usage
- **Memory**: ~5.5GB with all models loaded (within 7GB limit)
- **Disk**: ~5.6GB for all model files
- **Load Time**: 3-5 seconds per model on M3 Pro

### User Experience
- More accurate code search results
- Better understanding of programming patterns
- Reduced cross-type noise in results
- Language-specific code understanding

## Implementation Summary

The specialized embeddings feature for v0.3.3 has been successfully implemented, providing:

1. **Content-aware embedding models** that understand the specific nature of code, configuration, and documentation
2. **Memory-efficient management** with LRU eviction keeping only 3 models in memory
3. **Seamless backward compatibility** through the UnifiedEmbeddingsManager
4. **Progressive context support** maintained throughout the implementation
5. **Comprehensive testing** with 38 unit tests validating all functionality

This feature represents a major leap in search quality, transforming the RAG server from a generic semantic search tool to a specialized code intelligence system that truly understands different types of technical content.

## References
- [Optimized Models Research](./optimized-models-for-collection-use.md)
- [Advanced RAG Roadmap](./advanced-rag-implementation-roadmap.md)
- [v0.3.3 Changelog](../../CHANGELOG.md#033---2025-06-04)