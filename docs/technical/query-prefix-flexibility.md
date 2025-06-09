# Query Prefix Flexibility in Specialized Embeddings

## Overview

The specialized embeddings system now supports flexible query prefix configuration while maintaining backward compatibility with model-specific requirements like CodeRankEmbed.

## How It Works

The system uses a three-tier approach for query prefixes:

1. **Configuration Flag**: Check if model requires a query prefix (`requires_query_prefix`)
2. **Custom Prefix**: Use custom prefix if configured (via env var or config)
3. **Model-Specific Fallback**: Fall back to known model-specific prefixes

## Configuration

### Environment Variables

```bash
# Set a custom query prefix for code embeddings
export QDRANT_CODE_QUERY_PREFIX="Find code that implements:"

# Or leave empty to use model defaults
export QDRANT_CODE_QUERY_PREFIX=""
```

### Server Config (server_config.json)

```json
"specialized_embeddings": {
  "models": {
    "code": {
      "primary": "nomic-ai/CodeRankEmbed",
      "requires_query_prefix": true,
      "query_prefix": "${QDRANT_CODE_QUERY_PREFIX:-}"
    }
  }
}
```

### Programmatic Configuration

```python
from utils.specialized_embeddings import get_specialized_embedding_manager

manager = get_specialized_embedding_manager()

# Set a custom prefix
manager.set_query_prefix('code', 'Custom prefix:', requires_prefix=True)

# Disable query prefix
manager.set_query_prefix('code', None, requires_prefix=False)
```

## Model-Specific Requirements

### CodeRankEmbed
- **Required Prefix**: "Represent this query for searching relevant code:"
- **Documentation**: The model was trained with this specific prefix for queries
- **Fallback**: This prefix is hardcoded as a fallback to ensure compatibility

### Other Models
- Most models don't require query prefixes
- Documentation models may use instruction prefixes (different mechanism)
- Config and general models typically don't need prefixes

## Implementation Details

The query prefix logic is implemented in `specialized_embeddings.py`:

```python
# Apply query prefix for models that require it
if len(texts) == 1 and model_config.get('requires_query_prefix', False):
    # This is likely a query, not a document
    # First check if a custom prefix is configured
    if model_config.get('query_prefix'):
        texts = [f"{model_config['query_prefix']} {texts[0]}"]
    # Fallback to known model-specific prefixes
    elif 'CodeRankEmbed' in model_config['name']:
        # CodeRankEmbed requires this specific prefix as per their documentation
        texts = [f"Represent this query for searching relevant code: {texts[0]}"]
```

## Benefits

1. **Flexibility**: Users can customize query prefixes for experimentation
2. **Backward Compatibility**: Existing behavior is preserved
3. **Model Safety**: Required prefixes are maintained as fallbacks
4. **Future-Proof**: Easy to add support for new models with different requirements

## Testing

Run the test suite to verify the implementation:

```bash
python tests/test_query_prefix_flexibility.py
```

This tests:
- Default CodeRankEmbed prefix behavior
- Custom prefix override via environment
- Programmatic prefix configuration
- Non-code models remain unaffected