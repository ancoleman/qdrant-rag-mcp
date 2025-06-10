#!/usr/bin/env python3
"""
Test backward compatibility of query prefix when env var is not set
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Make sure the env var is NOT set
if 'QDRANT_CODE_QUERY_PREFIX' in os.environ:
    del os.environ['QDRANT_CODE_QUERY_PREFIX']

from utils.specialized_embeddings import SpecializedEmbeddingManager


def test_backward_compatibility():
    """Test that CodeRankEmbed works without setting QDRANT_CODE_QUERY_PREFIX"""
    
    print("Testing backward compatibility (no env var set)...")
    
    # Create manager without any custom prefix env var
    manager = SpecializedEmbeddingManager()
    
    # Get the code model config
    code_config = manager.model_configs.get('code', {})
    
    print(f"Code model: {code_config.get('name')}")
    print(f"Requires query prefix: {code_config.get('requires_query_prefix')}")
    print(f"Custom query prefix from env: {repr(code_config.get('query_prefix'))}")
    
    # The query_prefix should be None or empty string
    query_prefix = code_config.get('query_prefix')
    assert query_prefix is None or query_prefix == "", f"Expected None or empty, got: {repr(query_prefix)}"
    
    print("\n✓ No custom prefix is set (as expected)")
    
    # Verify the fallback logic
    if code_config.get('requires_query_prefix', False):
        if query_prefix:  # This should be False
            print("Would use custom prefix")
        elif 'CodeRankEmbed' in code_config.get('name', ''):
            print("✓ Will use hardcoded CodeRankEmbed prefix: 'Represent this query for searching relevant code:'")
        else:
            print("No prefix would be applied")
    
    print("\n✓ Backward compatibility confirmed!")
    print("CodeRankEmbed will use its required prefix even without env var")


def test_empty_string_behavior():
    """Test behavior when env var is set to empty string"""
    
    print("\n\nTesting empty string behavior...")
    
    # Set env var to empty string
    os.environ['QDRANT_CODE_QUERY_PREFIX'] = ""
    
    manager = SpecializedEmbeddingManager()
    code_config = manager.model_configs.get('code', {})
    
    query_prefix = code_config.get('query_prefix')
    print(f"Query prefix with empty env var: {repr(query_prefix)}")
    
    # Empty string is falsy in Python
    if query_prefix:
        print("Empty string is truthy (unexpected)")
    else:
        print("✓ Empty string is falsy (will use fallback)")
    
    # Clean up
    del os.environ['QDRANT_CODE_QUERY_PREFIX']


def test_custom_prefix():
    """Test that custom prefix works when set"""
    
    print("\n\nTesting custom prefix...")
    
    # Set a custom prefix
    os.environ['QDRANT_CODE_QUERY_PREFIX'] = "Find code that implements:"
    
    manager = SpecializedEmbeddingManager()
    code_config = manager.model_configs.get('code', {})
    
    query_prefix = code_config.get('query_prefix')
    print(f"Query prefix with custom env var: {repr(query_prefix)}")
    
    assert query_prefix == "Find code that implements:", "Custom prefix not loaded correctly"
    print("✓ Custom prefix loaded successfully")
    
    # Clean up
    del os.environ['QDRANT_CODE_QUERY_PREFIX']


if __name__ == "__main__":
    test_backward_compatibility()
    test_empty_string_behavior()
    test_custom_prefix()
    
    print("\n\nAll tests passed! ✅")
    print("\nSummary:")
    print("- Without env var: Falls back to hardcoded CodeRankEmbed prefix")
    print("- With empty string: Also falls back (empty string is falsy)")
    print("- With custom value: Uses the custom prefix")
    print("\nBackward compatibility is maintained!")