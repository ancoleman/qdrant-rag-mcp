#!/usr/bin/env python3
"""
Simple test to verify Documentation Indexer is working and integrated.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from indexers.documentation_indexer import DocumentationIndexer
        print("✓ DocumentationIndexer imported successfully")
        
        from indexers import DocumentationIndexer as DI2
        print("✓ DocumentationIndexer available from indexers package")
        
        # Check it's in __all__
        from indexers import __all__
        assert "DocumentationIndexer" in __all__
        print("✓ DocumentationIndexer is in __all__")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic indexer functionality."""
    from indexers.documentation_indexer import DocumentationIndexer
    
    print("\nTesting basic functionality...")
    
    # Create indexer
    indexer = DocumentationIndexer()
    print("✓ Indexer created")
    
    # Test file support
    assert indexer.is_supported("test.md")
    assert indexer.is_supported("README.markdown")
    assert not indexer.is_supported("test.py")
    print("✓ File type detection works")
    
    # Test with a small markdown content
    test_content = """# Test Document

This is a test paragraph.

## Section 1

Some content here.

### Subsection 1.1

More detailed content.

```python
def hello():
    print("Hello, world!")
```

## Section 2

Final section.
"""
    
    # Create a temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        # Index the file
        chunks = indexer.index_file(temp_path)
        print(f"✓ Indexed test file: {len(chunks)} chunks created")
        
        # Verify chunks
        assert len(chunks) > 0
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        print("✓ Chunks have correct structure")
        
        # Check metadata
        headings = [c['metadata'].get('heading') for c in chunks if c['metadata'].get('heading')]
        assert 'Test Document' in headings
        assert 'Section 1' in headings
        print(f"✓ Extracted headings: {headings}")
        
        # Check code block detection
        code_chunks = [c for c in chunks if c['metadata'].get('has_code_blocks')]
        assert len(code_chunks) > 0
        print(f"✓ Code blocks detected in {len(code_chunks)} chunks")
        
    finally:
        # Clean up
        import os
        os.unlink(temp_path)
    
    return True


def verify_mcp_integration():
    """Verify the documentation indexer is integrated in the MCP server."""
    print("\nVerifying MCP server integration...")
    
    try:
        # Check if the functions exist in the main server file
        with open("src/qdrant_mcp_context_aware.py", "r") as f:
            content = f.read()
            
        checks = [
            ("index_documentation", "def index_documentation"),
            ("search_docs", "def search_docs"),
            ("DocumentationIndexer import", "DocumentationIndexer"),
            ("documentation collection support", "_documentation"),
            ("*.md in patterns", '"*.md"')
        ]
        
        all_found = True
        for name, pattern in checks:
            if pattern in content:
                print(f"✓ Found {name}")
            else:
                print(f"❌ Missing {name}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        return False


if __name__ == "__main__":
    print("Documentation Indexer Verification")
    print("=" * 50)
    
    success = True
    
    # Run tests
    success &= test_imports()
    success &= test_basic_functionality()
    success &= verify_mcp_integration()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All checks passed! Documentation Indexer is ready.")
        print("\nVersion 0.2.3 implementation is complete with:")
        print("- DocumentationIndexer class ✓")
        print("- Markdown parsing and chunking ✓")
        print("- MCP server integration ✓")
        print("- Enhanced index_directory support ✓")
    else:
        print("❌ Some checks failed. Please review the output above.")
        sys.exit(1)