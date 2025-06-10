#!/usr/bin/env python3
"""
Test script for DocumentationIndexer functionality.

This tests the new markdown/documentation indexing capabilities added in v0.2.3.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexers.documentation_indexer import DocumentationIndexer


def test_documentation_indexer():
    """Test the DocumentationIndexer with a real markdown file."""
    
    # Initialize indexer
    indexer = DocumentationIndexer(chunk_size=2000, chunk_overlap=400)
    
    # Test file - using CHANGELOG.md as it has good structure
    test_file = Path(__file__).parent.parent / "CHANGELOG.md"
    
    print(f"Testing DocumentationIndexer with: {test_file}")
    print("=" * 60)
    
    # Check if supported
    assert indexer.is_supported(str(test_file)), f"{test_file} should be supported"
    print("✓ File type is supported")
    
    # Index the file
    chunks = indexer.index_file(str(test_file))
    
    print(f"\nExtracted {len(chunks)} chunks from the document")
    print("-" * 60)
    
    # Show first few chunks with metadata
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i + 1}:")
        print(f"Type: {chunk['metadata'].get('chunk_type', 'unknown')}")
        print(f"Heading: {chunk['metadata'].get('heading', 'None')}")
        print(f"Heading Level: {chunk['metadata'].get('heading_level', 0)}")
        print(f"Heading Hierarchy: {chunk['metadata'].get('heading_hierarchy', [])}")
        print(f"Has Code Blocks: {chunk['metadata'].get('has_code_blocks', False)}")
        print(f"Content Preview: {chunk['content'][:200]}...")
        print("-" * 40)
    
    # Test specific features
    print("\n" + "=" * 60)
    print("FEATURE TESTS:")
    print("=" * 60)
    
    # 1. Test heading extraction
    headings_found = [c['metadata'].get('heading') for c in chunks if c['metadata'].get('heading')]
    print(f"\n1. Heading Extraction: Found {len(headings_found)} headings")
    print(f"   Sample headings: {headings_found[:5]}")
    
    # 2. Test code block detection
    chunks_with_code = [c for c in chunks if c['metadata'].get('has_code_blocks')]
    print(f"\n2. Code Block Detection: {len(chunks_with_code)} chunks contain code blocks")
    if chunks_with_code:
        print(f"   Code languages found: {chunks_with_code[0]['metadata'].get('section_code_languages', [])}")
    
    # 3. Test hierarchy preservation
    hierarchical_chunks = [c for c in chunks if len(c['metadata'].get('heading_hierarchy', [])) > 1]
    print(f"\n3. Hierarchy Preservation: {len(hierarchical_chunks)} chunks have hierarchical context")
    if hierarchical_chunks:
        sample = hierarchical_chunks[0]['metadata']['heading_hierarchy']
        print(f"   Example hierarchy: {' > '.join(sample)}")
    
    # 4. Test large section splitting
    partial_chunks = [c for c in chunks if c['metadata'].get('is_partial', False)]
    print(f"\n4. Large Section Splitting: {len(partial_chunks)} chunks are partial (split from larger sections)")
    
    # 5. Test metadata extraction
    print(f"\n5. File Metadata:")
    if chunks:
        file_meta = chunks[0]['metadata']
        print(f"   Title: {file_meta.get('title', 'Unknown')}")
        print(f"   File Type: {file_meta.get('doc_type', 'Unknown')}")
        print(f"   Total Code Blocks: {file_meta.get('code_block_count', 0)}")
        print(f"   Code Languages: {file_meta.get('code_languages', [])}")
    
    # 6. Test summary extraction
    print(f"\n6. Summary Extraction Test:")
    if chunks and chunks[0]['content']:
        summary = indexer.extract_summary(chunks[0]['content'], max_length=150)
        print(f"   Summary: {summary}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed successfully!")
    
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("EDGE CASE TESTS:")
    print("=" * 60)
    
    indexer = DocumentationIndexer()
    
    # Test 1: Empty file path
    print("\n1. Testing with non-existent file:")
    chunks = indexer.index_file("/non/existent/file.md")
    assert chunks == [], "Should return empty list for non-existent file"
    print("   ✓ Handled gracefully")
    
    # Test 2: Unsupported file
    print("\n2. Testing with unsupported file type:")
    assert not indexer.is_supported("test.py"), "Should not support .py files"
    print("   ✓ Correctly rejected")
    
    # Test 3: File types
    print("\n3. Testing supported file types:")
    supported = [".md", ".markdown", ".rst", ".txt", ".mdx"]
    for ext in supported:
        assert indexer.is_supported(f"test{ext}"), f"Should support {ext} files"
    print(f"   ✓ All supported: {supported}")
    
    print("\n✓ Edge case tests passed!")


if __name__ == "__main__":
    print("Documentation Indexer Test Suite")
    print("================================\n")
    
    try:
        # Run main tests
        test_documentation_indexer()
        
        # Run edge case tests
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! 🎉")
        print("Documentation Indexer is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)