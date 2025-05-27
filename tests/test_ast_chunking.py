#!/usr/bin/env python3
"""
Test AST chunking functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.ast_chunker import create_ast_chunker
from indexers.code_indexer import CodeIndexer
import json

def test_ast_chunker():
    """Test the AST chunker directly"""
    print("Testing AST Chunker...")
    
    # Create chunker
    chunker = create_ast_chunker('python', max_chunk_size=1000)
    
    # Test file
    test_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'hybrid_search.py')
    
    # Get chunks
    chunks = chunker.chunk_file(test_file)
    
    print(f"\nFound {len(chunks)} chunks in {test_file}")
    print("\nChunk types and names:")
    for chunk in chunks:
        print(f"  - {chunk.chunk_type}: {chunk.name} (lines {chunk.line_start}-{chunk.line_end})")
        print(f"    Hierarchy: {' > '.join(chunk.hierarchy)}")
        if chunk.metadata:
            print(f"    Metadata: {json.dumps(chunk.metadata, indent=6)}")
    
    print("\nFirst chunk content preview:")
    if chunks:
        print(chunks[0].content[:200] + "..." if len(chunks[0].content) > 200 else chunks[0].content)

def test_code_indexer_with_ast():
    """Test the code indexer with AST chunking enabled"""
    print("\n\nTesting Code Indexer with AST...")
    
    # Create indexer with AST enabled
    indexer = CodeIndexer(chunk_size=1500, use_ast_chunking=True)
    
    # Test file
    test_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'utils', 'embeddings.py')
    
    # Index file
    chunks = indexer.index_file(test_file)
    
    print(f"\nIndexed {len(chunks)} chunks from {test_file}")
    print("\nChunk details:")
    for chunk in chunks[:5]:  # Show first 5
        meta = chunk.metadata
        print(f"  - Chunk {chunk.chunk_index}: {meta.get('chunk_type', 'unknown')} - {meta.get('name', 'unnamed')}")
        if 'hierarchy' in meta:
            print(f"    Hierarchy: {' > '.join(meta['hierarchy'])}")
        print(f"    Lines: {chunk.line_start}-{chunk.line_end}")

def compare_chunking_methods():
    """Compare AST vs traditional chunking"""
    print("\n\nComparing Chunking Methods...")
    
    test_file = os.path.join(os.path.dirname(__file__), '..', 'src', 'qdrant_mcp_context_aware.py')
    
    # Traditional chunking
    traditional_indexer = CodeIndexer(chunk_size=1500, use_ast_chunking=False)
    traditional_chunks = traditional_indexer.index_file(test_file)
    
    # AST chunking
    ast_indexer = CodeIndexer(chunk_size=1500, use_ast_chunking=True)
    ast_chunks = ast_indexer.index_file(test_file)
    
    print(f"\nFile: {test_file}")
    print(f"Traditional chunks: {len(traditional_chunks)}")
    print(f"AST chunks: {len(ast_chunks)}")
    print(f"Reduction: {(1 - len(ast_chunks)/len(traditional_chunks))*100:.1f}%")
    
    # Calculate total tokens (rough estimate)
    traditional_tokens = sum(len(chunk.content.split()) for chunk in traditional_chunks)
    ast_tokens = sum(len(chunk.content.split()) for chunk in ast_chunks)
    
    print(f"\nEstimated tokens:")
    print(f"Traditional: ~{traditional_tokens} tokens")
    print(f"AST: ~{ast_tokens} tokens")
    print(f"Token reduction: {(1 - ast_tokens/traditional_tokens)*100:.1f}%")

if __name__ == "__main__":
    test_ast_chunker()
    test_code_indexer_with_ast()
    compare_chunking_methods()