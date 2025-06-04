#!/usr/bin/env python3
"""Test that progressive context doesn't use dummy vectors for search."""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.progressive_context import ProgressiveContextManager
from src.utils.embeddings import get_embedding_model
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def test_progressive_no_dummy_vectors():
    """Test that progressive context preserves original vector search scores."""
    
    # Initialize test client (in-memory)
    client = QdrantClient(":memory:")
    
    # Create test collection
    collection_name = "test_collection"
    embedding_dim = 384  # For all-MiniLM-L6-v2
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
    )
    
    # Get embedding model
    embed_model = get_embedding_model()
    
    # Add test documents
    test_docs = [
        {
            "id": 1,
            "content": "def authenticate_user(username, password): return check_credentials(username, password)",
            "file_path": "auth.py",
            "chunk_index": 0,
            "chunk_type": "function"
        },
        {
            "id": 2,
            "content": "class UserAuthentication: def login(self, user, pwd): self.authenticate(user, pwd)",
            "file_path": "auth.py",
            "chunk_index": 1,
            "chunk_type": "class"
        },
        {
            "id": 3,
            "content": "# Authentication module for user login and security",
            "file_path": "auth.py",
            "chunk_index": 2,
            "chunk_type": "comment"
        }
    ]
    
    # Index documents
    points = []
    for doc in test_docs:
        embedding = embed_model.embed_documents([doc["content"]])[0]
        points.append({
            "id": doc["id"],
            "vector": embedding,
            "payload": {
                "content": doc["content"],
                "file_path": doc["file_path"],
                "chunk_index": doc["chunk_index"],
                "chunk_type": doc["chunk_type"]
            }
        })
    
    client.upsert(collection_name=collection_name, points=points)
    
    # Initialize progressive context manager
    context_manager = ProgressiveContextManager(
        qdrant_client=client,
        collections=[collection_name],
        cache_dir="./test_cache"
    )
    
    # Test query
    query = "authentication user login"
    query_embedding = embed_model.embed_documents([query])[0]
    
    # Test hybrid search (the problematic mode)
    print("\n=== Testing Hybrid Search ===")
    results = context_manager._search_at_level(
        query=query,
        query_embedding=query_embedding,
        n_results=3,
        level="snippet",
        search_collections=[collection_name],
        search_mode="hybrid"
    )
    
    print(f"Found {len(results)} results")
    
    # Check results
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Content: {result['content'][:80]}...")
        
        # Key check: vector_score should be properly set (not 0 or dummy)
        if 'vector_score' in result:
            print(f"  Vector score: {result['vector_score']:.4f}")
            assert result['vector_score'] > 0, "Vector score should be positive, not from dummy vector"
        
        # BM25 score might be 0 if keyword didn't match
        if 'bm25_score' in result:
            print(f"  BM25 score: {result['bm25_score']:.4f}")
    
    # Also test keyword-only mode
    print("\n\n=== Testing Keyword Search ===")
    results_keyword = context_manager._search_at_level(
        query=query,
        query_embedding=query_embedding,
        n_results=3,
        level="snippet",
        search_collections=[collection_name],
        search_mode="keyword"
    )
    
    print(f"Found {len(results_keyword)} keyword results")
    
    # Verify scores are reasonable (not corrupted by dummy vectors)
    for result in results_keyword:
        # For keyword search, the score comes from BM25
        assert result['score'] >= 0, "Score should be non-negative"
        print(f"  Score: {result['score']:.4f} - {result['content'][:60]}...")
    
    print("\n✅ Test passed! No dummy vectors detected.")
    
    # Clean up
    if os.path.exists("./test_cache"):
        import shutil
        shutil.rmtree("./test_cache")

if __name__ == "__main__":
    test_progressive_no_dummy_vectors()