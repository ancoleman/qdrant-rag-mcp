#!/usr/bin/env python3
"""Test BM25 indexing directly"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from utils.hybrid_search import get_hybrid_searcher
from qdrant_client import QdrantClient

# Get the hybrid searcher instance
hybrid_searcher = get_hybrid_searcher()

print("Before indexing - BM25 indices:", list(hybrid_searcher.bm25_manager.indices.keys()))

# Get some documents from the documentation collection
client = QdrantClient(host="localhost", port=6333)
collection_name = "project_qdrant_rag_documentation"

# Fetch some documents
points, _ = client.scroll(
    collection_name=collection_name,
    limit=10,
    with_payload=True,
    with_vectors=False
)

if points:
    print(f"\nFound {len(points)} documents in {collection_name}")
    
    # Prepare documents for BM25
    documents = []
    for point in points:
        doc = {
            "id": point.id,
            "content": point.payload.get("content", ""),
            "file_path": point.payload.get("file_path", ""),
            "chunk_index": point.payload.get("chunk_index", 0),
            "doc_type": point.payload.get("doc_type", ""),
            "heading": point.payload.get("heading", ""),
            "chunk_type": point.payload.get("chunk_type", "")
        }
        documents.append(doc)
    
    print(f"Indexing {len(documents)} documents into BM25...")
    hybrid_searcher.bm25_manager.update_index(collection_name, documents)
    
    print("\nAfter indexing - BM25 indices:", list(hybrid_searcher.bm25_manager.indices.keys()))
    
    # Test search
    results = hybrid_searcher.bm25_manager.search(collection_name, "installation setup guide", k=5)
    print(f"\nSearch results for 'installation setup guide': {len(results)} found")
    for doc_id, score in results[:3]:
        print(f"  - {doc_id}: score={score:.4f}")
else:
    print("No documents found in collection")