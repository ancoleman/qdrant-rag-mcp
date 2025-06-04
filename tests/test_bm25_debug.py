#!/usr/bin/env python3
"""Debug BM25 indexing issue"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from utils.hybrid_search import get_hybrid_searcher
from qdrant_client import QdrantClient
import time

# Get the hybrid searcher instance
hybrid_searcher = get_hybrid_searcher()

print("Current BM25 indices:")
for collection_name, index in hybrid_searcher.bm25_manager.indices.items():
    doc_count = len(hybrid_searcher.bm25_manager.documents.get(collection_name, []))
    print(f"  {collection_name}: {doc_count} documents")

print("\nChecking Qdrant collections:")
client = QdrantClient(host="localhost", port=6333)
collections = client.get_collections().collections

for collection in collections:
    if "documentation" in collection.name:
        print(f"\n  {collection.name}:")
        # Get a sample point to check
        points = client.scroll(
            collection_name=collection.name,
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        if points[0]:
            print(f"    Points exist: Yes")
            print(f"    Sample content: {points[0][0].payload.get('content', '')[:100]}...")
        else:
            print(f"    Points exist: No")

print("\nTesting BM25 search on documentation collection:")
doc_collections = [c.name for c in collections if "documentation" in c.name]
if doc_collections:
    collection = doc_collections[0]
    results = hybrid_searcher.bm25_manager.search(collection, "installation setup guide", k=5)
    print(f"  Results for '{collection}': {len(results)} found")
    for doc_id, score in results[:2]:
        print(f"    - {doc_id}: score={score}")