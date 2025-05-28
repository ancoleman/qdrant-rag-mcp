#!/usr/bin/env python3
"""
Check if dependency metadata is being stored in Qdrant
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_client import QdrantClient

# Create client
client = QdrantClient(host="localhost", port=6333)

print("Checking dependency metadata in Qdrant...\n")

# Check project_qdrant_rag_code collection
collection_name = "project_qdrant_rag_code"

# Scroll through first chunks only
scroll_filter = {
    "must": [
        {"key": "chunk_index", "match": {"value": 0}}
    ]
}

try:
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter=scroll_filter,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    
    points, _ = results
    
    print(f"Found {len(points)} first chunks. Showing ones with dependencies:\n")
    
    deps_count = 0
    for point in points:
        payload = point.payload
        file_path = payload.get("file_path", "")
        deps = payload.get("dependencies", {})
        
        if deps or "code_indexer" in file_path or "ast_chunker" in file_path or "dependency_graph" in file_path:
            print(f"File: {file_path}")
            if deps:
                deps_count += 1
                print(f"  Has dependencies metadata: YES")
                print(f"  Imports: {deps.get('imports', [])}")
                print(f"  Exports: {deps.get('exports', [])}")
            else:
                print(f"  Has dependencies metadata: NO")
            print()
    
    print(f"\nTotal files with dependencies metadata: {deps_count}/{len(points)}")
        
except Exception as e:
    print(f"Error: {e}")