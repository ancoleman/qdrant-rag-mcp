#!/usr/bin/env python3
"""
Test specialized embeddings with current project data
Uses the existing project collections to verify model usage
"""

import os
import sys
import json
import requests
from pathlib import Path

# Base URL for HTTP API
BASE_URL = "http://localhost:8081"

def test_current_project_search():
    """Test search in current project with specialized models"""
    print("=" * 60)
    print("Testing Specialized Embeddings with Current Project")
    print("=" * 60)
    
    # 1. Check health and current context
    print("\n1. Checking server health and context...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✓ Server healthy")
        print(f"  Current project: {health.get('current_project', 'unknown')}")
        if 'embedding_manager' in health:
            print(f"  Embedding manager: {health['embedding_manager'].get('type', 'unknown')}")
    else:
        print("✗ Server not healthy")
        return
    
    # 2. Test code search with specialized model
    print("\n2. Testing code search (should use CodeRankEmbed)...")
    response = requests.post(
        f"{BASE_URL}/search_code",
        json={
            "query": "SpecializedEmbeddingManager load_model LRU eviction",
            "n_results": 3
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Code search successful - {len(results['results'])} results")
        for i, result in enumerate(results['results']):
            print(f"\n  Result {i+1}:")
            print(f"    File: {Path(result['file_path']).name}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Chunk type: {result.get('chunk_type', 'unknown')}")
            # Check if metadata indicates model used
            if 'metadata' in result:
                print(f"    Metadata: {json.dumps(result['metadata'], indent=6)}")
    
    # 3. Test config search
    print("\n3. Testing config search (should use jina-embeddings-v3)...")
    response = requests.post(
        f"{BASE_URL}/search",
        json={
            "query": "specialized_embeddings models configuration",
            "n_results": 3
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Config search successful - {len(results['results'])} results")
        for i, result in enumerate(results['results'][:2]):
            print(f"\n  Result {i+1}:")
            print(f"    File: {Path(result['file_path']).name}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Collection: {result.get('collection', 'unknown')}")
    
    # 4. Test documentation search
    print("\n4. Testing documentation search (should use instructor-large)...")
    response = requests.post(
        f"{BASE_URL}/search_docs",
        json={
            "query": "specialized embeddings implementation plan complete",
            "n_results": 3
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Documentation search successful - {len(results['results'])} results")
        for i, result in enumerate(results['results'][:2]):
            print(f"\n  Result {i+1}:")
            print(f"    File: {Path(result['file_path']).name}")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Heading: {result.get('heading', 'unknown')}")
    
    # 5. Test progressive mode with specialized embeddings
    print("\n5. Testing progressive mode with specialized embeddings...")
    response = requests.post(
        f"{BASE_URL}/search_code",
        json={
            "query": "UnifiedEmbeddingsManager encode method",
            "n_results": 3,
            "progressive_mode": True,
            "context_level": "method"
        }
    )
    
    if response.status_code == 200:
        results = response.json()
        print(f"✓ Progressive search successful")
        if 'progressive' in results:
            prog = results['progressive']
            print(f"  Level used: {prog.get('level_used', 'unknown')}")
            print(f"  Token estimate: {prog.get('token_estimate', 0)}")
            print(f"  Token reduction: {prog.get('token_reduction', '0%')}")
    
    print("\n" + "=" * 60)
    print("✅ Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_current_project_search()