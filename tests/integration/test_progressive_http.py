#!/usr/bin/env python3
"""Test Progressive Context through HTTP API."""

import requests
import json
import sys
from typing import Dict, Any

# Base URL for the HTTP server
BASE_URL = "http://localhost:8081"


def test_search_with_progressive_context(
    query: str,
    context_level: str = "auto",
    progressive_mode: bool = True,
    n_results: int = 5
) -> Dict[str, Any]:
    """Test search with progressive context parameters."""
    
    url = f"{BASE_URL}/search"
    payload = {
        "query": query,
        "n_results": n_results,
        "search_mode": "hybrid",
        "context_level": context_level,
        "progressive_mode": progressive_mode,
        "include_expansion_options": True,
        "semantic_cache": True
    }
    
    print(f"\n{'='*60}")
    print(f"Testing Progressive Search")
    print(f"Query: {query}")
    print(f"Context Level: {context_level}")
    print(f"Progressive Mode: {progressive_mode}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Print results
        print(f"Total Results: {result.get('total', 0)}")
        print(f"Search Mode: {result.get('search_mode', 'unknown')}")
        
        # Check if progressive metadata is included
        if "progressive" in result:
            prog = result["progressive"]
            print(f"\nProgressive Context Metadata:")
            print(f"- Level Used: {prog.get('level_used', 'unknown')}")
            print(f"- Token Estimate: {prog.get('token_estimate', 0)}")
            print(f"- Token Reduction: {prog.get('token_reduction', '0%')}")
            print(f"- Cache Hit: {prog.get('cache_hit', False)}")
            
            if prog.get('query_intent'):
                intent = prog['query_intent']
                print(f"- Query Intent: {intent.get('type', 'unknown')} (confidence: {intent.get('confidence', 0):.2f})")
            
            # Print expansion options if available
            if prog.get('expansion_options'):
                print(f"\nExpansion Options ({len(prog['expansion_options'])} available):")
                for i, opt in enumerate(prog['expansion_options'][:3]):  # Show first 3
                    print(f"  {i+1}. {opt['type']} -> {opt['path']}")
                    print(f"     Tokens: {opt['estimated_tokens']}, Relevance: {opt['relevance']:.2f}")
        
        # Print some results
        print(f"\nSearch Results:")
        for i, res in enumerate(result.get("results", [])[:3]):
            print(f"\n{i+1}. File: {res.get('file_path', 'unknown')}")
            print(f"   Score: {res.get('score', 0):.3f}")
            print(f"   Type: {res.get('chunk_type', 'unknown')}")
            
            # Show truncated content for progressive results
            content = res.get('content', '')
            if res.get('_truncated') or res.get('_summarize'):
                print(f"   Content: {content[:100]}...")
            else:
                print(f"   Content: {content[:200]}...")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return {"error": str(e)}


def test_cache_behavior():
    """Test semantic cache behavior."""
    print("\n" + "="*60)
    print("Testing Semantic Cache Behavior")
    print("="*60)
    
    # First query
    query1 = "What does the authentication system do?"
    result1 = test_search_with_progressive_context(query1, context_level="file")
    
    # Similar query (should hit cache)
    query2 = "Explain the authentication system"
    result2 = test_search_with_progressive_context(query2, context_level="file")
    
    # Check cache hit
    if "progressive" in result2:
        if result2["progressive"].get("cache_hit"):
            print("\n✓ Cache hit detected for similar query!")
        else:
            print("\n✗ No cache hit for similar query")


def test_different_context_levels():
    """Test different context levels."""
    query = "search functionality"
    
    for level in ["file", "class", "method"]:
        result = test_search_with_progressive_context(query, context_level=level)
        
        if "progressive" in result:
            prog = result["progressive"]
            print(f"\nLevel '{level}' Summary:")
            print(f"- Token estimate: {prog.get('token_estimate', 0)}")
            print(f"- Token reduction: {prog.get('token_reduction', '0%')}")
            print(f"- Results shape: {len(result.get('results', []))} items")


def test_auto_classification():
    """Test automatic query intent classification."""
    test_queries = [
        ("What does the configuration system do?", "file"),
        ("Show me the bug in line 123", "method"),
        ("Find the DatabaseManager class", "class"),
        ("How does the caching work?", "file"),
        ("Fix the error in the save function", "method")
    ]
    
    print("\n" + "="*60)
    print("Testing Query Intent Auto-Classification")
    print("="*60)
    
    for query, expected_level in test_queries:
        result = test_search_with_progressive_context(query, context_level="auto", n_results=2)
        
        if "progressive" in result:
            actual_level = result["progressive"].get("level_used", "unknown")
            match = "✓" if actual_level == expected_level else "✗"
            print(f"\n{match} Query: '{query}'")
            print(f"  Expected: {expected_level}, Got: {actual_level}")


def main():
    """Run all tests."""
    print("Progressive Context HTTP API Tests")
    print("Make sure the HTTP server is running on port 8081")
    print("And progressive_context is enabled in server_config.json")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        response.raise_for_status()
        print("\n✓ Server is running")
    except Exception as e:
        print(f"\n✗ Server not accessible: {e}")
        print("Please start the server with: python src/http_server.py")
        sys.exit(1)
    
    # Run tests
    if len(sys.argv) > 1:
        if sys.argv[1] == "cache":
            test_cache_behavior()
        elif sys.argv[1] == "levels":
            test_different_context_levels()
        elif sys.argv[1] == "auto":
            test_auto_classification()
        else:
            # Custom query test
            query = " ".join(sys.argv[1:])
            test_search_with_progressive_context(query)
    else:
        # Run a basic test
        test_search_with_progressive_context("What does the authentication system do?", context_level="file")
        
        print("\n\nOther test options:")
        print("  python test_progressive_http.py cache    # Test cache behavior")
        print("  python test_progressive_http.py levels   # Test different context levels")
        print("  python test_progressive_http.py auto     # Test auto-classification")
        print("  python test_progressive_http.py <query>  # Test custom query")


if __name__ == "__main__":
    main()