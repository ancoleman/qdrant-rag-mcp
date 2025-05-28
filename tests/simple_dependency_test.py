#!/usr/bin/env python3
"""
Simple test for dependency-aware search
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after setting path
from qdrant_mcp_context_aware import index_code, search
from utils.dependency_graph import get_dependency_graph

print("Testing dependency-aware search...")

# Index some files from the actual codebase
print("\n1. Indexing code_indexer.py (which imports ast_chunker and dependency_graph)...")
result = index_code("src/indexers/code_indexer.py")
print(f"   Result: {result.get('status', 'error')}")

print("\n2. Indexing ast_chunker.py...")
result = index_code("src/utils/ast_chunker.py")
print(f"   Result: {result.get('status', 'error')}")

print("\n3. Indexing dependency_graph.py...")
result = index_code("src/utils/dependency_graph.py")
print(f"   Result: {result.get('status', 'error')}")

# Build reverse dependencies
print("\n4. Building dependency graph...")
dep_graph = get_dependency_graph()
dep_graph.build_reverse_dependencies()
stats = dep_graph.get_statistics()
print(f"   Files tracked: {stats['total_files']}")
print(f"   Total imports: {stats['total_imports']}")

# Show specific dependencies
print("\n5. Checking dependencies for code_indexer.py...")
abs_path = str(Path("src/indexers/code_indexer.py").resolve())
deps = dep_graph.get_file_dependencies(abs_path)
if deps:
    print(f"   Imports: {[imp.module for imp in deps.imports]}")
    print(f"   Imported by: {list(deps.imported_by)}")
else:
    print(f"   No dependencies found for: {abs_path}")

# Wait for indexing
time.sleep(1)

# Test search without dependencies
print("\n6. Search WITHOUT dependencies for 'ast_chunker':")
results = search("ast_chunker", n_results=5, include_dependencies=False)
for i, res in enumerate(results.get('results', [])):
    print(f"   {i+1}. {res.get('file_path', 'unknown')} (score: {res.get('score', 0):.3f})")

# Test search with dependencies
print("\n7. Search WITH dependencies for 'ast_chunker':")
results = search("ast_chunker", n_results=10, include_dependencies=True)
for i, res in enumerate(results.get('results', [])):
    is_dep = " [DEPENDENCY]" if res.get('is_dependency') else ""
    print(f"   {i+1}. {res.get('file_path', 'unknown')} (score: {res.get('score', 0):.3f}){is_dep}")

print("\nTest complete!")