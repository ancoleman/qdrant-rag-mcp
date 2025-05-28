#!/usr/bin/env python3
"""
Test script for dependency-aware search functionality

This script tests the new include_dependencies parameter in search functions.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qdrant_mcp_context_aware import search, search_code, index_code, index_directory
from utils.dependency_graph import get_dependency_graph, reset_dependency_graph


def setup_test_files():
    """Create test files with clear dependencies"""
    test_dir = Path("tests/dependency_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create main.py that imports utils
    main_content = '''"""Main application file"""
import os
import sys
from utils.helpers import format_data
from utils.validators import validate_input

def main():
    """Main function"""
    data = {"name": "test"}
    if validate_input(data):
        formatted = format_data(data)
        print(formatted)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    # Create utils directory
    utils_dir = test_dir / "utils"
    utils_dir.mkdir(exist_ok=True)
    
    # Create utils/__init__.py
    (utils_dir / "__init__.py").write_text('"""Utils package"""')
    
    # Create utils/helpers.py
    helpers_content = '''"""Helper functions"""
import json
from typing import Dict, Any

def format_data(data: Dict[str, Any]) -> str:
    """Format data as JSON string"""
    return json.dumps(data, indent=2)

def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data"""
    return {k: v.upper() if isinstance(v, str) else v for k, v in data.items()}
'''
    
    # Create utils/validators.py that imports helpers
    validators_content = '''"""Validation functions"""
from typing import Dict, Any
from .helpers import format_data

def validate_input(data: Dict[str, Any]) -> bool:
    """Validate input data"""
    if not isinstance(data, dict):
        return False
    if "name" not in data:
        return False
    # Log validation
    print(f"Validating: {format_data(data)}")
    return True

def validate_output(data: str) -> bool:
    """Validate output data"""
    return len(data) > 0
'''
    
    # Write files
    (test_dir / "main.py").write_text(main_content)
    (utils_dir / "helpers.py").write_text(helpers_content)
    (utils_dir / "validators.py").write_text(validators_content)
    
    return test_dir


def test_dependency_aware_search():
    """Test the dependency-aware search functionality"""
    print("Setting up test files...")
    test_dir = setup_test_files()
    
    try:
        # Reset dependency graph
        reset_dependency_graph()
        
        # Index the test directory
        print(f"\nIndexing test directory: {test_dir}")
        result = index_directory(str(test_dir), recursive=True)
        print(f"Indexed {result.get('total_indexed', 0)} files")
        
        # Wait a bit for indexing to complete
        time.sleep(2)
        
        # Test 1: Search without dependencies
        print("\n=== Test 1: Search WITHOUT dependencies ===")
        results = search("format_data json", n_results=5, include_dependencies=False)
        print(f"Found {len(results.get('results', []))} results")
        for i, res in enumerate(results.get('results', [])):
            print(f"{i+1}. {res.get('file_path', 'unknown')} (score: {res.get('score', 0):.3f})")
        
        # Test 2: Search with dependencies
        print("\n=== Test 2: Search WITH dependencies ===")
        results = search("format_data json", n_results=10, include_dependencies=True)
        print(f"Found {len(results.get('results', []))} results")
        for i, res in enumerate(results.get('results', [])):
            is_dep = " [DEPENDENCY]" if res.get('is_dependency') else ""
            print(f"{i+1}. {res.get('file_path', 'unknown')} (score: {res.get('score', 0):.3f}){is_dep}")
        
        # Test 3: Code-specific search with dependencies
        print("\n=== Test 3: Code search WITH dependencies ===")
        results = search_code("validate", language="python", n_results=10, include_dependencies=True)
        print(f"Found {len(results.get('results', []))} results")
        for i, res in enumerate(results.get('results', [])):
            is_dep = " [DEPENDENCY]" if res.get('is_dependency') else ""
            print(f"{i+1}. {res.get('file_path', 'unknown')} (score: {res.get('score', 0):.3f}){is_dep}")
        
        # Show dependency graph statistics
        print("\n=== Dependency Graph Statistics ===")
        dep_graph = get_dependency_graph()
        stats = dep_graph.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test specific file dependencies
        print("\n=== File Dependencies ===")
        main_file = str(test_dir / "main.py")
        deps = dep_graph.get_file_dependencies(main_file)
        if deps:
            print(f"\n{main_file}:")
            print(f"  Imports: {[imp.module for imp in deps.imports]}")
            print(f"  Imported by: {list(deps.imported_by)}")
        
        helpers_file = str(test_dir / "utils" / "helpers.py")
        deps = dep_graph.get_file_dependencies(helpers_file)
        if deps:
            print(f"\n{helpers_file}:")
            print(f"  Imports: {[imp.module for imp in deps.imports]}")
            print(f"  Imported by: {list(deps.imported_by)}")
        
    finally:
        # Cleanup
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print("\nCleaned up test files")


if __name__ == "__main__":
    test_dependency_aware_search()