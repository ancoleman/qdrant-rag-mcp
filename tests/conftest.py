"""
Pytest configuration for Qdrant RAG MCP tests
"""

import sys
import os
from pathlib import Path

# Add src to Python path for all tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Common fixtures can be added here
import pytest

@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary directory for test files"""
    test_dir = tmp_path / "test_project"
    test_dir.mkdir()
    return test_dir

@pytest.fixture
def sample_python_file(temp_test_dir):
    """Create a sample Python file for testing"""
    file_path = temp_test_dir / "sample.py"
    file_path.write_text("""
def hello_world():
    '''A simple hello world function'''
    print("Hello, World!")
    
class ExampleClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
""")
    return file_path

@pytest.fixture
def sample_config_file(temp_test_dir):
    """Create a sample config file for testing"""
    file_path = temp_test_dir / "config.json"
    file_path.write_text("""{
    "name": "test-project",
    "version": "1.0.0",
    "settings": {
        "debug": true,
        "port": 8080
    }
}""")
    return file_path

@pytest.fixture
def sample_markdown_file(temp_test_dir):
    """Create a sample markdown file for testing"""
    file_path = temp_test_dir / "README.md"
    file_path.write_text("""# Test Project

This is a test project for unit testing.

## Features

- Feature 1
- Feature 2

## Installation

```bash
pip install test-project
```
""")
    return file_path