# Examples and Test Data

This directory contains example test data, sample projects, and validation guides used for testing and development.

## Structure

- **test_projects/** - Sample projects for integration testing
  - `reindex-test/` - Used for testing reindexing functionality
  
- **validation_guides/** - Guides for manual validation of features
  - `validation_dep_aware.md` - Testing dependency-aware search
  - `test_reindex.md` - Testing reindex functionality

- **simple_dependency_test.py** - Simple example of dependency testing

## Usage

### Test Projects
Used by integration tests to have realistic project structures:
```python
test_project = Path(__file__).parent.parent / "examples/test_projects/reindex-test"
```

### Validation Guides
Step-by-step guides for manually testing features in Claude Code:
- Follow the instructions in the markdown files
- Use these when developing new features or debugging issues

### Example Scripts
Simple standalone examples demonstrating specific functionality