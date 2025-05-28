# Validation Queries for Dependency-Aware Search (v0.1.9)

When you restart Claude Code with the updated MCP server, use these queries to validate the new dependency-aware search feature.

## 1. Basic Validation

First, ensure the MCP server is working correctly:

```
"What's my current project context?"
"Check system health"
```

## 2. Index Your Code

Make sure your code is indexed:

```
"Index all Python files in this project"
# or
"Reindex the src directory"
```

## 3. Test Dependency-Aware Search

### Without dependencies (baseline):
```
"Search for 'get_embedding_model'"
"Find files containing 'ast_chunker'"
```

### With dependencies (new feature):
```
"Search for 'get_embedding_model' and include files that import it"
"Find 'ast_chunker' including dependent files"
"Search for 'DependencyGraphBuilder' with related files"
```

## 4. Specific Test Cases

To see the dependency feature in action:

```
# This should show hybrid_search.py and files that import it
"Search for 'get_hybrid_searcher' including dependencies"

# This should show ast_chunker.py and code_indexer.py (which imports it)
"Find 'create_ast_chunker' with dependent files"

# This should show logging.py and many files that import it
"Search for 'get_project_logger' including related files"
```

## 5. Code-specific Search with Dependencies

```
"Search Python code for 'index_code' including dependencies"
"Find JavaScript files containing 'import' with related files"
```

## What to Look For

1. **Results marked [DEPENDENCY]** - These are files included because they import or are imported by the main results
2. **Lower scores for dependencies** - Dependencies appear with 0.7x the score of direct matches
3. **More comprehensive results** - You should see related files that help understand the code context

## Expected Behavior

- Direct search results appear first with higher scores
- Related files (imports/importers) appear after with lower scores
- Results clearly indicate which files are dependencies
- The search provides better context for understanding code relationships

## Technical Details

The feature is backwards compatible - searches work normally unless you specifically ask to:
- "include dependencies"
- "with related files"
- "including dependent files"
- "and files that import it"

## Under the Hood

When `include_dependencies=True`:
1. The search finds direct matches first
2. For each result, it looks up files that import or are imported by that file
3. These related files are fetched and added to results with reduced scores
4. The dependency graph is built from AST data during indexing

## Troubleshooting

If dependencies aren't showing:
1. Ensure files are indexed with v0.1.9+ (reindex if needed)
2. Check that the files have import statements
3. Verify the dependency graph is being built during indexing
4. Look for Python files first (best AST support)