# GitHub MCP Tools Execution Flow

## Example: `github_list_repositories()` call

```
1. User calls: github_list_repositories(owner="octocat")
   ↓
2. @github_operation decorator intercepts the call
   ↓
3. Decorator calls validate_github_prerequisites()
   ↓
4. validate_github_prerequisites() calls _create_github_instances()
   ↓
5. _create_github_instances() returns (github_client, issue_analyzer, ...)
   ↓
6. Decorator stores instances in _github_instances context variable
   ↓
7. Decorator executes the original function
   ↓
8. Inside github_list_repositories():
   - Calls get_github_instances() [from decorator module]
   - Retrieves instances from context variable
   - Uses github_client to list repositories
   ↓
9. Decorator cleans up context (_github_instances = None)
   ↓
10. Result returned to user
```

## Key Points

- `_create_github_instances()` is called ONCE per operation by the decorator
- The decorator stores instances in a context variable
- The decorated function retrieves instances from context using `get_github_instances()`
- This ensures all GitHub operations have consistent error handling and logging