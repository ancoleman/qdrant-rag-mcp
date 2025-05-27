# Project-Aware Logging Implementation Plan

## Overview

Since the MCP server is installed globally and used across different projects, we need a logging system that:
1. Keeps logs separated by project
2. Provides easy debugging per project
3. Doesn't mix logs from concurrent operations
4. Handles the global vs project context properly

## Design Principles

### 1. Log Directory Structure
```
~/.mcp-servers/qdrant-rag/logs/
├── global/                      # Logs before project detection
│   └── 2025-05-27.log
├── projects/                    # Project-specific logs
│   ├── project_hash_abc123/    # Hash of project path
│   │   ├── metadata.json       # Project name, path mapping
│   │   ├── 2025-05-27.log      # Daily rotating logs
│   │   └── 2025-05-26.log
│   └── project_hash_def456/
│       ├── metadata.json
│       └── 2025-05-27.log
└── errors/                      # Critical errors across all projects
    └── 2025-05-27.log
```

### 2. Log Levels and Categories
```python
# Structured log format
{
    "timestamp": "2025-05-27T10:30:45.123Z",
    "level": "INFO",
    "project": "qdrant_rag",
    "project_path": "/Users/user/repos/qdrant-rag",
    "operation": "index_code",
    "file": "src/main.py",
    "message": "Successfully indexed file",
    "duration_ms": 45,
    "metadata": {
        "chunks": 5,
        "vectors": 5,
        "collection": "project_qdrant_rag_code"
    }
}
```

### 3. Implementation Components

#### A. Logger Factory
```python
class ProjectAwareLogger:
    def __init__(self, base_log_dir: Path):
        self.base_log_dir = base_log_dir
        self.current_project = None
        self.handlers = {}
    
    def get_logger(self, project_context=None):
        if project_context:
            return self._get_project_logger(project_context)
        return self._get_global_logger()
```

#### B. Log Rotation
- Daily rotation with 7-day retention
- Size-based rotation at 10MB per file
- Compression of old logs (gzip)

#### C. Performance Considerations
- Async logging to prevent blocking MCP operations
- Buffered writes for batch operations
- Lazy initialization of project loggers

## Configuration Options

### server_config.json
```json
{
  "logging": {
    "enabled": true,
    "level": "INFO",
    "base_dir": "~/.mcp-servers/qdrant-rag/logs",
    "rotation": {
      "max_days": 7,
      "max_size_mb": 10,
      "compress": true
    },
    "categories": {
      "indexing": "DEBUG",
      "search": "INFO",
      "errors": "ERROR"
    },
    "performance": {
      "async": true,
      "buffer_size": 1000,
      "flush_interval_seconds": 5
    }
  }
}
```

### Environment Variables
```bash
# Override log level
export QDRANT_LOG_LEVEL=DEBUG

# Disable logging entirely
export QDRANT_LOGGING_ENABLED=false

# Custom log directory
export QDRANT_LOG_DIR=/custom/path/logs
```

## Log Viewer Utility

### CLI Tool: `qdrant-logs`
```bash
# View logs for current project
qdrant-logs

# View logs for specific project
qdrant-logs --project /path/to/project

# Filter by operation
qdrant-logs --operation index_code

# Filter by date range
qdrant-logs --from 2025-05-26 --to 2025-05-27

# Follow logs in real-time
qdrant-logs --follow

# Search logs
qdrant-logs --search "error.*authentication"

# Export logs
qdrant-logs --export json > logs.json
```

## Integration Points

### 1. MCP Server Start
```python
# In qdrant_mcp_context_aware.py
logger = ProjectAwareLogger(get_log_base_dir())

@mcp.context()
async def on_request_context():
    project = get_current_project()
    return {"logger": logger.get_logger(project)}
```

### 2. Operation Logging
```python
# Decorator for automatic operation logging
@log_operation("index_code")
def index_code(file_path: str, force_global: bool = False):
    # Operation logic
    pass
```

### 3. Error Handling
```python
# Global error handler
@mcp.error_handler()
async def handle_error(error: Exception, context: dict):
    error_logger = logger.get_error_logger()
    error_logger.error(
        "Unhandled error in MCP operation",
        exc_info=error,
        extra=context
    )
```

## Privacy and Security

1. **No sensitive data in logs**:
   - File paths are relative when possible
   - No file contents are logged
   - API keys/tokens are masked

2. **Log file permissions**:
   - 600 permissions (owner read/write only)
   - Logs directory: 700 permissions

3. **Configurable redaction**:
   ```python
   redaction_patterns = [
       r"api[_-]?key['\"]?\s*[:=]\s*['\"]?([^'\"]+)",
       r"token['\"]?\s*[:=]\s*['\"]?([^'\"]+)"
   ]
   ```

## Testing Strategy

1. **Unit Tests**:
   - Logger initialization
   - Project context switching
   - Log rotation logic
   - Redaction patterns

2. **Integration Tests**:
   - Multi-project concurrent logging
   - Performance under load
   - Log viewer functionality

3. **Manual Testing**:
   - Install globally and test across projects
   - Verify log separation
   - Test log viewer filters

## Rollout Plan

1. **Phase 1**: Basic structured logging
   - Project-aware file logging
   - JSON structured format
   - Basic rotation

2. **Phase 2**: Advanced features
   - Async logging
   - Log viewer utility
   - Performance optimizations

3. **Phase 3**: Enhancements
   - Web UI for log viewing
   - Log aggregation/analytics
   - Integration with monitoring tools

## Example Usage

After implementation, users will see:

```bash
# Normal operation (logs happen automatically)
claude

# Check logs for debugging
qdrant-logs --tail 50

# Investigate specific issue
qdrant-logs --operation reindex_directory --level ERROR

# Export logs for support
qdrant-logs --export --from yesterday > debug-logs.json
```

## Migration Notes

- Existing installations will start fresh with new logging
- No migration of old logs needed
- Logging is opt-out (enabled by default)
- Minimal performance impact (<1% overhead)