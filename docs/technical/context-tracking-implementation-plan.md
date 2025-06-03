# Context Tracking Implementation Plan

## Overview

This document outlines the implementation plan for adding context tracking capabilities to the Qdrant RAG MCP Server. This feature will provide developers with transparency into what information Claude has in its context window during a session.

## Motivation

When working with AI assistants like Claude, developers often wonder:
- What files has Claude read in this session?
- How much of the context window is being used?
- What searches have been performed?
- Why might Claude have "forgotten" something from earlier?

Context tracking addresses these concerns by providing real-time visibility into context usage.

## Architecture Design

### 1. Core Components

#### A. Session Tracker Class
```python
class SessionContextTracker:
    """Tracks all context-consuming operations in a session"""
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        self.context_events = []
        self.files_read = {}
        self.searches_performed = []
        self.todos_tracked = []
        self.indexed_directories = []
        self.total_tokens_estimate = 0
        
    def track_file_read(self, file_path: str, content: str, metadata: dict):
        """Track when a file is read into context"""
        
    def track_search(self, query: str, results: list, search_type: str):
        """Track search operations and results"""
        
    def track_tool_use(self, tool_name: str, params: dict, result_size: int):
        """Track any tool usage that adds to context"""
        
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
```

#### B. Persistent Session Storage
```python
class SessionStore:
    """Manages persistent storage of session data"""
    
    def __init__(self, base_dir: Path):
        self.sessions_dir = base_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        
    def save_session(self, tracker: SessionContextTracker):
        """Save session data to JSON file"""
        
    def load_session(self, session_id: str) -> dict:
        """Load session data from file"""
        
    def get_current_session(self) -> dict:
        """Get the most recent session"""
```

### 2. MCP Tool Integration

#### New MCP Tools

```python
@mcp_tool
def get_context_status() -> dict:
    """Get current context window usage and statistics"""
    return {
        "session_id": tracker.session_id,
        "uptime_minutes": (datetime.now() - tracker.session_start).seconds // 60,
        "context_usage": {
            "estimated_tokens": tracker.total_tokens_estimate,
            "percentage_used": (tracker.total_tokens_estimate / 50000) * 100,
            "tokens_remaining": 50000 - tracker.total_tokens_estimate
        },
        "files_in_context": len(tracker.files_read),
        "searches_performed": len(tracker.searches_performed),
        "largest_files": sorted(
            tracker.files_read.items(), 
            key=lambda x: x[1]["tokens"], 
            reverse=True
        )[:5]
    }

@mcp_tool
def get_context_timeline() -> list:
    """Get chronological timeline of context events"""
    return tracker.context_events

@mcp_tool
def clear_context_except(keep_files: list = None) -> dict:
    """Clear context except for specified essential files"""
    # This would need Claude Code support for actual context clearing
    # For now, it can track what *should* be cleared
    
@mcp_tool
def get_context_summary() -> str:
    """Get a natural language summary of current context"""
    return f"""
Current Session Context Summary:
- Session started: {tracker.session_start}
- Files read: {len(tracker.files_read)} files totaling ~{sum(f['tokens'] for f in tracker.files_read.values())} tokens
- Top files: {', '.join(list(tracker.files_read.keys())[:3])}
- Searches: {len(tracker.searches_performed)} searches performed
- Current project: {tracker.current_project}
- Context usage: {tracker.total_tokens_estimate / 50000:.1%} of available window
"""
```

### 3. Integration Points

#### A. Modify Existing Tools

```python
# In Read tool
def read_file(file_path: str, ...):
    content = # ... existing read logic
    
    # Track the read
    tracker.track_file_read(
        file_path=file_path,
        content=content,
        metadata={
            "lines": len(content.splitlines()),
            "characters": len(content),
            "timestamp": datetime.now()
        }
    )
    
    return content

# In Search tools
def search(query: str, ...):
    results = # ... existing search logic
    
    # Track the search
    tracker.track_search(
        query=query,
        results=results,
        search_type="general"
    )
    
    return results
```

#### B. Session File Format

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "session_start": "2025-01-06T14:30:00Z",
  "project_context": {
    "path": "/Users/user/projects/my-app",
    "name": "my-app",
    "collection_prefix": "project_a1b2c3"
  },
  "context_events": [
    {
      "timestamp": "2025-01-06T14:30:05Z",
      "event_type": "file_read",
      "details": {
        "file_path": "README.md",
        "lines": 500,
        "tokens_estimate": 3500
      }
    },
    {
      "timestamp": "2025-01-06T14:30:10Z",
      "event_type": "search",
      "details": {
        "query": "authentication",
        "search_type": "code",
        "results_count": 5,
        "tokens_estimate": 2000
      }
    }
  ],
  "summary": {
    "total_files_read": 4,
    "total_searches": 2,
    "total_tokens_estimate": 15000,
    "context_percentage": 30
  }
}
```

### 4. Visual Indicators

#### A. Context Status in Responses

Add a lightweight context indicator that can be included in responses:

```python
def get_context_indicator() -> str:
    """Get a compact context status indicator"""
    return f"[Context: {tracker.total_tokens_estimate//1000}k/{50}k tokens | {len(tracker.files_read)} files | {len(tracker.searches_performed)} searches]"
```

#### B. Progress Warnings

```python
def check_context_usage() -> dict:
    """Check if context usage is approaching limits"""
    usage_percent = (tracker.total_tokens_estimate / 50000) * 100
    
    if usage_percent > 80:
        return {
            "warning": "HIGH_CONTEXT_USAGE",
            "message": f"Context window is {usage_percent:.0f}% full",
            "suggestion": "Consider clearing non-essential context"
        }
    elif usage_percent > 60:
        return {
            "warning": "MODERATE_CONTEXT_USAGE",
            "message": f"Context window is {usage_percent:.0f}% full",
            "suggestion": "Be mindful of large file reads"
        }
    
    return {"status": "OK", "usage_percent": usage_percent}
```

### 5. Configuration

Add to `server_config.json`:

```json
{
  "context_tracking": {
    "enabled": true,
    "session_dir": "${HOME}/.mcp-servers/qdrant-rag/sessions",
    "max_session_files": 100,
    "token_estimation_method": "simple",
    "context_window_size": 50000,
    "warning_threshold": 0.8,
    "auto_save_interval": 300,
    "include_in_responses": false
  }
}
```

## Implementation Phases

### Phase 1: Core Tracking (Week 1)
1. Implement `SessionContextTracker` class
2. Add token estimation logic
3. Integrate with Read and Search tools
4. Create session storage mechanism

### Phase 2: MCP Tools (Week 2)
1. Implement `get_context_status` tool
2. Implement `get_context_timeline` tool
3. Implement `get_context_summary` tool
4. Add context indicators to responses

### Phase 3: Advanced Features (Week 3)
1. Implement session persistence
2. Add context usage warnings
3. Create session viewer utility
4. Add configuration options

### Phase 4: Testing & Documentation (Week 4)
1. Unit tests for context tracking
2. Integration tests with MCP tools
3. Update CLAUDE.md with usage examples
4. Create user documentation

## Testing Strategy

### Unit Tests
- Token estimation accuracy
- Event tracking correctness
- Session persistence
- Context calculations

### Integration Tests
- Tool integration verification
- Session continuity across restarts
- Performance impact measurement

### User Testing
- Developer experience feedback
- Context visibility usefulness
- Performance overhead assessment

## Performance Considerations

1. **Minimal Overhead**: Tracking should add <1% latency
2. **Memory Efficient**: Store summaries, not full content
3. **Async Operations**: Save sessions asynchronously
4. **Configurable**: Allow disabling for performance-critical uses

## Security & Privacy

1. **Local Storage Only**: Sessions stored locally
2. **No Sensitive Data**: Don't store file contents, only metadata
3. **User Control**: Easy session cleanup commands
4. **Configurable Retention**: Auto-cleanup old sessions

## Success Metrics

1. **Developer Satisfaction**: Improved understanding of context usage
2. **Reduced Confusion**: Fewer "why doesn't Claude remember X" questions
3. **Better Resource Usage**: More efficient context management
4. **Performance**: <1% overhead on operations

## Future Enhancements

1. **Visual Dashboard**: Web UI for session viewing
2. **Context Optimization**: Suggestions for better context usage
3. **Pattern Analysis**: Identify common context usage patterns
4. **Export Capabilities**: Export session data for analysis

## Related Documentation

- [Advanced RAG Implementation Roadmap](./advanced-rag-implementation-roadmap.md) - Overall feature roadmap
- [Why MCP RAG for Agentic Coding](../reference/why-mcp-rag-agentic-coding.md) - Context efficiency motivation
- [GitHub Token Optimization Guide](../github-token-optimization-guide.md) - Token reduction strategies