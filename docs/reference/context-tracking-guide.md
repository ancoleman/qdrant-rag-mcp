# Context Tracking Guide

This guide explains how to use and test the context tracking feature (v0.3.1) that provides visibility into Claude's context window usage.

## Overview

The context tracking system helps developers understand:
- How much of Claude's context window is being used
- What searches and operations have been performed
- Which operations consume the most tokens
- When they're approaching context limits

## Quick Start

### Basic Commands

```bash
# Check current context usage
"Get context status"
"What's my context usage?"
"Show context summary"

# View timeline of events
"Get context timeline"
"Show my search history"

# Get natural language summary
"Get context summary"
"What do you know about this session?"
```

## Understanding Context Usage

### Token Estimation

The system uses a simple approximation:
- **1 token ≈ 4 characters** of text
- Context window: **50,000 tokens** (configurable)
- System prompt: ~2,000 tokens overhead

### What Consumes Tokens

1. **Search Operations**
   - General search: 500-2,000 tokens per search
   - Code search: 1,000-2,500 tokens (includes context)
   - Documentation search: 500-1,500 tokens

2. **File Operations**
   - This server doesn't have a Read tool
   - File content is only read during indexing (doesn't consume context)

3. **Tool Uses**
   - Most tools add 100-500 tokens
   - Search tools are the primary token consumers

## Testing Context Tracking

### 1. Start a Fresh Session

```bash
# Initialize with correct working directory
"Get pwd, export MCP_CLIENT_CWD to that value, then run health check"
```

### 2. Build Up Context

Perform various searches to add to context:

```bash
# Different search types
"Search for context tracking implementation"
"Search code for SessionContextTracker class"
"Search docs for context tracking guide"
```

### 3. Check Context Status

```bash
"Get context status"
```

Example response:
```json
{
  "session_id": "abc123-def456-...",
  "uptime_minutes": 5,
  "context_usage": {
    "estimated_tokens": 3500,
    "percentage_used": 7.0,
    "tokens_remaining": 46500
  },
  "activity_summary": {
    "files_read": 0,
    "searches_performed": 3,
    "indexed_directories": 0
  },
  "token_breakdown": {
    "system_prompt": 2000,
    "searches": 1500,
    "files": 0,
    "tools": 0
  },
  "top_files": [],
  "usage_status": {
    "status": "OK"
  }
}
```

### 4. View Timeline

```bash
"Get context timeline"
```

Returns chronological list of events:
```json
[
  {
    "timestamp": "2025-01-06T15:30:45.123Z",
    "event_type": "search",
    "tokens_estimate": 500,
    "details": {
      "query": "context tracking implementation",
      "search_type": "general",
      "results_count": 5
    }
  }
]
```

### 5. Get Human-Readable Summary

```bash
"Get context summary"
```

Example output:
```
Current Session Context Summary:
- Session started: 5 minutes ago
- Current project: qdrant_rag

Activity:
- Files read: 0 files
- Searches performed: 3 searches
- Directories indexed: 0 directories

Context usage: 7.0% of available window
- Estimated tokens used: 3,500
- Tokens remaining: 46,500

Recent searches:
  - "context tracking implementation" (general, 5 results)
  - "SessionContextTracker class" (code, 3 results)
  - "context tracking guide" (documentation, 2 results)
```

## Testing Warning Thresholds

### Moderate Usage Warning (60%)

Perform many searches to reach 60% usage:

```bash
# Multiple searches
"Search for authentication patterns"
"Search code for error handling"
"Search for database configuration"
# ... continue until warning appears

"Get context summary"
```

You'll see:
```
⚠️ Context window is 65% full
   Be mindful of large file reads and search results
```

### High Usage Warning (80%)

Continue adding to context:

```bash
# More searches with expanded results
"Search for common patterns with 10 results"
"Search all Python files for imports"

"Get context summary"
```

You'll see:
```
⚠️ Context window is 85% full
   Consider clearing non-essential context or starting a new session
```

## Session Persistence

Sessions are automatically saved every 5 minutes and on shutdown.

### View Saved Sessions

Use the session viewer utility:

```bash
# List recent sessions
./scripts/qdrant-sessions

# Example output:
Session ID                               Started              Duration   Tokens     Files  Searches
----------------------------------------------------------------------------------------------------
abc123-def456-789012-345678-901234567890 2025-01-06 15:30    25m        15000      0      12
def456-789012-345678-901234-567890abcdef 2025-01-06 14:00    45m        35000      0      28
```

### View Session Details

```bash
./scripts/qdrant-sessions -s SESSION_ID
```

Shows:
- Basic information (start time, duration, project)
- Token usage breakdown
- Activity summary
- Top token-consuming operations
- Search history

### View Session Timeline

```bash
./scripts/qdrant-sessions -t SESSION_ID
```

Shows chronological event list with timestamps.

### Analyze All Sessions

```bash
./scripts/qdrant-sessions --analyze
```

Provides aggregate statistics across all sessions.

## Configuration

Edit `config/server_config.json` to customize:

```json
{
  "context_tracking": {
    "enabled": true,
    "context_window": {
      "size": 50000,              // Total context window size
      "warning_threshold": 0.8,    // Warning at 80% usage
      "critical_threshold": 0.9    // Critical at 90% usage
    },
    "token_estimation": {
      "method": "simple",
      "chars_per_token": 4        // Characters per token estimate
    },
    "auto_save": {
      "enabled": true,
      "interval_seconds": 300,    // Save every 5 minutes
      "on_shutdown": true
    }
  }
}
```

## Advanced Testing Scenarios

### 1. Test Different Search Types

```bash
# General search (uses all collections)
"Search for error handling patterns"

# Code-specific search
"Search code for class definitions"

# Documentation search
"Search docs for installation guide"

# Config search
"Search config for database settings"

# Check token usage by type
"Get context status"
```

### 2. Test Search Modifiers

```bash
# Search with more results (uses more tokens)
"Search for main function with 10 results"

# Search with context expansion
"Search code for authenticate with context"

# Cross-project search (searches more collections)
"Search for login across all projects"
```

### 3. Monitor Real Work

During actual development:

```bash
# Periodic checks
"How much context am I using?"
"What's consuming the most tokens?"
"Show my recent searches"

# Before large operations
"Check context status before searching"
"Am I close to the limit?"
```

### 4. Test Session Continuity

```bash
# Note current session ID
"Get context status"

# Exit Claude Code
# Restart Claude Code

# Check new session
"Get context status"
# Should show different session ID
```

## Best Practices

### 1. **Regular Monitoring**
- Check context usage every 10-15 operations
- Use `"Show context usage"` as a quick check

### 2. **Efficient Searching**
- Use specific queries to get fewer, more relevant results
- Limit result count when appropriate: `"Search for X with 3 results"`

### 3. **Context Management**
- Start new sessions for different tasks
- Be aware of cumulative token usage
- Plan large search operations when context is low

### 4. **Using the Data**
- Review session timelines to understand patterns
- Use the session analyzer to optimize workflows
- Identify which operations use the most tokens

## Troubleshooting

### "Context tracking not available"
- Ensure you're using v0.3.1 or later
- Check that context_tracking is enabled in config

### "No session data"
- Sessions are saved to `~/.mcp-servers/qdrant-rag/sessions/`
- Check directory permissions
- Ensure auto_save is enabled

### "Token estimates seem wrong"
- The 4 chars/token is an approximation
- Actual usage varies by content type
- Code typically uses more tokens than prose

### "Warnings not appearing"
- Check warning thresholds in configuration
- Ensure you're checking with `get_context_summary`
- Warnings only appear above threshold percentages

## Understanding the Output

### Context Status Fields

- **session_id**: Unique identifier for this session
- **uptime_minutes**: How long the session has been active
- **estimated_tokens**: Total tokens used (approximate)
- **percentage_used**: Portion of context window consumed
- **token_breakdown**: Tokens by category (system, searches, etc.)
- **usage_status**: OK, WARNING, or CRITICAL state

### Timeline Event Types

- **search**: Any search operation
- **file_read**: File content added (not used in this server)
- **tool_use**: Other tool invocations
- **index_directory**: Directory indexing operations

### Session Persistence

- Sessions saved as JSON in `~/.mcp-servers/qdrant-rag/sessions/`
- Filename format: `session_[ID]_[TIMESTAMP].json`
- Automatically cleaned up after max_session_files limit

## Integration with Development Workflow

### Starting a Project Session

```bash
# 1. Set up context
"Get pwd, export MCP_CLIENT_CWD to that value, then check health"

# 2. Initial exploration
"Search for main entry points"
"Get context status"

# 3. Focused work
"Search code for authentication implementation"
"Search for related tests"

# 4. Monitor usage
"Show context summary"
```

### During Code Review

```bash
# Check context before starting
"What's my current context usage?"

# Search for patterns
"Search for similar implementations"
"Search code for error handling patterns"

# Check if approaching limits
"Am I running out of context?"
```

### End of Session

```bash
# Review what was done
"Get context timeline"
"Show session summary"

# Session is auto-saved on exit
```

This context tracking system helps you work more efficiently with Claude by understanding and managing the context window effectively.