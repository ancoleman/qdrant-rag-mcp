# Progressive Search Usage Guide

This guide explains how to use Progressive Context Management effectively in Claude Code to reduce token usage by 50-70% while maintaining access to full details when needed.

## Overview

Progressive Context Management (v0.3.2) introduces multi-level context retrieval that automatically adapts to your query intent. Instead of always returning full code chunks, it provides the appropriate level of detail based on what you're trying to accomplish.

## Token Reduction by Context Level

| Context Level | Token Reduction | Best For | Example Queries |
|---------------|-----------------|----------|-----------------|
| File | 70% | Understanding architecture | "What does the auth system do?" |
| Class | 50% | Navigation & exploration | "Find the UserManager class" |
| Method | 20% | Debugging & implementation | "Fix the bug in save_user" |
| Full | 0% | Complete details | "Show complete validation logic" |

## Configuration Options

### 1. Default Configuration (Set Once)

Edit `config/server_config.json` to change the default behavior:

```json
"progressive_context": {
  "enabled": true,          // true = progressive ON, false = regular search
  "default_level": "auto",  // "auto", "file", "class", "method", or "full"
  "cache": {
    "enabled": true,
    "similarity_threshold": 0.85,  // Cache hits for 85%+ similar queries
    "ttl_seconds": 3600           // Cache expires after 1 hour
  }
}
```

### 2. Toggle Progressive Mode

Use the provided toggle script to switch between progressive and regular search:

```bash
# Toggle progressive context on/off
./scripts/toggle_progressive.sh

# Check current status
grep -A2 '"progressive_context"' config/server_config.json
```

## Runtime Control in Claude Code

### Force Progressive Mode ON

Even if disabled in config, you can use progressive search:

```bash
# Explicit progressive mode
"Search for authentication with progressive_mode=true"

# Specify context level (implies progressive mode)
"Find database connections with context_level=file"
"Search for UserModel with context_level=class"
"Debug save_user error with context_level=method"
```

### Force Progressive Mode OFF

Even if enabled in config, you can use regular search:

```bash
# Explicit regular search
"Search for validation functions with progressive_mode=false"

# Use 'full' level for no reduction
"Search for logger with context_level=full"
```

### Let Auto-Detection Work

The system analyzes your query intent automatically:

```bash
# These queries auto-detect the appropriate level:
"What does the configuration system do?"     # → File level (70% reduction)
"Show me the DatabaseManager class"          # → Class level (50% reduction)
"Fix the error in the validate function"     # → Method level (20% reduction)
"How does caching work in this project?"     # → File level (70% reduction)
"Find where User.save is implemented"        # → Method level (20% reduction)
```

## Query Patterns and Auto-Detection

### File Level (70% reduction) - "Understanding"
Triggered by keywords like:
- "what does", "explain", "overview", "how does"
- "architecture", "structure", "purpose of"
- "understand", "describe", "summary"

Examples:
```bash
"What does the authentication system do?"
"Explain the database architecture"
"Give me an overview of the API structure"
```

### Class Level (50% reduction) - "Navigation"
Triggered by keywords like:
- "find", "where is", "show me", "locate"
- "search for", "looking for", "need"

Examples:
```bash
"Find the ConfigManager class"
"Where is the user validation implemented?"
"Show me the API router setup"
```

### Method Level (20% reduction) - "Debugging"
Triggered by keywords like:
- "bug in", "error in", "fix", "debug"
- "implementation", "specific", "exact"
- "line", "trace", "issue", "problem"

Examples:
```bash
"Debug the error in save_user function"
"Fix the bug on line 234"
"Show exact implementation of validate_token"
```

## Semantic Caching

Similar queries automatically use cached results:

```bash
# First query - hits the database
"What does the authentication system do?"

# Similar query - uses cache (85%+ similarity)
"Explain the auth system"
"How does authentication work?"

# Cache metadata shows in results:
# "cache_hit": true
# "similarity": 0.92
```

### Cache Management

```bash
# Cache is automatic, but you can control it:
"Search for config with semantic_cache=false"  # Skip cache

# Cache persists across sessions
# Located at: ~/.mcp-servers/qdrant-rag/progressive_cache/
# TTL: 1 hour by default
```

## Expansion Options

Progressive results include drill-down options:

```bash
# Start with overview
"What does the user management system do?"
# Returns file-level summary + expansion options

# Response includes:
# "expansion_options": [
#   {
#     "type": "class",
#     "path": "auth/user.py::UserManager",
#     "estimated_tokens": 800,
#     "relevance": 0.92
#   }
# ]

# Then drill down to specific class
"Show me the UserManager class in detail"
```

## Best Practices

### 1. Start Broad, Then Narrow
```bash
# First: Understand the system
"What does the payment processing do?"  # File level

# Then: Find specific components
"Show me the PaymentGateway class"     # Class level

# Finally: Debug specific issues
"Fix the timeout in process_payment"    # Method level
```

### 2. Use Explicit Levels for Consistency
```bash
# When you know what you want:
"Search for all validators with context_level=class"
"Find error handlers with context_level=file"
"Show complete auth flow with context_level=full"
```

### 3. Disable for Detail-Heavy Tasks
```bash
# When you need everything:
"Review the entire SecurityManager with progressive_mode=false"
"Audit all database queries with context_level=full"
```

### 4. Monitor Token Savings
Look for token metrics in responses:
- `token_estimate`: Tokens used with progressive
- `token_reduction`: Percentage saved
- `level_used`: Which level was applied

## Troubleshooting

### Not Getting Expected Reduction?
```bash
# Check if progressive is enabled
"Show progressive context configuration"

# Force specific level
"Search for auth with context_level=file"

# Verify with full search
"Search for auth with progressive_mode=false"
```

### Cache Not Working?
```bash
# Check cache settings
grep -A5 '"cache"' config/server_config.json

# Skip cache for testing
"Search for config with semantic_cache=false"
```

### Wrong Auto-Detection?
```bash
# Override with explicit level
"Find UserModel with context_level=class"  # Instead of auto-detected
```

## Examples by Use Case

### Code Review
```bash
# Get high-level understanding
"What do the API endpoints do?" # 70% reduction

# Review specific endpoints
"Show me the user endpoints with context_level=class" # 50% reduction
```

### Debugging
```bash
# Find the problem area
"Where are database connections handled?" # Auto → class level

# Get full implementation
"Show complete connection pool logic with progressive_mode=false"
```

### Learning New Codebase
```bash
# Start with architecture
"Explain the project structure" # 70% reduction

# Explore key components
"Find the main service classes" # 50% reduction

# Dive into specifics
"How is authentication implemented in UserService?" # 20% reduction
```

### Performance Optimization
```bash
# Find all database queries
"Search for database queries with context_level=file" # Get overview

# Analyze specific queries
"Show the user lookup queries with context_level=method" # Get details
```

## Configuration Reference

Full configuration options in `server_config.json`:

```json
{
  "progressive_context": {
    "enabled": true,
    "default_level": "auto",
    "cache": {
      "enabled": true,
      "similarity_threshold": 0.85,
      "max_cache_size": 1000,
      "ttl_seconds": 3600,
      "persistence_enabled": true,
      "persistence_path": "~/.mcp-servers/qdrant-rag/progressive_cache"
    },
    "levels": {
      "file": {
        "token_reduction_target": 0.7
      },
      "class": {
        "token_reduction_target": 0.5
      },
      "method": {
        "token_reduction_target": 0.2
      }
    },
    "query_classification": {
      "confidence_threshold": 0.7,
      "fallback_level": "class"
    }
  }
}
```