"""
Context tracking system for monitoring Claude's context window usage.

This module provides tools to track what information Claude has in its context
window during a session, helping developers understand and optimize their
interactions with the AI assistant.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from utils.logging import get_project_logger
from utils.memory_manager import MemoryComponent, get_memory_manager

logger = get_project_logger()


@dataclass
class ContextEvent:
    """Represents a single context-consuming event."""
    timestamp: str
    event_type: str  # file_read, search, tool_use, etc.
    details: Dict[str, Any]
    tokens_estimate: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FileReadInfo:
    """Information about a file read into context."""
    file_path: str
    read_at: str
    lines: int
    characters: int
    tokens_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def calculate_system_context_tokens() -> Dict[str, int]:
    """
    Calculate the actual system context tokens from known sources.
    
    Returns dict with breakdown of system context tokens.
    """
    system_tokens = {}
    
    # 1. Calculate CLAUDE.md tokens if it exists
    claude_md_path = Path(__file__).parent.parent.parent / "CLAUDE.md"
    if claude_md_path.exists():
        try:
            with open(claude_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Use the standard 4 chars per token approximation
                system_tokens["CLAUDE.md"] = len(content) // 4
        except:
            system_tokens["CLAUDE.md"] = 6000  # Fallback estimate
    
    # 2. Estimate MCP tool definitions
    # Each tool has name, description, parameters schema
    # Rough estimate: ~200 tokens per tool for definition
    # We have 26 tools currently
    system_tokens["mcp_tools"] = 26 * 200  # ~5200 tokens
    
    # 3. Claude's base system prompt (estimated)
    # This includes Claude Code instructions, behavior guidelines, etc.
    system_tokens["claude_system"] = 3000  # Conservative estimate
    
    # 4. Other context (file paths, system info, etc.)
    system_tokens["other_context"] = 500
    
    return system_tokens


class SessionContextTracker(MemoryComponent):
    """Tracks all context-consuming operations in a session."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize a new session tracker."""
        # Get memory limits from config
        memory_manager = get_memory_manager()
        memory_config = memory_manager.config
        component_limits = memory_config.get('component_limits', {}).get('context_tracking', {})
        
        # Initialize parent MemoryComponent
        super().__init__(
            name="context_tracking",
            max_memory_mb=float(component_limits.get('max_memory_mb', 100))
        )
        
        # Register with memory manager
        memory_manager.register_component("context_tracking", self)
        
        self.session_id = session_id or str(uuid.uuid4())
        self.session_start = datetime.now()
        self.context_events: List[ContextEvent] = []
        self.files_read: Dict[str, FileReadInfo] = {}
        self.searches_performed: List[Dict[str, Any]] = []
        self.todos_tracked: List[Dict[str, Any]] = []
        self.indexed_directories: List[str] = []
        self.current_project: Optional[Dict[str, Any]] = None
        
        # Get memory limits from config
        self.max_files_tracked = int(component_limits.get('max_files', 100))
        self.max_timeline_events = int(component_limits.get('max_timeline_events', 500))
        self.max_search_results = 50  # Not in config, keeping default
        self.max_content_preview = 1000  # chars per file
        
        # Track token usage by category
        self.token_usage = defaultdict(int)
        
        # Calculate actual system context
        system_breakdown = calculate_system_context_tokens()
        system_total = sum(system_breakdown.values())
        
        self.token_usage["system_prompt"] = system_total
        self.token_usage["system_breakdown"] = system_breakdown  # Store breakdown for transparency
        
        # Initialize total with system prompt
        self.total_tokens_estimate = self.token_usage["system_prompt"]
        
        logger.info(f"Session context tracker initialized: {self.session_id}")
        logger.info(f"System context breakdown: {system_breakdown}")
        logger.info(f"Total system context: {system_total} tokens")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Simple estimation: ~4 characters per token (rough average for English).
        This can be made more sophisticated with tiktoken or similar libraries.
        """
        return len(text) // 4
    
    def track_file_read(self, file_path: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Track when a file is read into context."""
        lines = len(content.splitlines())
        characters = len(content)
        tokens_estimate = self.estimate_tokens(content)
        
        # Create file info
        file_info = FileReadInfo(
            file_path=file_path,
            read_at=datetime.now().isoformat(),
            lines=lines,
            characters=characters,
            tokens_estimate=tokens_estimate,
            metadata=metadata or {}
        )
        
        # Store file info
        self.files_read[file_path] = file_info
        
        # Create event
        event = ContextEvent(
            timestamp=datetime.now().isoformat(),
            event_type="file_read",
            details={
                "file_path": file_path,
                "lines": lines,
                "characters": characters,
                "metadata": metadata
            },
            tokens_estimate=tokens_estimate
        )
        
        # Limit number of timeline events
        if len(self.context_events) >= self.max_timeline_events:
            # Remove oldest events
            self.context_events = self.context_events[-(self.max_timeline_events-1):]
        
        self.context_events.append(event)
        self.total_tokens_estimate += tokens_estimate
        self.token_usage["files"] += tokens_estimate
        
        # Limit number of files tracked
        if len(self.files_read) >= self.max_files_tracked:
            # Remove oldest file (first key in dict)
            oldest_key = next(iter(self.files_read))
            del self.files_read[oldest_key]
        
        logger.debug(f"Tracked file read: {file_path} (~{tokens_estimate} tokens)")
    
    def track_search(self, query: str, results: List[Dict[str, Any]], search_type: str = "general"):
        """Track search operations and results."""
        # Limit number of results tracked to save memory
        tracked_results = results[:10]  # Only track top 10 results
        
        # Estimate tokens for search results
        # Only count the content that Claude actually sees, not all the metadata
        content_length = 0
        for result in tracked_results:
            # Count the main content
            content_length += len(result.get("content", ""))
            # Count expanded content if present
            content_length += len(result.get("expanded_content", ""))
            # Add some overhead for formatting (file paths, scores, etc)
            content_length += 200  # Rough estimate for metadata per result
        
        # Add query length
        content_length += len(query)
        
        # Convert character count to token estimate (4 chars ≈ 1 token)
        tokens_estimate = content_length // 4
        
        # Create lightweight search info (don't store full results)
        search_info = {
            "query": query,
            "search_type": search_type,
            "results_count": len(results),
            "timestamp": datetime.now().isoformat(),
            "tokens_estimate": tokens_estimate,
            "top_results": [
                {
                    "file_path": r.get("file_path", ""),
                    "score": r.get("score", 0),
                    "type": r.get("chunk_type", "")
                }
                for r in tracked_results
            ]
        }
        
        # Limit total searches tracked
        if len(self.searches_performed) >= self.max_search_results:
            # Remove oldest searches
            self.searches_performed = self.searches_performed[-(self.max_search_results-1):]
        
        self.searches_performed.append(search_info)
        
        # Create event
        event = ContextEvent(
            timestamp=datetime.now().isoformat(),
            event_type="search",
            details={
                "query": query,
                "search_type": search_type,
                "results_count": len(results)
            },
            tokens_estimate=tokens_estimate
        )
        
        # Limit number of timeline events
        if len(self.context_events) >= self.max_timeline_events:
            # Remove oldest events
            self.context_events = self.context_events[-(self.max_timeline_events-1):]
        
        self.context_events.append(event)
        self.total_tokens_estimate += tokens_estimate
        self.token_usage["searches"] += tokens_estimate
        
        logger.debug(f"Tracked search: '{query}' ({search_type}) ~{tokens_estimate} tokens")
    
    def track_tool_use(self, tool_name: str, params: Dict[str, Any], result_size: int):
        """Track any tool usage that adds to context."""
        # Estimate tokens for tool interaction
        params_text = json.dumps(params)
        tokens_estimate = self.estimate_tokens(params_text) + (result_size // 4)
        
        # Create event
        event = ContextEvent(
            timestamp=datetime.now().isoformat(),
            event_type="tool_use",
            details={
                "tool_name": tool_name,
                "params": params,
                "result_size": result_size
            },
            tokens_estimate=tokens_estimate
        )
        
        # Limit number of timeline events
        if len(self.context_events) >= self.max_timeline_events:
            # Remove oldest events
            self.context_events = self.context_events[-(self.max_timeline_events-1):]
        
        self.context_events.append(event)
        self.total_tokens_estimate += tokens_estimate
        self.token_usage["tools"] += tokens_estimate
        
        logger.debug(f"Tracked tool use: {tool_name} ~{tokens_estimate} tokens")
    
    def track_index_operation(self, directory: str, file_count: int):
        """Track directory indexing operations."""
        self.indexed_directories.append(directory)
        
        # Create event
        event = ContextEvent(
            timestamp=datetime.now().isoformat(),
            event_type="index_directory",
            details={
                "directory": directory,
                "file_count": file_count
            },
            tokens_estimate=0  # Indexing doesn't consume context tokens
        )
        
        # Limit number of timeline events
        if len(self.context_events) >= self.max_timeline_events:
            # Remove oldest events
            self.context_events = self.context_events[-(self.max_timeline_events-1):]
        
        self.context_events.append(event)
        logger.debug(f"Tracked index operation: {directory} ({file_count} files)")
    
    def set_current_project(self, project_info: Dict[str, Any]):
        """Set the current project context."""
        self.current_project = project_info
        logger.debug(f"Set current project: {project_info.get('name', 'unknown')}")
    
    def get_context_usage_percentage(self, context_window_size: int = 200000) -> float:
        """Get percentage of context window used."""
        return (self.total_tokens_estimate / context_window_size) * 100
    
    def get_top_files_by_tokens(self, limit: int = 5) -> List[tuple]:
        """Get files consuming the most tokens."""
        sorted_files = sorted(
            self.files_read.items(),
            key=lambda x: x[1].tokens_estimate,
            reverse=True
        )
        return [(path, info.tokens_estimate) for path, info in sorted_files[:limit]]
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        uptime_seconds = (datetime.now() - self.session_start).total_seconds()
        
        # Get system breakdown if available
        system_breakdown = self.token_usage.get("system_breakdown", {})
        
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "uptime_minutes": int(uptime_seconds // 60),
            "total_events": len(self.context_events),
            "files_read": len(self.files_read),
            "searches_performed": len(self.searches_performed),
            "indexed_directories": len(self.indexed_directories),
            "total_tokens_estimate": self.total_tokens_estimate,
            "token_usage_by_category": dict(self.token_usage),
            "system_context_breakdown": system_breakdown,
            "context_usage_percentage": self.get_context_usage_percentage(),
            "current_project": self.current_project
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire session to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "current_project": self.current_project,
            "context_events": [event.to_dict() for event in self.context_events],
            "files_read": {
                path: {
                    "file_path": info.file_path,
                    "read_at": info.read_at,
                    "lines": info.lines,
                    "characters": info.characters,
                    "tokens_estimate": info.tokens_estimate,
                    "metadata": info.metadata
                }
                for path, info in self.files_read.items()
            },
            "searches_performed": self.searches_performed,
            "indexed_directories": self.indexed_directories,
            "summary": self.get_session_summary()
        }


class SessionStore:
    """Manages persistent storage of session data."""
    
    def __init__(self, base_dir: Path):
        """Initialize session store with base directory."""
        self.sessions_dir = base_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Session store initialized at: {self.sessions_dir}")
    
    def get_session_file_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.sessions_dir / f"session_{session_id}_{timestamp}.json"
    
    def save_session(self, tracker: SessionContextTracker):
        """Save session data to JSON file."""
        file_path = self.get_session_file_path(tracker.session_id)
        
        try:
            with open(file_path, 'w') as f:
                json.dump(tracker.to_dict(), f, indent=2)
            logger.info(f"Session saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from file."""
        # Find the most recent session file with this ID
        session_files = list(self.sessions_dir.glob(f"session_{session_id}_*.json"))
        
        if not session_files:
            logger.warning(f"No session found with ID: {session_id}")
            return None
        
        # Sort by modification time and get the most recent
        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return None
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """Get the most recent session."""
        session_files = list(self.sessions_dir.glob("session_*.json"))
        
        if not session_files:
            return None
        
        # Sort by modification time and get the most recent
        latest_file = max(session_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load current session: {e}")
            return None
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent sessions with basic info."""
        session_files = sorted(
            self.sessions_dir.glob("session_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:limit]
        
        sessions = []
        for file_path in session_files:
            try:
                # Extract session ID from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    # Skip 'session' prefix and join the UUID parts
                    session_id = '-'.join(parts[1:-2]) + '-' + parts[-2]
                    
                    # Load summary only
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        summary = data.get('summary', {})
                        summary['file_path'] = str(file_path)
                        sessions.append(summary)
            except Exception as e:
                logger.error(f"Failed to read session file {file_path}: {e}")
        
        return sessions


def check_context_usage(tracker: SessionContextTracker, context_window_size: int = 200000) -> Dict[str, Any]:
    """Check if context usage is approaching limits and provide warnings."""
    usage_percent = tracker.get_context_usage_percentage(context_window_size)
    
    if usage_percent > 80:
        return {
            "warning": "HIGH_CONTEXT_USAGE",
            "message": f"Context window is {usage_percent:.0f}% full",
            "suggestion": "Consider clearing non-essential context or starting a new session",
            "usage_percent": usage_percent,
            "tokens_used": tracker.total_tokens_estimate,
            "tokens_remaining": context_window_size - tracker.total_tokens_estimate
        }
    elif usage_percent > 60:
        return {
            "warning": "MODERATE_CONTEXT_USAGE",
            "message": f"Context window is {usage_percent:.0f}% full",
            "suggestion": "Be mindful of large file reads and search results",
            "usage_percent": usage_percent,
            "tokens_used": tracker.total_tokens_estimate,
            "tokens_remaining": context_window_size - tracker.total_tokens_estimate
        }
    
    return {
        "status": "OK",
        "usage_percent": usage_percent,
        "tokens_used": tracker.total_tokens_estimate,
        "tokens_remaining": context_window_size - tracker.total_tokens_estimate
    }


def get_context_indicator(tracker: SessionContextTracker) -> str:
    """Get a compact context status indicator for inclusion in responses."""
    files = len(tracker.files_read)
    searches = len(tracker.searches_performed)
    tokens_k = tracker.total_tokens_estimate // 1000
    
    return f"[Context: {tokens_k}k/200k tokens | {files} files | {searches} searches]"


# Add MemoryComponent methods to SessionContextTracker
SessionContextTracker.get_memory_usage = lambda self: self._estimate_memory_usage()
SessionContextTracker.get_item_count = lambda self: len(self.context_events) + len(self.files_read) + len(self.searches_performed)

def _estimate_memory_usage(self) -> float:
    """Estimate memory usage in MB"""
    import sys
    
    # Rough estimation of memory usage
    memory_mb = 0.0
    
    # Files read
    for file_info in self.files_read.values():
        memory_mb += sys.getsizeof(file_info) / (1024**2)
    
    # Events
    for event in self.context_events:
        memory_mb += sys.getsizeof(event) / (1024**2)
    
    # Searches
    for search in self.searches_performed:
        memory_mb += sys.getsizeof(search) / (1024**2)
    
    return memory_mb

def _cleanup(self, aggressive: bool = False) -> int:
    """Perform cleanup and return number of items removed"""
    removed = 0
    
    if aggressive:
        # Remove oldest 50% of events
        if len(self.context_events) > 10:
            to_remove = len(self.context_events) // 2
            self.context_events = self.context_events[-to_remove:]
            removed += to_remove
        
        # Remove oldest 50% of searches
        if len(self.searches_performed) > 10:
            to_remove = len(self.searches_performed) // 2
            self.searches_performed = self.searches_performed[-to_remove:]
            removed += to_remove
    else:
        # Remove oldest 20% of events
        if len(self.context_events) > 50:
            to_remove = len(self.context_events) // 5
            self.context_events = self.context_events[-to_remove:]
            removed += to_remove
    
    self.mark_cleanup()
    return removed

# Bind methods to class
SessionContextTracker._estimate_memory_usage = _estimate_memory_usage
SessionContextTracker.cleanup = _cleanup