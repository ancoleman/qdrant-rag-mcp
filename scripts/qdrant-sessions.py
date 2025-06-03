#!/usr/bin/env python3
"""
Session viewer utility for Qdrant RAG MCP Server context tracking.

View and analyze context tracking sessions to understand what Claude knows.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from utils.context_tracking import SessionStore


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def format_duration(seconds: int) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    minutes = minutes % 60
    return f"{hours}h {minutes}m"


def list_sessions(store: SessionStore, limit: int = 10):
    """List recent sessions."""
    sessions = store.list_sessions(limit=limit)
    
    if not sessions:
        print("No sessions found.")
        return
    
    print(f"Recent Sessions (showing {len(sessions)} of {limit} max):\n")
    print(f"{'Session ID':<40} {'Started':<20} {'Duration':<10} {'Tokens':<10} {'Files':<6} {'Searches':<8}")
    print("-" * 100)
    
    for session in sessions:
        session_id = session.get('session_id', 'unknown')[:36]
        start_time = session.get('session_start', 'unknown')
        if start_time != 'unknown':
            try:
                # Parse ISO format timestamp
                dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                start_time = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        duration = format_duration(session.get('uptime_minutes', 0) * 60)
        tokens = session.get('total_tokens_estimate', 0)
        files = session.get('files_read', 0)
        searches = session.get('searches_performed', 0)
        
        print(f"{session_id:<40} {start_time:<20} {duration:<10} {tokens:<10} {files:<6} {searches:<8}")


def show_session_details(store: SessionStore, session_id: str):
    """Show detailed information about a specific session."""
    session = store.load_session(session_id)
    
    if not session:
        print(f"Session not found: {session_id}")
        return
    
    summary = session.get('summary', {})
    
    print(f"\n=== Session Details: {session_id} ===\n")
    
    # Basic info
    print("Basic Information:")
    print(f"  Started: {session.get('session_start', 'unknown')}")
    print(f"  Duration: {summary.get('uptime_minutes', 0)} minutes")
    print(f"  Project: {summary.get('current_project', {}).get('name', 'No project')}")
    
    # Token usage
    print(f"\nToken Usage:")
    print(f"  Total: {summary.get('total_tokens_estimate', 0):,} tokens")
    print(f"  Usage: {summary.get('context_usage_percentage', 0):.1f}% of context window")
    
    # Token breakdown
    breakdown = summary.get('token_usage_by_category', {})
    if breakdown:
        print("  Breakdown:")
        for category, tokens in breakdown.items():
            print(f"    {category}: {tokens:,} tokens")
    
    # Activity
    print(f"\nActivity Summary:")
    print(f"  Files read: {summary.get('files_read', 0)}")
    print(f"  Searches: {summary.get('searches_performed', 0)}")
    print(f"  Directories indexed: {summary.get('indexed_directories', 0)}")
    
    # Files
    files = session.get('files_read', {})
    if files:
        print(f"\nFiles in Context ({len(files)} total):")
        # Sort by tokens
        sorted_files = sorted(files.items(), key=lambda x: x[1].get('tokens_estimate', 0), reverse=True)
        for i, (path, info) in enumerate(sorted_files[:10]):
            tokens = info.get('tokens_estimate', 0)
            lines = info.get('lines', 0)
            print(f"  {i+1}. {path} ({tokens:,} tokens, {lines} lines)")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
    
    # Searches
    searches = session.get('searches_performed', [])
    if searches:
        print(f"\nSearch History ({len(searches)} total):")
        for i, search in enumerate(searches[-10:]):
            query = search.get('query', 'unknown')
            search_type = search.get('search_type', 'unknown')
            results = search.get('results_count', 0)
            print(f"  {i+1}. \"{query}\" ({search_type}, {results} results)")


def show_timeline(store: SessionStore, session_id: str, limit: int = 50):
    """Show timeline of events for a session."""
    session = store.load_session(session_id)
    
    if not session:
        print(f"Session not found: {session_id}")
        return
    
    events = session.get('context_events', [])
    
    if not events:
        print("No events recorded for this session.")
        return
    
    print(f"\n=== Session Timeline: {session_id} ===")
    print(f"Showing {min(len(events), limit)} of {len(events)} events:\n")
    
    print(f"{'Time':<20} {'Event Type':<15} {'Tokens':<10} {'Details'}")
    print("-" * 80)
    
    for event in events[-limit:]:
        timestamp = event.get('timestamp', 'unknown')
        if timestamp != 'unknown':
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%H:%M:%S')
            except:
                pass
        
        event_type = event.get('event_type', 'unknown')
        tokens = event.get('tokens_estimate', 0)
        details = event.get('details', {})
        
        # Format details based on event type
        detail_str = ""
        if event_type == 'file_read':
            detail_str = f"File: {details.get('file_path', 'unknown')}"
        elif event_type == 'search':
            detail_str = f"Query: \"{details.get('query', 'unknown')}\" ({details.get('results_count', 0)} results)"
        elif event_type == 'tool_use':
            detail_str = f"Tool: {details.get('tool_name', 'unknown')}"
        elif event_type == 'index_directory':
            detail_str = f"Dir: {details.get('directory', 'unknown')} ({details.get('file_count', 0)} files)"
        
        print(f"{timestamp:<20} {event_type:<15} {tokens:<10} {detail_str}")


def analyze_sessions(store: SessionStore):
    """Analyze all sessions for patterns and statistics."""
    sessions = store.list_sessions(limit=100)
    
    if not sessions:
        print("No sessions to analyze.")
        return
    
    print(f"\n=== Session Analysis ===")
    print(f"Analyzing {len(sessions)} sessions...\n")
    
    # Aggregate statistics
    total_tokens = sum(s.get('total_tokens_estimate', 0) for s in sessions)
    total_files = sum(s.get('files_read', 0) for s in sessions)
    total_searches = sum(s.get('searches_performed', 0) for s in sessions)
    total_duration = sum(s.get('uptime_minutes', 0) for s in sessions)
    
    # Averages
    avg_tokens = total_tokens // len(sessions) if sessions else 0
    avg_files = total_files // len(sessions) if sessions else 0
    avg_searches = total_searches // len(sessions) if sessions else 0
    avg_duration = total_duration // len(sessions) if sessions else 0
    
    print("Aggregate Statistics:")
    print(f"  Total sessions: {len(sessions)}")
    print(f"  Total duration: {format_duration(total_duration * 60)}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Total files read: {total_files}")
    print(f"  Total searches: {total_searches}")
    
    print("\nAverage per Session:")
    print(f"  Duration: {avg_duration} minutes")
    print(f"  Tokens: {avg_tokens:,}")
    print(f"  Files: {avg_files}")
    print(f"  Searches: {avg_searches}")
    
    # Find extremes
    if sessions:
        longest = max(sessions, key=lambda s: s.get('uptime_minutes', 0))
        most_tokens = max(sessions, key=lambda s: s.get('total_tokens_estimate', 0))
        most_files = max(sessions, key=lambda s: s.get('files_read', 0))
        most_searches = max(sessions, key=lambda s: s.get('searches_performed', 0))
        
        print("\nNotable Sessions:")
        print(f"  Longest: {longest.get('session_id', 'unknown')[:36]} ({longest.get('uptime_minutes', 0)} minutes)")
        print(f"  Most tokens: {most_tokens.get('session_id', 'unknown')[:36]} ({most_tokens.get('total_tokens_estimate', 0):,} tokens)")
        print(f"  Most files: {most_files.get('session_id', 'unknown')[:36]} ({most_files.get('files_read', 0)} files)")
        print(f"  Most searches: {most_searches.get('session_id', 'unknown')[:36]} ({most_searches.get('searches_performed', 0)} searches)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="View and analyze Qdrant RAG context tracking sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # List recent sessions
  %(prog)s -s SESSION_ID      # Show session details
  %(prog)s -t SESSION_ID      # Show session timeline
  %(prog)s --analyze          # Analyze all sessions
  %(prog)s --limit 20         # List 20 most recent sessions
"""
    )
    
    parser.add_argument("-s", "--session", help="Show details for specific session ID")
    parser.add_argument("-t", "--timeline", help="Show timeline for specific session ID")
    parser.add_argument("--analyze", action="store_true", help="Analyze all sessions")
    parser.add_argument("--limit", type=int, default=10, help="Number of sessions to list (default: 10)")
    parser.add_argument("--session-dir", help="Custom session directory")
    
    args = parser.parse_args()
    
    # Determine session directory
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = Path.home() / ".mcp-servers" / "qdrant-rag"
    
    # Create session store
    store = SessionStore(session_dir)
    
    # Execute requested action
    if args.session:
        show_session_details(store, args.session)
    elif args.timeline:
        show_timeline(store, args.timeline, limit=args.limit)
    elif args.analyze:
        analyze_sessions(store)
    else:
        list_sessions(store, limit=args.limit)


if __name__ == "__main__":
    main()