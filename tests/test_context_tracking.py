#!/usr/bin/env python3
"""
Test context tracking functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.context_tracking import SessionContextTracker, SessionStore, check_context_usage
import tempfile
import json


def test_session_tracker():
    """Test basic session tracker functionality."""
    print("Testing SessionContextTracker...")
    
    # Create tracker
    tracker = SessionContextTracker()
    print(f"✓ Created tracker with session ID: {tracker.session_id}")
    
    # Test token estimation
    test_text = "This is a test string for token estimation."
    tokens = tracker.estimate_tokens(test_text)
    expected = len(test_text) // 4
    assert tokens == expected, f"Expected {expected} tokens, got {tokens}"
    print(f"✓ Token estimation: '{test_text}' = {tokens} tokens")
    
    # Test file tracking
    tracker.track_file_read(
        file_path="/test/file.py",
        content="def hello():\n    print('Hello world')\n" * 10,
        metadata={"language": "python"}
    )
    assert len(tracker.files_read) == 1
    assert tracker.total_tokens_estimate > 0
    print(f"✓ Tracked file read, total tokens: {tracker.total_tokens_estimate}")
    
    # Test search tracking
    tracker.track_search(
        query="test search",
        results=[{"content": "result 1"}, {"content": "result 2"}],
        search_type="code"
    )
    assert len(tracker.searches_performed) == 1
    print(f"✓ Tracked search, total searches: {len(tracker.searches_performed)}")
    
    # Test context usage
    usage_percent = tracker.get_context_usage_percentage()
    assert 0 <= usage_percent <= 100
    print(f"✓ Context usage: {usage_percent:.2f}%")
    
    # Test session summary
    summary = tracker.get_session_summary()
    assert summary["files_read"] == 1
    assert summary["searches_performed"] == 1
    assert summary["session_id"] == tracker.session_id
    print("✓ Session summary generated")
    
    return tracker


def test_session_store():
    """Test session persistence."""
    print("\nTesting SessionStore...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SessionStore(Path(tmpdir))
        print(f"✓ Created session store at: {tmpdir}")
        
        # Create and save a session
        tracker = SessionContextTracker()
        tracker.track_file_read("/test.py", "test content")
        
        store.save_session(tracker)
        print(f"✓ Saved session: {tracker.session_id}")
        
        # Load the session
        loaded = store.load_session(tracker.session_id)
        assert loaded is not None
        assert loaded["session_id"] == tracker.session_id
        assert len(loaded["files_read"]) == 1
        print("✓ Loaded session successfully")
        
        # Test listing sessions
        sessions = store.list_sessions()
        assert len(sessions) >= 1
        print(f"✓ Listed {len(sessions)} sessions")


def test_context_warnings():
    """Test context usage warnings."""
    print("\nTesting context warnings...")
    
    tracker = SessionContextTracker()
    
    # Test OK status
    status = check_context_usage(tracker)
    assert status["status"] == "OK"
    print("✓ Low usage returns OK status")
    
    # Simulate moderate usage (65%)
    tracker.total_tokens_estimate = 32500
    status = check_context_usage(tracker)
    assert status["warning"] == "MODERATE_CONTEXT_USAGE"
    print(f"✓ Moderate usage warning at {status['usage_percent']:.0f}%")
    
    # Simulate high usage (85%)
    tracker.total_tokens_estimate = 42500
    status = check_context_usage(tracker)
    assert status["warning"] == "HIGH_CONTEXT_USAGE"
    print(f"✓ High usage warning at {status['usage_percent']:.0f}%")


def test_timeline_events():
    """Test context event timeline."""
    print("\nTesting timeline events...")
    
    tracker = SessionContextTracker()
    
    # Add various events
    tracker.track_file_read("/file1.py", "content1")
    tracker.track_search("search query", [{"result": 1}])
    tracker.track_tool_use("index_code", {"file": "/file2.py"}, 1000)
    
    # Check timeline
    assert len(tracker.context_events) == 3
    
    # Verify event types
    event_types = [e.event_type for e in tracker.context_events]
    assert "file_read" in event_types
    assert "search" in event_types
    assert "tool_use" in event_types
    
    print(f"✓ Timeline has {len(tracker.context_events)} events")
    print(f"✓ Event types: {', '.join(event_types)}")


def main():
    """Run all tests."""
    print("=== Context Tracking Tests ===\n")
    
    try:
        test_session_tracker()
        test_session_store()
        test_context_warnings()
        test_timeline_events()
        
        print("\n✅ All tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()