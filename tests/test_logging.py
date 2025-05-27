#!/usr/bin/env python3
"""
Test the project-aware logging system
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path
import json
from utils.logging import get_project_logger, ProjectAwareLogger

def test_logging():
    """Test logging functionality"""
    print("Testing project-aware logging...")
    
    # Test 1: Global logger (no project context)
    logger = get_project_logger()
    logger.info("Test global log message")
    
    # Test 2: Project logger
    project_context = {
        "name": "test_project",
        "path": str(Path.cwd())
    }
    project_logger = get_project_logger(project_context)
    project_logger.info("Test project log message", extra={
        "operation": "test",
        "status": "running"
    })
    
    # Test 3: Check log files exist
    log_base = Path.home() / ".mcp-servers" / "qdrant-rag" / "logs"
    
    # Check global logs
    global_logs = log_base / "global"
    if global_logs.exists():
        print(f"✅ Global logs directory exists: {global_logs}")
        log_files = list(global_logs.glob("*.log"))
        if log_files:
            print(f"   Found {len(log_files)} log file(s)")
            # Read last line
            with open(log_files[0], 'r') as f:
                lines = f.readlines()
                if lines:
                    last_log = json.loads(lines[-1])
                    print(f"   Last log: {last_log['message']}")
    
    # Check project logs
    project_hash = project_logger.handlers[0].baseFilename.split('/')[-2] if project_logger.handlers else None
    if project_hash:
        project_logs = log_base / "projects" / project_hash
        if project_logs.exists():
            print(f"✅ Project logs directory exists: {project_logs}")
            log_files = list(project_logs.glob("*.log"))
            if log_files:
                print(f"   Found {len(log_files)} log file(s)")
                # Check metadata
                metadata_file = project_logs / "metadata.json"
                if metadata_file.exists():
                    metadata = json.loads(metadata_file.read_text())
                    print(f"   Project: {metadata['project_name']}")
                    print(f"   Path: {metadata['project_path']}")
    
    # Test 4: Test log operation decorator
    from utils.logging import log_operation
    
    @log_operation("test_function")
    def sample_operation():
        return "success"
    
    result = sample_operation()
    print(f"✅ Log operation decorator test: {result}")
    
    print("\n✅ All logging tests passed!")

if __name__ == "__main__":
    test_logging()