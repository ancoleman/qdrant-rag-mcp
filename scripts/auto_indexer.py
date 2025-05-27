#!/usr/bin/env python3
"""
[OPTIONAL] Standalone file watcher and indexer for Qdrant RAG

NOTE: The MCP server (qdrant_mcp_context_aware.py) has built-in file watching.
This standalone script is provided for advanced use cases where you want to run
the file watcher separately from the MCP server.

For most users, enable auto-indexing via environment variable instead:
  export QDRANT_RAG_AUTO_INDEX=true

Features:
- Watches for file changes
- Debouncing to avoid excessive reindexing
- Smart filtering (only reindex relevant files)
- Incremental updates (only reindex changed files)
"""

import os
import sys
import time
import json
import requests
import argparse
from pathlib import Path
from typing import Set
import logging

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌ watchdog not installed. Install with: pip install watchdog")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RagIndexHandler(FileSystemEventHandler):
    """Handles file system events and triggers reindexing"""
    
    def __init__(self, http_api_url="http://localhost:8081", debounce_seconds=3):
        self.http_api_url = http_api_url
        self.debounce_seconds = debounce_seconds
        self.pending_files: Set[str] = set()
        self.last_index_time = 0
        self.file_patterns = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
            '.json', '.yaml', '.yml', '.xml', '.toml', '.ini',
            '.md', '.sh', '.dockerfile', '.gitignore', '.dockerignore'
        }
        self.exclude_dirs = {
            '.git', '__pycache__', 'node_modules', '.venv', 'venv',
            'data', 'logs', 'dist', 'build', '.pytest_cache', '.cache'
        }
        
    def should_index_file(self, path: str) -> bool:
        """Check if file should be indexed"""
        path_obj = Path(path)
        
        # Check if in excluded directory
        for parent in path_obj.parents:
            if parent.name in self.exclude_dirs:
                return False
                
        # Check file extension
        return path_obj.suffix.lower() in self.file_patterns
        
    def on_modified(self, event):
        if not event.is_directory and self.should_index_file(event.src_path):
            logger.info(f"File modified: {event.src_path}")
            self.pending_files.add(event.src_path)
            self.schedule_reindex()
            
    def on_created(self, event):
        if not event.is_directory and self.should_index_file(event.src_path):
            logger.info(f"File created: {event.src_path}")
            self.pending_files.add(event.src_path)
            self.schedule_reindex()
            
    def on_deleted(self, event):
        if not event.is_directory and self.should_index_file(event.src_path):
            logger.info(f"File deleted: {event.src_path}")
            # TODO: Implement deletion from index
            self.schedule_reindex()
            
    def schedule_reindex(self):
        """Schedule a reindex with debouncing"""
        current_time = time.time()
        time_since_last = current_time - self.last_index_time
        
        if time_since_last >= self.debounce_seconds:
            self.perform_reindex()
        
    def perform_reindex(self):
        """Perform the actual reindexing"""
        if not self.pending_files and time.time() - self.last_index_time < 60:
            return
            
        logger.info(f"🔄 Reindexing {len(self.pending_files)} files...")
        
        try:
            if self.pending_files:
                # Index specific files
                success = 0
                for file_path in self.pending_files:
                    if self.index_file(file_path):
                        success += 1
                        
                logger.info(f"✅ Indexed {success}/{len(self.pending_files)} files")
            else:
                # Full directory reindex
                response = requests.post(
                    f"{self.http_api_url}/index_directory",
                    json={"directory": "."},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Indexed {result.get('total', 0)} files")
                else:
                    logger.error(f"❌ Reindex failed: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Reindex error: {e}")
        finally:
            self.pending_files.clear()
            self.last_index_time = time.time()
            
    def index_file(self, file_path: str) -> bool:
        """Index a single file"""
        try:
            # Determine endpoint based on file type
            ext = Path(file_path).suffix.lower()
            if ext in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini']:
                endpoint = "index_config"
            else:
                endpoint = "index_code"
                
            response = requests.post(
                f"{self.http_api_url}/{endpoint}",
                json={"file_path": file_path},
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Auto-index files for Qdrant RAG")
    parser.add_argument("path", nargs="?", default=".", help="Path to watch")
    parser.add_argument("--api-url", default="http://localhost:8081", help="HTTP API URL")
    parser.add_argument("--debounce", type=int, default=3, help="Debounce seconds")
    parser.add_argument("--initial-index", action="store_true", help="Perform initial index")
    
    args = parser.parse_args()
    
    # Check if server is running
    try:
        response = requests.get(f"{args.api_url}/health", timeout=2)
        if response.status_code != 200:
            logger.error("❌ RAG server not responding")
            sys.exit(1)
    except:
        logger.error("❌ Cannot connect to RAG server")
        logger.info("Start the HTTP server with: python src/http_server.py")
        sys.exit(1)
        
    # Create event handler and observer
    event_handler = RagIndexHandler(args.api_url, args.debounce)
    observer = Observer()
    observer.schedule(event_handler, args.path, recursive=True)
    
    # Initial index if requested
    if args.initial_index:
        logger.info("📦 Performing initial index...")
        event_handler.perform_reindex()
    
    # Start watching
    observer.start()
    logger.info(f"👀 Watching {args.path} for changes...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
            # Check for pending reindex
            if event_handler.pending_files:
                current_time = time.time()
                time_since_last = current_time - event_handler.last_index_time
                if time_since_last >= event_handler.debounce_seconds:
                    event_handler.perform_reindex()
    except KeyboardInterrupt:
        observer.stop()
        logger.info("🛑 Stopping file watcher")
        
    observer.join()

if __name__ == "__main__":
    main()