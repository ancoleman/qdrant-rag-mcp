#!/usr/bin/env python3
"""
Qdrant RAG MCP Server Log Viewer

A utility to view and search logs from the project-aware logging system.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Optional, List, Dict
import hashlib


def get_project_hash(project_path: str) -> str:
    """Generate a hash for the project path."""
    return hashlib.md5(project_path.encode()).hexdigest()[:12]


def find_project_by_path(base_dir: Path, project_path: str) -> Optional[Path]:
    """Find project log directory by path."""
    projects_dir = base_dir / "projects"
    if not projects_dir.exists():
        return None
    
    # Try to find by matching metadata
    for project_dir in projects_dir.iterdir():
        if project_dir.is_dir():
            metadata_file = project_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    if metadata.get("project_path") == project_path:
                        return project_dir
                except:
                    pass
    
    # Fallback to old hash-based method
    project_hash = get_project_hash(project_path)
    old_project_dir = projects_dir / f"project_{project_hash}"
    if old_project_dir.exists():
        return old_project_dir
    
    return None


def list_projects(base_dir: Path) -> List[Dict[str, str]]:
    """List all projects with logs."""
    projects = []
    projects_dir = base_dir / "projects"
    
    if not projects_dir.exists():
        return projects
    
    for project_dir in projects_dir.iterdir():
        if project_dir.is_dir():
            metadata_file = project_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text())
                    projects.append({
                        "name": metadata.get("project_name", "Unknown"),
                        "path": metadata.get("project_path", "Unknown"),
                        "hash": metadata.get("hash", ""),
                        "dir_name": project_dir.name,
                        "created": metadata.get("created_at", "Unknown")
                    })
                except:
                    pass
    
    return sorted(projects, key=lambda x: x["name"])


def read_logs(log_file: Path, filters: Dict) -> List[Dict]:
    """Read and filter logs from a file."""
    logs = []
    
    if not log_file.exists():
        return logs
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                
                # Apply filters
                if filters.get("level") and log.get("level") != filters["level"]:
                    continue
                
                if filters.get("operation") and log.get("operation") != filters["operation"]:
                    continue
                
                if filters.get("search"):
                    pattern = filters["search"]
                    if not any(re.search(pattern, str(v), re.IGNORECASE) for v in log.values()):
                        continue
                
                if filters.get("from_date"):
                    log_date = datetime.fromisoformat(log["timestamp"])
                    if log_date < filters["from_date"]:
                        continue
                
                if filters.get("to_date"):
                    log_date = datetime.fromisoformat(log["timestamp"])
                    if log_date > filters["to_date"]:
                        continue
                
                logs.append(log)
                
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
            except Exception as e:
                print(f"Error parsing log line: {e}", file=sys.stderr)
                continue
    
    return logs


def format_log(log: Dict, format_type: str = "pretty") -> str:
    """Format a log entry for display."""
    if format_type == "json":
        return json.dumps(log)
    
    # Pretty format
    timestamp = log.get("timestamp", "Unknown")
    level = log.get("level", "INFO")
    message = log.get("message", "")
    
    # Color codes for different levels
    color_codes = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m"  # Magenta
    }
    reset_code = "\033[0m"
    
    color = color_codes.get(level, "")
    
    output = f"{timestamp} {color}[{level}]{reset_code} {message}"
    
    # Add extra fields
    extras = []
    for key, value in log.items():
        if key not in ["timestamp", "level", "message", "logger", "module", "function", "line"]:
            extras.append(f"{key}={value}")
    
    if extras:
        output += f" | {' '.join(extras)}"
    
    return output


def follow_logs(log_files: List[Path], filters: Dict, format_type: str = "pretty"):
    """Follow logs in real-time."""
    import time
    
    # Track file positions
    file_positions = {str(f): f.stat().st_size if f.exists() else 0 for f in log_files}
    seen_lines = set()
    
    print("Following logs... (Ctrl+C to stop)")
    
    try:
        while True:
            for log_file in log_files:
                if not log_file.exists():
                    continue
                
                current_size = log_file.stat().st_size
                last_position = file_positions.get(str(log_file), 0)
                
                if current_size > last_position:
                    with open(log_file, 'r') as f:
                        f.seek(last_position)
                        for line in f:
                            line = line.strip()
                            if line and line not in seen_lines:
                                seen_lines.add(line)
                                try:
                                    log = json.loads(line)
                                    
                                    # Apply filters
                                    skip = False
                                    if filters.get("level") and log.get("level") != filters["level"]:
                                        skip = True
                                    if filters.get("operation") and log.get("operation") != filters["operation"]:
                                        skip = True
                                    if filters.get("search"):
                                        pattern = filters["search"]
                                        if not any(re.search(pattern, str(v), re.IGNORECASE) for v in log.values()):
                                            skip = True
                                    
                                    if not skip:
                                        print(format_log(log, format_type))
                                        
                                except json.JSONDecodeError:
                                    pass
                        
                        file_positions[str(log_file)] = f.tell()
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopped following logs.")


def main():
    parser = argparse.ArgumentParser(description="View and search Qdrant RAG MCP Server logs")
    
    # Log location
    parser.add_argument("--log-dir", type=str, 
                       default=str(Path.home() / ".mcp-servers" / "qdrant-rag" / "logs"),
                       help="Base log directory")
    
    # Project selection
    parser.add_argument("--project", type=str, help="Project path to view logs for")
    parser.add_argument("--list-projects", action="store_true", help="List all projects with logs")
    parser.add_argument("--global", dest="global_logs", action="store_true", 
                       help="View global logs (non-project specific)")
    parser.add_argument("--errors", action="store_true", help="View error logs only")
    
    # Filters
    parser.add_argument("--level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Filter by log level")
    parser.add_argument("--operation", type=str, help="Filter by operation name")
    parser.add_argument("--search", type=str, help="Search logs (regex supported)")
    parser.add_argument("--from", dest="from_date", type=str, 
                       help="From date (YYYY-MM-DD or 'yesterday', 'today')")
    parser.add_argument("--to", dest="to_date", type=str, 
                       help="To date (YYYY-MM-DD or 'yesterday', 'today')")
    
    # Output options
    parser.add_argument("--tail", type=int, metavar="N", help="Show last N logs")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs in real-time")
    parser.add_argument("--export", choices=["json", "pretty"], default="pretty",
                       help="Export format")
    
    args = parser.parse_args()
    
    base_dir = Path(args.log_dir)
    if not base_dir.exists():
        print(f"Log directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)
    
    # List projects
    if args.list_projects:
        projects = list_projects(base_dir)
        if not projects:
            print("No projects found with logs.")
        else:
            print("Projects with logs:")
            for proj in projects:
                print(f"  - {proj['name']} ({proj['path']})")
        return
    
    # Determine which logs to read
    log_files = []
    
    if args.errors:
        # Error logs
        errors_dir = base_dir / "errors"
        if errors_dir.exists():
            log_files.extend(sorted(errors_dir.glob("*.log"), reverse=True))
    elif args.global_logs:
        # Global logs
        global_dir = base_dir / "global"
        if global_dir.exists():
            log_files.extend(sorted(global_dir.glob("*.log"), reverse=True))
    elif args.project:
        # Project-specific logs
        project_dir = find_project_by_path(base_dir, args.project)
        if not project_dir:
            print(f"No logs found for project: {args.project}", file=sys.stderr)
            sys.exit(1)
        log_files.extend(sorted(project_dir.glob("*.log"), reverse=True))
    else:
        # Default to current directory as project
        current_dir = Path.cwd()
        project_dir = find_project_by_path(base_dir, str(current_dir))
        if project_dir:
            log_files.extend(sorted(project_dir.glob("*.log"), reverse=True))
        else:
            # Fall back to global logs
            global_dir = base_dir / "global"
            if global_dir.exists():
                log_files.extend(sorted(global_dir.glob("*.log"), reverse=True))
    
    if not log_files:
        print("No log files found.", file=sys.stderr)
        sys.exit(1)
    
    # Parse date filters
    filters = {}
    if args.level:
        filters["level"] = args.level
    if args.operation:
        filters["operation"] = args.operation
    if args.search:
        filters["search"] = args.search
    
    if args.from_date:
        if args.from_date == "today":
            filters["from_date"] = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif args.from_date == "yesterday":
            filters["from_date"] = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            filters["from_date"] = datetime.strptime(args.from_date, "%Y-%m-%d")
    
    if args.to_date:
        if args.to_date == "today":
            filters["to_date"] = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        elif args.to_date == "yesterday":
            filters["to_date"] = (datetime.now() - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            filters["to_date"] = datetime.strptime(args.to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    # Follow mode
    if args.follow:
        follow_logs(log_files, filters, args.export)
        return
    
    # Read and display logs
    all_logs = []
    for log_file in log_files:
        logs = read_logs(log_file, filters)
        all_logs.extend(logs)
    
    # Sort by timestamp
    all_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Apply tail limit
    if args.tail:
        all_logs = all_logs[:args.tail]
    
    # Display logs
    for log in reversed(all_logs):  # Show oldest first
        print(format_log(log, args.export))


if __name__ == "__main__":
    main()