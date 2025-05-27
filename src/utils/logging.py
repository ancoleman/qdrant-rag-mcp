"""
Project-aware logging system for Qdrant RAG MCP Server.

This module provides logging that separates logs by project context,
ensuring logs from different projects don't mix when the server is
used globally.
"""

import json
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import os


class ProjectAwareLogger:
    """Logger that maintains separate log files per project."""
    
    def __init__(self, base_log_dir: Optional[Path] = None):
        """Initialize the project-aware logging system."""
        if base_log_dir is None:
            base_log_dir = Path.home() / ".mcp-servers" / "qdrant-rag" / "logs"
        
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.global_log_dir = self.base_log_dir / "global"
        self.projects_log_dir = self.base_log_dir / "projects"
        self.errors_log_dir = self.base_log_dir / "errors"
        
        for dir_path in [self.global_log_dir, self.projects_log_dir, self.errors_log_dir]:
            dir_path.mkdir(exist_ok=True)
            # Set restrictive permissions
            os.chmod(dir_path, 0o700)
        
        self._loggers = {}
        self._handlers = {}
        self._formatter = JsonFormatter()
        
        # Initialize global and error loggers
        self._init_global_logger()
        self._init_error_logger()
    
    def _get_project_hash(self, project_path: str) -> str:
        """Generate a hash for the project path."""
        return hashlib.md5(project_path.encode()).hexdigest()[:12]
    
    def _get_project_log_dir_name(self, project_name: str, project_path: str) -> str:
        """Generate a user-friendly project log directory name."""
        # Sanitize project name for filesystem
        safe_name = project_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
        
        # Add short hash to handle duplicate names
        short_hash = self._get_project_hash(project_path)[:6]
        
        return f"{safe_name}_{short_hash}"
    
    def _init_global_logger(self):
        """Initialize the global logger for non-project operations."""
        logger = logging.getLogger("qdrant_rag.global")
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add rotating file handler
        handler = TimedRotatingFileHandler(
            self.global_log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log",
            when='midnight',
            interval=1,
            backupCount=7,
            encoding='utf-8'
        )
        handler.setFormatter(self._formatter)
        logger.addHandler(handler)
        
        self._loggers['global'] = logger
    
    def _init_error_logger(self):
        """Initialize the error logger for critical errors."""
        logger = logging.getLogger("qdrant_rag.errors")
        logger.setLevel(logging.ERROR)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add rotating file handler
        handler = TimedRotatingFileHandler(
            self.errors_log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log",
            when='midnight',
            interval=1,
            backupCount=30,  # Keep error logs longer
            encoding='utf-8'
        )
        handler.setFormatter(self._formatter)
        logger.addHandler(handler)
        
        self._loggers['errors'] = logger
    
    def get_logger(self, project_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
        """Get a logger for the given project context."""
        if not project_context:
            return self._loggers['global']
        
        project_path = project_context.get('path', '')
        project_name = project_context.get('name', 'unknown')
        
        if not project_path:
            return self._loggers['global']
        
        # Check if we already have a logger for this project
        project_hash = self._get_project_hash(project_path)
        logger_key = f"project_{project_hash}"
        
        if logger_key in self._loggers:
            return self._loggers[logger_key]
        
        # Create new project logger
        return self._create_project_logger(project_hash, project_name, project_path)
    
    def _create_project_logger(self, project_hash: str, project_name: str, project_path: str) -> logging.Logger:
        """Create a new logger for a specific project."""
        # Use friendly directory name
        friendly_dir_name = self._get_project_log_dir_name(project_name, project_path)
        project_log_dir = self.projects_log_dir / friendly_dir_name
        project_log_dir.mkdir(exist_ok=True)
        os.chmod(project_log_dir, 0o700)
        
        # Save project metadata
        metadata_file = project_log_dir / "metadata.json"
        metadata = {
            "project_name": project_name,
            "project_path": project_path,
            "created_at": datetime.now().isoformat(),
            "hash": project_hash
        }
        metadata_file.write_text(json.dumps(metadata, indent=2))
        os.chmod(metadata_file, 0o600)
        
        # Create logger
        logger_name = f"qdrant_rag.projects.{project_hash}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # Add rotating file handler
        handler = RotatingFileHandler(
            project_log_dir / f"{datetime.now().strftime('%Y-%m-%d')}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        handler.setFormatter(self._formatter)
        logger.addHandler(handler)
        
        # Store logger
        logger_key = f"project_{project_hash}"
        self._loggers[logger_key] = logger
        
        # Log initialization
        logger.info(
            "Initialized project logger",
            extra={
                "project_name": project_name,
                "project_path": project_path,
                "project_hash": project_hash
            }
        )
        
        return logger
    
    def get_error_logger(self) -> logging.Logger:
        """Get the global error logger."""
        return self._loggers['errors']


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add any extra fields
        if hasattr(record, 'project_name'):
            log_data['project_name'] = record.project_name
        if hasattr(record, 'project_path'):
            log_data['project_path'] = record.project_path
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        # Add any additional metadata
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                         'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                         'pathname', 'process', 'processName', 'relativeCreated', 
                         'thread', 'threadName', 'getMessage', 'project_name', 
                         'project_path', 'operation', 'duration_ms']:
                log_data[key] = value
        
        # Handle exceptions
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


# Singleton instance
_logger_instance: Optional[ProjectAwareLogger] = None


def get_project_logger(project_context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """Get a logger for the current project context."""
    global _logger_instance
    
    if _logger_instance is None:
        # Initialize with environment variable override if set
        log_dir = os.environ.get('QDRANT_LOG_DIR')
        _logger_instance = ProjectAwareLogger(Path(log_dir) if log_dir else None)
    
    return _logger_instance.get_logger(project_context)


def get_error_logger() -> logging.Logger:
    """Get the global error logger."""
    global _logger_instance
    
    if _logger_instance is None:
        log_dir = os.environ.get('QDRANT_LOG_DIR')
        _logger_instance = ProjectAwareLogger(Path(log_dir) if log_dir else None)
    
    return _logger_instance.get_error_logger()


# Convenience function for operation logging
def log_operation(operation_name: str):
    """Decorator to automatically log operations with timing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger = get_project_logger()
            
            # Log start
            logger.info(
                f"Starting {operation_name}",
                extra={"operation": operation_name}
            )
            
            try:
                result = func(*args, **kwargs)
                
                # Log success
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.info(
                    f"Completed {operation_name}",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "success"
                    }
                )
                
                return result
                
            except Exception as e:
                # Log failure
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000
                logger.error(
                    f"Failed {operation_name}: {str(e)}",
                    extra={
                        "operation": operation_name,
                        "duration_ms": duration_ms,
                        "status": "error",
                        "error_type": type(e).__name__
                    },
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator