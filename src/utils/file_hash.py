"""File hashing utilities for change detection."""

import hashlib
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Calculate hash of file content.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use (default: sha256)
        
    Returns:
        Hash string in format "algorithm:hexdigest"
    """
    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return f"{algorithm}:{hash_obj.hexdigest()}"
    
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        raise


def get_file_info(file_path: str) -> dict:
    """
    Get file information including hash and metadata.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = path.stat()
    
    return {
        "file_path": str(path.absolute()),
        "file_hash": calculate_file_hash(file_path),
        "file_size": stat.st_size,
        "file_mtime": stat.st_mtime,
        "exists": True
    }


def has_file_changed(
    file_path: str, 
    stored_hash: Optional[str] = None,
    stored_mtime: Optional[float] = None
) -> bool:
    """
    Check if file has changed compared to stored metadata.
    
    Args:
        file_path: Path to the file
        stored_hash: Previously stored hash
        stored_mtime: Previously stored modification time
        
    Returns:
        True if file has changed, False otherwise
    """
    try:
        current_info = get_file_info(file_path)
        
        # If we have a hash, use it (most reliable)
        if stored_hash:
            return current_info["file_hash"] != stored_hash
        
        # Fall back to mtime if no hash
        if stored_mtime:
            return current_info["file_mtime"] != stored_mtime
        
        # If no stored info, consider it changed
        return True
        
    except FileNotFoundError:
        # File deleted
        return True
    except Exception as e:
        logger.error(f"Error checking file change for {file_path}: {e}")
        # Assume changed on error to be safe
        return True