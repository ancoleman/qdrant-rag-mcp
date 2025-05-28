"""
Dependency resolver for search results

This module rebuilds dependency relationships from stored metadata
to enable dependency-aware search.
"""

from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DependencyResolver:
    """Resolves dependencies from search results metadata"""
    
    def __init__(self, qdrant_client, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self._file_dependencies: Dict[str, Dict[str, Any]] = {}
        self._import_map: Dict[str, Set[str]] = {}  # module -> files that export it
        
    def load_dependencies_from_collection(self):
        """Load all dependency metadata from the collection"""
        try:
            # Scroll through all documents to build dependency map
            # Focus on first chunks which contain dependency metadata
            scroll_filter = {
                "must": [
                    {"key": "chunk_index", "match": {"value": 0}}
                ]
            }
            
            offset = None
            while True:
                results = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=scroll_filter,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = results
                
                if not points:
                    break
                    
                for point in points:
                    payload = point.payload
                    file_path = payload.get("file_path", "")
                    
                    # Extract dependency info if present
                    deps = payload.get("dependencies", {})
                    if deps:
                        self._file_dependencies[file_path] = deps
                        
                        # Build import map
                        exports = deps.get("exports", [])
                        if exports:
                            # Extract module name from file path
                            module_name = self._file_to_module(file_path)
                            if module_name:
                                if module_name not in self._import_map:
                                    self._import_map[module_name] = set()
                                self._import_map[module_name].add(file_path)
                                
                if offset is None:
                    break
                    
            logger.info(f"Loaded dependencies for {len(self._file_dependencies)} files")
            
        except Exception as e:
            logger.error(f"Failed to load dependencies: {e}")
            
    def _file_to_module(self, file_path: str) -> Optional[str]:
        """Convert file path to module name"""
        path = Path(file_path)
        
        # Remove extension
        if path.suffix == '.py':
            if path.stem == '__init__':
                # Package init
                parts = list(path.parent.parts)
            else:
                # Regular module
                parts = list(path.parent.parts) + [path.stem]
                
            # Find likely module root
            for i, part in enumerate(parts):
                if part in ['src', 'lib', 'app'] and i < len(parts) - 1:
                    parts = parts[i+1:]
                    break
                    
            return '.'.join(parts)
        return None
        
    def find_dependencies_for_files(self, file_paths: Set[str]) -> Set[str]:
        """Find all files that import or are imported by the given files"""
        dependent_files = set()
        
        for file_path in file_paths:
            # Get files that this file imports
            deps = self._file_dependencies.get(file_path, {})
            imports = deps.get("imports", [])
            
            for imp in imports:
                module = imp.get("module", "")
                if module:
                    # Check if we know which files export this module
                    if module in self._import_map:
                        dependent_files.update(self._import_map[module])
                    
                    # Also check for relative imports
                    if module.startswith('.'):
                        # Handle relative imports
                        base_path = Path(file_path).parent
                        level = len(module) - len(module.lstrip('.'))
                        
                        for _ in range(level - 1):
                            base_path = base_path.parent
                            
                        if module.strip('.'):
                            potential_file = base_path / f"{module.strip('.').replace('.', '/')}.py"
                            if str(potential_file) in self._file_dependencies:
                                dependent_files.add(str(potential_file))
                                
            # Get files that import this file
            module_name = self._file_to_module(file_path)
            if module_name:
                # Check all files for imports of this module
                for other_file, other_deps in self._file_dependencies.items():
                    if other_file == file_path:
                        continue
                        
                    other_imports = other_deps.get("imports", [])
                    for imp in other_imports:
                        if imp.get("module", "") == module_name:
                            dependent_files.add(other_file)
                            
        # Remove the original files
        dependent_files -= file_paths
        
        return dependent_files