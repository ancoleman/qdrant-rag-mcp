"""
Dependency Graph Builder for Code Intelligence

This module extracts and manages code dependencies from AST-parsed chunks,
enabling dependency-aware search and retrieval.
"""

import re
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from utils.logging import get_project_logger

logger = get_project_logger()


@dataclass
class ImportInfo:
    """Information about an import statement"""
    module: str  # The module being imported (e.g., 'os', 'package.module')
    names: List[str] = field(default_factory=list)  # Specific names imported
    is_relative: bool = False  # Whether it's a relative import
    level: int = 0  # Relative import level (number of dots)
    line_number: int = 0
    
    
@dataclass
class FileDependencies:
    """Dependencies for a single file"""
    file_path: str
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # Exported names (classes, functions)
    imported_by: Set[str] = field(default_factory=set)  # Files that import this file
    

class DependencyGraphBuilder:
    """Builds and manages dependency relationships between code files"""
    
    def __init__(self):
        self.dependencies: Dict[str, FileDependencies] = {}
        self._module_to_files: Dict[str, Set[str]] = {}  # Map module names to file paths
        
    def extract_dependencies_from_chunks(self, chunks: List[Dict[str, Any]], file_path: str) -> FileDependencies:
        """Extract dependency information from AST chunks"""
        file_deps = FileDependencies(file_path=file_path)
        
        for chunk in chunks:
            chunk_type = chunk.get('chunk_type', '')
            
            # Extract imports from import chunks
            if chunk_type == 'imports':
                imports = self._extract_imports_from_chunk(chunk)
                file_deps.imports.extend(imports)
                
            # Extract exports (public API) from other chunks
            elif chunk_type in ['class', 'function', 'class_definition']:
                name = chunk.get('name', '')
                if name and not name.startswith('_'):  # Public names only
                    file_deps.exports.append(name)
                    
        # Store in our graph
        self.dependencies[file_path] = file_deps
        
        # Update module mapping
        self._update_module_mapping(file_path, file_deps)
        
        logger.info(f"Extracted dependencies from {file_path}", extra={
            "operation": "extract_dependencies",
            "file_path": file_path,
            "import_count": len(file_deps.imports),
            "export_count": len(file_deps.exports)
        })
        
        return file_deps
        
    def _extract_imports_from_chunk(self, chunk: Dict[str, Any]) -> List[ImportInfo]:
        """Extract import information from an imports chunk"""
        imports = []
        
        # Get modules from metadata if available
        modules = chunk.get('metadata', {}).get('modules', [])
        content = chunk.get('content', '')
        
        # Parse content for more detailed import info
        import_lines = content.strip().split('\n')
        
        for line in import_lines:
            line = line.strip()
            if not line:
                continue
                
            # Handle 'import module' style
            if line.startswith('import '):
                match = re.match(r'import\s+(.+)', line)
                if match:
                    module_str = match.group(1)
                    # Handle multiple imports: import os, sys
                    for module in module_str.split(','):
                        module = module.strip()
                        if ' as ' in module:
                            module = module.split(' as ')[0].strip()
                        imports.append(ImportInfo(
                            module=module,
                            is_relative=False
                        ))
                        
            # Handle 'from module import ...' style
            elif line.startswith('from '):
                match = re.match(r'from\s+(\.*)([\w.]*)\s+import\s+(.+)', line)
                if match:
                    dots = match.group(1)
                    module = match.group(2)
                    names_str = match.group(3)
                    
                    # Parse imported names
                    names = []
                    if names_str.strip() != '*':
                        for name in names_str.split(','):
                            name = name.strip()
                            if ' as ' in name:
                                name = name.split(' as ')[0].strip()
                            names.append(name)
                    
                    imports.append(ImportInfo(
                        module=module,
                        names=names,
                        is_relative=bool(dots),
                        level=len(dots)
                    ))
                    
        return imports
        
    def _update_module_mapping(self, file_path: str, file_deps: FileDependencies):
        """Update the mapping from module names to file paths"""
        # Convert file path to potential module name
        path = Path(file_path)
        
        # Remove file extension and convert to module path
        if path.suffix == '.py':
            if path.stem == '__init__':
                # Package init file
                module_parts = path.parent.parts
            else:
                # Regular module
                module_parts = list(path.parent.parts) + [path.stem]
                
            # Find the likely module name (this is heuristic)
            # Look for common Python project roots
            for i, part in enumerate(module_parts):
                if part in ['src', 'lib', 'app'] and i < len(module_parts) - 1:
                    module_parts = module_parts[i+1:]
                    break
                    
            module_name = '.'.join(module_parts)
            
            if module_name:
                if module_name not in self._module_to_files:
                    self._module_to_files[module_name] = set()
                self._module_to_files[module_name].add(file_path)
                
    def resolve_import_to_file(self, import_info: ImportInfo, from_file: str) -> Optional[str]:
        """Resolve an import to its likely file path"""
        # Handle relative imports
        if import_info.is_relative:
            from_path = Path(from_file)
            current_dir = from_path.parent
            
            # Go up directories based on level
            for _ in range(import_info.level - 1):
                current_dir = current_dir.parent
                
            # Try to find the module
            if import_info.module:
                potential_file = current_dir / f"{import_info.module.replace('.', '/')}.py"
                if potential_file.exists():
                    return str(potential_file)
                    
                # Try as package
                potential_init = current_dir / import_info.module.replace('.', '/') / '__init__.py'
                if potential_init.exists():
                    return str(potential_init)
        else:
            # Absolute import - check our module mapping
            if import_info.module in self._module_to_files:
                files = self._module_to_files[import_info.module]
                if files:
                    return list(files)[0]  # Return first match
                    
        return None
        
    def build_reverse_dependencies(self):
        """Build the imported_by relationships"""
        for file_path, file_deps in self.dependencies.items():
            for import_info in file_deps.imports:
                # Try to resolve the import to a file
                imported_file = self.resolve_import_to_file(import_info, file_path)
                
                if imported_file and imported_file in self.dependencies:
                    self.dependencies[imported_file].imported_by.add(file_path)
                    
    def get_file_dependencies(self, file_path: str) -> Optional[FileDependencies]:
        """Get dependencies for a specific file"""
        return self.dependencies.get(file_path)
        
    def get_dependent_files(self, file_path: str, max_depth: int = 1) -> Set[str]:
        """Get files that depend on the given file (import it)"""
        dependent_files = set()
        
        if file_path in self.dependencies:
            # Direct dependents
            direct_deps = self.dependencies[file_path].imported_by
            dependent_files.update(direct_deps)
            
            # Recursive if depth > 1
            if max_depth > 1:
                for dep_file in list(direct_deps):
                    recursive_deps = self.get_dependent_files(dep_file, max_depth - 1)
                    dependent_files.update(recursive_deps)
                    
        return dependent_files
        
    def get_imported_files(self, file_path: str, max_depth: int = 1) -> Set[str]:
        """Get files that are imported by the given file"""
        imported_files = set()
        
        if file_path in self.dependencies:
            file_deps = self.dependencies[file_path]
            
            # Resolve imports to files
            for import_info in file_deps.imports:
                imported_file = self.resolve_import_to_file(import_info, file_path)
                if imported_file:
                    imported_files.add(imported_file)
                    
                    # Recursive if depth > 1
                    if max_depth > 1:
                        recursive_imports = self.get_imported_files(imported_file, max_depth - 1)
                        imported_files.update(recursive_imports)
                        
        return imported_files
        
    def calculate_dependency_distance(self, file1: str, file2: str, max_depth: int = 3) -> Optional[int]:
        """Calculate the minimum dependency distance between two files"""
        if file1 == file2:
            return 0
            
        # Check direct imports
        if file1 in self.dependencies:
            imported_files = self.get_imported_files(file1, max_depth=1)
            if file2 in imported_files:
                return 1
                
        if file2 in self.dependencies:
            imported_files = self.get_imported_files(file2, max_depth=1)
            if file1 in imported_files:
                return 1
                
        # Check deeper connections
        for depth in range(2, max_depth + 1):
            file1_imports = self.get_imported_files(file1, max_depth=depth)
            file2_imports = self.get_imported_files(file2, max_depth=depth)
            
            if file2 in file1_imports or file1 in file2_imports:
                return depth
                
        return None  # No connection found within max_depth
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dependency graph"""
        total_imports = sum(len(fd.imports) for fd in self.dependencies.values())
        total_exports = sum(len(fd.exports) for fd in self.dependencies.values())
        files_with_deps = sum(1 for fd in self.dependencies.values() if fd.imported_by)
        
        return {
            "total_files": len(self.dependencies),
            "total_imports": total_imports,
            "total_exports": total_exports,
            "files_with_dependents": files_with_deps,
            "unique_modules": len(self._module_to_files)
        }


# Global instance for the current project
_dependency_graph: Optional[DependencyGraphBuilder] = None


def get_dependency_graph() -> DependencyGraphBuilder:
    """Get or create the global dependency graph instance"""
    global _dependency_graph
    if _dependency_graph is None:
        _dependency_graph = DependencyGraphBuilder()
    return _dependency_graph


def reset_dependency_graph():
    """Reset the global dependency graph"""
    global _dependency_graph
    _dependency_graph = None