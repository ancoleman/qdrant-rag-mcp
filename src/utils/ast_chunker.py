"""
AST-Based Hierarchical Chunking for Code

This module provides structure-aware code chunking using Abstract Syntax Trees (AST).
It preserves complete code structures (functions, classes, methods) and maintains
hierarchical relationships for better retrieval.

Supports:
- Python (using built-in ast module)
- Future: JavaScript, TypeScript, Java, etc.
"""

import ast
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import textwrap

from utils.logging import get_project_logger

logger = get_project_logger()


@dataclass
class ASTChunk:
    """Represents a hierarchical code chunk from AST parsing"""
    content: str
    file_path: str
    chunk_index: int
    line_start: int
    line_end: int
    chunk_type: str  # 'module', 'class', 'function', 'method', 'import'
    name: str  # Name of the structure (function name, class name, etc.)
    hierarchy: List[str]  # ['module', 'ClassName', 'method_name']
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)


class PythonASTChunker:
    """AST-based chunker for Python code"""
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100, 
                 keep_class_together: bool = True):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        self.keep_class_together = keep_class_together  # New option for Phase 2
        
    def chunk_file(self, file_path: str) -> List[ASTChunk]:
        """Parse Python file and create hierarchical chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Get all lines for content extraction
            lines = content.splitlines(keepends=True)
            
            # Extract chunks
            chunks = []
            
            # First, get all imports as a single chunk
            import_chunk = self._extract_imports(tree, lines, file_path)
            if import_chunk:
                chunks.append(import_chunk)
            
            # Then process all top-level definitions
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    chunks.extend(self._process_class(node, lines, file_path, ['module']))
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    chunk = self._process_function(node, lines, file_path, ['module'])
                    if chunk:
                        chunks.append(chunk)
            
            # If no chunks were created (e.g., script with just code), create a module chunk
            if not chunks:
                chunks.append(self._create_module_chunk(content, file_path))
            
            logger.info(f"Created {len(chunks)} AST chunks from {file_path}", extra={
                "operation": "ast_chunk_file",
                "file_path": file_path,
                "chunk_count": len(chunks)
            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to AST parse {file_path}: {e}", extra={
                "operation": "ast_chunk_file_error",
                "file_path": file_path,
                "error": str(e)
            })
            # Fallback to simple chunking
            return self._fallback_chunk(content, file_path)
    
    def _extract_imports(self, tree: ast.AST, lines: List[str], file_path: str) -> Optional[ASTChunk]:
        """Extract all imports as a single chunk"""
        import_nodes = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
        
        if not import_nodes:
            return None
        
        # Sort by line number
        import_nodes.sort(key=lambda n: n.lineno)
        
        # Get the range of lines
        first_line = import_nodes[0].lineno
        last_line = max(n.end_lineno or n.lineno for n in import_nodes)
        
        # Extract content
        content = ''.join(lines[first_line-1:last_line])
        
        # Extract import names
        imports = []
        for node in import_nodes:
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.append(module)
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=first_line,
            line_end=last_line,
            chunk_type='imports',
            name='imports',
            hierarchy=['module', 'imports'],
            metadata={
                'import_count': len(import_nodes),
                'modules': list(set(imports))
            }
        )
        
        self.chunk_index += 1
        return chunk
    
    def _process_class(self, node: ast.ClassDef, lines: List[str], file_path: str, 
                      hierarchy: List[str]) -> List[ASTChunk]:
        """Process a class definition and its methods"""
        chunks = []
        class_hierarchy = hierarchy + [node.name]
        
        # Extract class docstring and signature
        class_start = node.lineno
        first_stmt_line = node.body[0].lineno if node.body else node.end_lineno
        
        # Check if we should create separate chunks for methods
        class_content = ''.join(lines[node.lineno-1:node.end_lineno])
        
        # Phase 2 improvement: Try to keep class and its methods together
        # even if slightly over the limit (up to 1.5x max_chunk_size)
        keep_together_threshold = self.max_chunk_size * 1.5 if self.keep_class_together else self.max_chunk_size
        
        if len(class_content) <= keep_together_threshold:
            # Keep class and methods together as single chunk
            chunk = ASTChunk(
                content=class_content.strip(),
                file_path=file_path,
                chunk_index=self.chunk_index,
                line_start=node.lineno,
                line_end=node.end_lineno,
                chunk_type='class',
                name=node.name,
                hierarchy=class_hierarchy,
                metadata={
                    'bases': [self._get_name(base) for base in node.bases],
                    'decorators': [self._get_name(dec) for dec in node.decorator_list],
                    'method_count': sum(1 for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
                }
            )
            self.chunk_index += 1
            chunks.append(chunk)
        else:
            # Large class - use smart splitting strategy
            if self.keep_class_together:
                # Phase 2: Try to create smart chunks that include class definition + key methods
                chunks.extend(self._smart_split_class(node, lines, file_path, class_hierarchy))
            else:
                # Original behavior: split into class definition and individual methods
                # First, create chunk for class signature and docstring
                class_def_end = first_stmt_line - 1
                
                # Find the end of docstring if present
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    class_def_end = node.body[0].end_lineno
                
                class_def_content = ''.join(lines[node.lineno-1:class_def_end])
                
                chunk = ASTChunk(
                    content=class_def_content.strip() + "\n    ...",  # Indicate continuation
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=node.lineno,
                    line_end=class_def_end,
                    chunk_type='class_definition',
                    name=node.name,
                    hierarchy=class_hierarchy,
                    metadata={
                        'bases': [self._get_name(base) for base in node.bases],
                        'decorators': [self._get_name(dec) for dec in node.decorator_list],
                        'has_methods': True
                    }
                )
                self.chunk_index += 1
                chunks.append(chunk)
                
                # Process each method separately
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_chunk = self._process_function(item, lines, file_path, class_hierarchy, is_method=True)
                        if method_chunk:
                            chunks.append(method_chunk)
        
        return chunks
    
    def _smart_split_class(self, node: ast.ClassDef, lines: List[str], file_path: str,
                          hierarchy: List[str]) -> List[ASTChunk]:
        """
        Smart splitting strategy for large classes that keeps related methods together.
        
        Phase 2 improvement: Group __init__ with class definition, keep related methods together
        """
        chunks = []
        class_hierarchy = hierarchy + [node.name]
        
        # Extract class definition and docstring
        class_start = node.lineno
        first_method_line = None
        
        # Find first method
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_method_line = item.lineno
                break
        
        if not first_method_line:
            # No methods, just return the whole class
            content = ''.join(lines[node.lineno-1:node.end_lineno])
            chunk = ASTChunk(
                content=content.strip(),
                file_path=file_path,
                chunk_index=self.chunk_index,
                line_start=node.lineno,
                line_end=node.end_lineno,
                chunk_type='class',
                name=node.name,
                hierarchy=class_hierarchy,
                metadata={
                    'bases': [self._get_name(base) for base in node.bases],
                    'decorators': [self._get_name(dec) for dec in node.decorator_list],
                    'method_count': 0
                }
            )
            self.chunk_index += 1
            return [chunk]
        
        # Group methods into logical chunks
        method_groups = []
        current_group = {
            'methods': [],
            'start_line': node.lineno,
            'end_line': None,
            'size': 0,
            'has_init': False
        }
        
        # Include class definition in first group
        class_def_lines = lines[node.lineno-1:first_method_line-1]
        current_group['size'] = sum(len(line) for line in class_def_lines)
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_content = ''.join(lines[item.lineno-1:item.end_lineno])
                method_size = len(method_content)
                
                # Special handling for __init__ - always keep with class definition
                if item.name == '__init__':
                    current_group['methods'].append(item)
                    current_group['has_init'] = True
                    current_group['size'] += method_size
                    current_group['end_line'] = item.end_lineno
                # If adding this method would exceed limit and we have methods, start new group
                elif current_group['methods'] and current_group['size'] + method_size > self.max_chunk_size:
                    method_groups.append(current_group)
                    current_group = {
                        'methods': [item],
                        'start_line': item.lineno,
                        'end_line': item.end_lineno,
                        'size': method_size,
                        'has_init': False
                    }
                else:
                    current_group['methods'].append(item)
                    current_group['size'] += method_size
                    current_group['end_line'] = item.end_lineno
        
        # Add the last group
        if current_group['methods']:
            method_groups.append(current_group)
        
        # Create chunks from groups
        for i, group in enumerate(method_groups):
            if i == 0:
                # First chunk includes class definition
                content = ''.join(lines[node.lineno-1:group['end_line']])
                chunk_type = 'class_with_methods'
                method_names = [m.name for m in group['methods']]
                
                chunk = ASTChunk(
                    content=content.strip(),
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=node.lineno,
                    line_end=group['end_line'],
                    chunk_type=chunk_type,
                    name=node.name,
                    hierarchy=class_hierarchy,
                    metadata={
                        'bases': [self._get_name(base) for base in node.bases],
                        'decorators': [self._get_name(dec) for dec in node.decorator_list],
                        'methods': method_names,
                        'has_init': group['has_init'],
                        'chunk_part': f"1/{len(method_groups)}"
                    }
                )
            else:
                # Subsequent chunks are method groups
                content = ''.join(lines[group['start_line']-1:group['end_line']])
                method_names = [m.name for m in group['methods']]
                
                # Add class context comment
                content = f"# Methods from class {node.name}\n" + content
                
                chunk = ASTChunk(
                    content=content.strip(),
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=group['start_line'],
                    line_end=group['end_line'],
                    chunk_type='class_methods',
                    name=f"{node.name}_methods_{i}",
                    hierarchy=class_hierarchy + ['methods'],
                    metadata={
                        'class_name': node.name,
                        'methods': method_names,
                        'chunk_part': f"{i+1}/{len(method_groups)}"
                    }
                )
            
            self.chunk_index += 1
            chunks.append(chunk)
        
        return chunks
    
    def _process_function(self, node: ast.FunctionDef, lines: List[str], file_path: str,
                         hierarchy: List[str], is_method: bool = False) -> Optional[ASTChunk]:
        """Process a function or method definition"""
        func_hierarchy = hierarchy + [node.name]
        
        # Extract function content
        content = ''.join(lines[node.lineno-1:node.end_lineno])
        
        # Skip if too small
        if len(content.strip()) < self.min_chunk_size:
            return None
        
        # If too large, truncate with indicator
        if len(content) > self.max_chunk_size:
            # Try to find a good truncation point
            truncated = content[:self.max_chunk_size]
            # Find last complete line
            last_newline = truncated.rfind('\n')
            if last_newline > 0:
                truncated = truncated[:last_newline]
            content = truncated + "\n    # ... (truncated)"
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=node.lineno,
            line_end=node.end_lineno,
            chunk_type='method' if is_method else 'function',
            name=node.name,
            hierarchy=func_hierarchy,
            metadata={
                'async': isinstance(node, ast.AsyncFunctionDef),
                'decorators': [self._get_name(dec) for dec in node.decorator_list],
                'args': self._extract_args(node.args),
                'returns': self._get_annotation(node.returns) if node.returns else None,
                'is_method': is_method
            }
        )
        
        self.chunk_index += 1
        return chunk
    
    def _create_module_chunk(self, content: str, file_path: str) -> ASTChunk:
        """Create a chunk for module-level code"""
        lines = content.splitlines()
        
        chunk = ASTChunk(
            content=content[:self.max_chunk_size].strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=1,
            line_end=len(lines),
            chunk_type='module',
            name=Path(file_path).stem,
            hierarchy=['module'],
            metadata={
                'is_script': True,
                'truncated': len(content) > self.max_chunk_size
            }
        )
        
        self.chunk_index += 1
        return chunk
    
    def _fallback_chunk(self, content: str, file_path: str) -> List[ASTChunk]:
        """Fallback to simple line-based chunking when AST parsing fails"""
        chunks = []
        lines = content.splitlines(keepends=True)
        
        current_chunk = []
        current_size = 0
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= self.max_chunk_size:
                chunk_content = ''.join(current_chunk).strip()
                if chunk_content:
                    chunk = ASTChunk(
                        content=chunk_content,
                        file_path=file_path,
                        chunk_index=self.chunk_index,
                        line_start=start_line,
                        line_end=i,
                        chunk_type='general',
                        name=f'chunk_{self.chunk_index}',
                        hierarchy=['module'],
                        metadata={'fallback': True}
                    )
                    chunks.append(chunk)
                    self.chunk_index += 1
                
                current_chunk = []
                current_size = 0
                start_line = i + 1
        
        # Add remaining content
        if current_chunk:
            chunk_content = ''.join(current_chunk).strip()
            if chunk_content:
                chunk = ASTChunk(
                    content=chunk_content,
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=start_line,
                    line_end=len(lines),
                    chunk_type='general',
                    name=f'chunk_{self.chunk_index}',
                    hierarchy=['module'],
                    metadata={'fallback': True}
                )
                chunks.append(chunk)
                self.chunk_index += 1
        
        return chunks
    
    def _get_name(self, node: ast.AST) -> str:
        """Extract name from various AST nodes"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif hasattr(node, 'id'):
            return node.id
        else:
            return ast.unparse(node)
    
    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Extract type annotation as string"""
        if node is None:
            return None
        try:
            return ast.unparse(node)
        except:
            return None
    
    def _extract_args(self, args: ast.arguments) -> Dict[str, Any]:
        """Extract function arguments info"""
        arg_info = {
            'args': [arg.arg for arg in args.args],
            'defaults': len(args.defaults),
            'kwonly': [arg.arg for arg in args.kwonlyargs],
            'vararg': args.vararg.arg if args.vararg else None,
            'kwarg': args.kwarg.arg if args.kwarg else None
        }
        return arg_info


class ShellScriptChunker:
    """Structure-aware chunker for Shell scripts"""
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100,
                 keep_class_together: bool = True):  # Accept param for compatibility
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        # Shell scripts don't have classes, but we keep functions together
        self.keep_function_together = keep_class_together
        
    def chunk_file(self, file_path: str) -> List[ASTChunk]:
        """Parse Shell script and create structural chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.splitlines(keepends=True)
            chunks = []
            
            # Extract functions
            function_chunks = self._extract_functions(lines, file_path)
            chunks.extend(function_chunks)
            
            # Extract top-level code (before first function)
            top_level_chunk = self._extract_top_level(lines, file_path, function_chunks)
            if top_level_chunk:
                chunks.insert(0, top_level_chunk)
            
            # If no chunks created, treat as script
            if not chunks:
                chunks.append(self._create_script_chunk(content, file_path))
            
            logger.info(f"Created {len(chunks)} shell chunks from {file_path}", extra={
                "operation": "shell_chunk_file",
                "file_path": file_path,
                "chunk_count": len(chunks)
            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to parse shell script {file_path}: {e}", extra={
                "operation": "shell_chunk_file_error",
                "file_path": file_path,
                "error": str(e)
            })
            return self._fallback_chunk(content, file_path)
    
    def _extract_functions(self, lines: List[str], file_path: str) -> List[ASTChunk]:
        """Extract shell functions"""
        import re
        chunks = []
        
        # Regex for shell function definitions
        func_pattern = re.compile(r'^(?:function\s+)?(\w+)\s*\(\s*\)\s*{?', re.MULTILINE)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = func_pattern.match(line.strip())
            
            if match:
                func_name = match.group(1)
                start_line = i + 1  # 1-based
                
                # Find the end of the function
                brace_count = 1 if '{' in line else 0
                end_line = i
                
                # Look for opening brace if not on same line
                if brace_count == 0:
                    for j in range(i + 1, min(i + 3, len(lines))):
                        if '{' in lines[j]:
                            brace_count = 1
                            end_line = j
                            break
                
                # Find closing brace
                if brace_count > 0:
                    for j in range(end_line + 1, len(lines)):
                        if '{' in lines[j]:
                            brace_count += lines[j].count('{')
                        if '}' in lines[j]:
                            brace_count -= lines[j].count('}')
                            if brace_count == 0:
                                end_line = j
                                break
                
                # Extract function content
                func_content = ''.join(lines[i:end_line + 1])
                
                # Phase 2: Check if we should keep related functions together
                if self.keep_function_together and chunks:
                    last_chunk = chunks[-1]
                    combined_size = len(last_chunk.content) + len(func_content)
                    
                    # If combined size is within threshold, merge functions
                    if combined_size <= self.max_chunk_size * 1.5:
                        # Check if functions are related (simple heuristic: proximity)
                        if start_line - last_chunk.line_end <= 5:
                            # Merge with previous chunk
                            combined_content = last_chunk.content + "\n\n" + func_content
                            last_chunk.content = combined_content.strip()
                            last_chunk.line_end = end_line + 1
                            last_chunk.chunk_type = 'functions'  # Multiple functions
                            last_chunk.name = f"{last_chunk.name}+{func_name}"
                            last_chunk.metadata['function_count'] = last_chunk.metadata.get('function_count', 1) + 1
                            last_chunk.metadata['functions'] = last_chunk.metadata.get('functions', [last_chunk.name.split('+')[0]]) + [func_name]
                            i = end_line + 1
                            continue
                
                chunk = ASTChunk(
                    content=func_content.strip(),
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=start_line,
                    line_end=end_line + 1,
                    chunk_type='function',
                    name=func_name,
                    hierarchy=['script', func_name],
                    metadata={
                        'language': 'shell',
                        'is_exported': 'export -f' in func_content,
                        'function_count': 1,
                        'functions': [func_name]
                    }
                )
                chunks.append(chunk)
                self.chunk_index += 1
                
                i = end_line + 1
            else:
                i += 1
        
        return chunks
    
    def _extract_top_level(self, lines: List[str], file_path: str, 
                          function_chunks: List[ASTChunk]) -> Optional[ASTChunk]:
        """Extract top-level code (variables, sourcing, etc.)"""
        if not function_chunks:
            return None
            
        # Find the first function start
        first_func_line = min(chunk.line_start for chunk in function_chunks)
        
        # Get content before first function
        top_content = ''.join(lines[:first_func_line - 1])
        
        # Skip if too small or just shebang/comments
        meaningful_lines = [l for l in top_content.splitlines() 
                          if l.strip() and not l.strip().startswith('#')]
        
        if len(meaningful_lines) < 2:
            return None
        
        chunk = ASTChunk(
            content=top_content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=1,
            line_end=first_func_line - 1,
            chunk_type='setup',
            name='setup',
            hierarchy=['script', 'setup'],
            metadata={
                'has_shebang': lines[0].startswith('#!'),
                'language': 'shell'
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _create_script_chunk(self, content: str, file_path: str) -> ASTChunk:
        """Create a chunk for the entire script"""
        lines = content.splitlines()
        
        chunk = ASTChunk(
            content=content[:self.max_chunk_size].strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=1,
            line_end=len(lines),
            chunk_type='script',
            name=Path(file_path).stem,
            hierarchy=['script'],
            metadata={
                'language': 'shell',
                'is_executable': content.startswith('#!'),
                'truncated': len(content) > self.max_chunk_size
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _fallback_chunk(self, content: str, file_path: str) -> List[ASTChunk]:
        """Fallback to simple chunking"""
        # Reuse Python chunker's fallback logic
        python_chunker = PythonASTChunker(self.max_chunk_size, self.min_chunk_size)
        python_chunker.chunk_index = self.chunk_index
        return python_chunker._fallback_chunk(content, file_path)


class GoChunker:
    """Structure-aware chunker for Go code"""
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100,
                 keep_class_together: bool = True):  # Accept param for compatibility
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        # Go doesn't have classes, but we keep structs with their methods
        self.keep_struct_together = keep_class_together
        
    def chunk_file(self, file_path: str) -> List[ASTChunk]:
        """Parse Go file and create structural chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.splitlines(keepends=True)
            chunks = []
            
            # Extract package and imports
            package_chunk = self._extract_package_imports(lines, file_path)
            if package_chunk:
                chunks.append(package_chunk)
            
            # Extract structs, interfaces, and functions
            code_chunks = self._extract_go_structures(lines, file_path)
            
            # Phase 2: Group structs with their methods if enabled
            if self.keep_struct_together:
                code_chunks = self._group_structs_with_methods(code_chunks, lines)
            
            chunks.extend(code_chunks)
            
            # If no chunks created, treat as module
            if not chunks:
                chunks.append(self._create_module_chunk(content, file_path))
            
            logger.info(f"Created {len(chunks)} Go chunks from {file_path}", extra={
                "operation": "go_chunk_file",
                "file_path": file_path,
                "chunk_count": len(chunks)
            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to parse Go file {file_path}: {e}", extra={
                "operation": "go_chunk_file_error", 
                "file_path": file_path,
                "error": str(e)
            })
            return self._fallback_chunk(content, file_path)
    
    def _extract_package_imports(self, lines: List[str], file_path: str) -> Optional[ASTChunk]:
        """Extract package declaration and imports"""
        import re
        
        package_line = None
        import_start = None
        import_end = None
        
        for i, line in enumerate(lines):
            if line.strip().startswith('package '):
                package_line = i
            elif line.strip().startswith('import'):
                if import_start is None:
                    import_start = i
                # Check for multi-line imports
                if '(' in line:
                    # Find closing paren
                    for j in range(i + 1, len(lines)):
                        if ')' in lines[j]:
                            import_end = j
                            break
                else:
                    import_end = i
        
        if package_line is not None:
            end_line = import_end if import_end is not None else package_line
            content = ''.join(lines[package_line:end_line + 1])
            
            # Extract package name
            package_match = re.search(r'package\s+(\w+)', lines[package_line])
            package_name = package_match.group(1) if package_match else 'unknown'
            
            chunk = ASTChunk(
                content=content.strip(),
                file_path=file_path,
                chunk_index=self.chunk_index,
                line_start=package_line + 1,
                line_end=end_line + 1,
                chunk_type='package',
                name=package_name,
                hierarchy=['package', package_name],
                metadata={
                    'language': 'go',
                    'has_imports': import_start is not None
                }
            )
            self.chunk_index += 1
            return chunk
        
        return None
    
    def _extract_go_structures(self, lines: List[str], file_path: str) -> List[ASTChunk]:
        """Extract Go functions, methods, structs, and interfaces"""
        import re
        chunks = []
        
        # Patterns for Go structures
        func_pattern = re.compile(r'^func\s+(?:\(.*?\)\s+)?(\w+)\s*\(')
        type_pattern = re.compile(r'^type\s+(\w+)\s+(struct|interface)\s*{')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for function/method
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(1)
                chunk = self._extract_function(lines, i, file_path, func_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            # Check for struct/interface
            type_match = type_pattern.match(line)
            if type_match:
                type_name = type_match.group(1)
                type_kind = type_match.group(2)
                chunk = self._extract_type(lines, i, file_path, type_name, type_kind)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            i += 1
        
        return chunks
    
    def _extract_function(self, lines: List[str], start_idx: int, 
                         file_path: str, func_name: str) -> Optional[ASTChunk]:
        """Extract a Go function or method"""
        import re
        
        # Check if it's a method
        method_pattern = re.compile(r'^func\s+\((\w+)\s+[*]?(\w+)\)\s+(\w+)')
        method_match = method_pattern.match(lines[start_idx].strip())
        
        is_method = False
        receiver_type = None
        if method_match:
            is_method = True
            receiver_type = method_match.group(2)
            func_name = method_match.group(3)
        
        # Find function body
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count == 0 and i > start_idx:
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        # Build hierarchy
        hierarchy = ['package']
        if is_method and receiver_type:
            hierarchy.extend([receiver_type, func_name])
        else:
            hierarchy.append(func_name)
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='method' if is_method else 'function',
            name=func_name,
            hierarchy=hierarchy,
            metadata={
                'language': 'go',
                'is_method': is_method,
                'receiver_type': receiver_type,
                'is_exported': func_name[0].isupper()
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_type(self, lines: List[str], start_idx: int, 
                     file_path: str, type_name: str, type_kind: str) -> Optional[ASTChunk]:
        """Extract a Go struct or interface"""
        # Find type body
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count == 0:
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type=type_kind,
            name=type_name,
            hierarchy=['package', type_name],
            metadata={
                'language': 'go',
                'type_kind': type_kind,
                'is_exported': type_name[0].isupper()
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _create_module_chunk(self, content: str, file_path: str) -> ASTChunk:
        """Create a chunk for the entire Go file"""
        lines = content.splitlines()
        
        chunk = ASTChunk(
            content=content[:self.max_chunk_size].strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=1,
            line_end=len(lines),
            chunk_type='module',
            name=Path(file_path).stem,
            hierarchy=['module'],
            metadata={
                'language': 'go',
                'truncated': len(content) > self.max_chunk_size
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _group_structs_with_methods(self, chunks: List[ASTChunk], lines: List[str]) -> List[ASTChunk]:
        """Group structs with their methods for better cohesion"""
        if not chunks:
            return chunks
        
        # Build a map of struct names to their chunks
        struct_map = {}
        method_chunks = []
        other_chunks = []
        
        for chunk in chunks:
            if chunk.chunk_type == 'struct':
                struct_map[chunk.name] = chunk
            elif chunk.chunk_type == 'method' and 'receiver_type' in chunk.metadata:
                method_chunks.append(chunk)
            else:
                other_chunks.append(chunk)
        
        # Group methods with their structs
        for struct_name, struct_chunk in struct_map.items():
            related_methods = [m for m in method_chunks 
                             if m.metadata.get('receiver_type') == struct_name]
            
            if not related_methods:
                continue
            
            # Calculate combined size
            struct_content = struct_chunk.content
            methods_content = []
            total_size = len(struct_content)
            
            for method in sorted(related_methods, key=lambda x: x.line_start):
                method_size = len(method.content)
                # Check if adding this method would exceed limit
                if total_size + method_size > self.max_chunk_size * 1.5:
                    break
                methods_content.append(method.content)
                total_size += method_size
                method_chunks.remove(method)
            
            if methods_content:
                # Combine struct with its methods
                combined_content = struct_content + "\n\n" + "\n\n".join(methods_content)
                struct_chunk.content = combined_content
                struct_chunk.chunk_type = 'struct_with_methods'
                struct_chunk.line_end = max(struct_chunk.line_end, 
                                          max(m.line_end for m in related_methods if m.content in methods_content))
                struct_chunk.metadata['method_count'] = len(methods_content)
                struct_chunk.metadata['methods'] = [m.name for m in related_methods if m.content in methods_content]
        
        # Combine all chunks back together
        all_chunks = []
        all_chunks.extend(other_chunks)
        all_chunks.extend(struct_map.values())
        all_chunks.extend(method_chunks)  # Remaining ungrouped methods
        
        # Sort by line number to maintain order
        all_chunks.sort(key=lambda x: x.line_start)
        
        return all_chunks
    
    def _fallback_chunk(self, content: str, file_path: str) -> List[ASTChunk]:
        """Fallback to simple chunking"""
        python_chunker = PythonASTChunker(self.max_chunk_size, self.min_chunk_size)
        python_chunker.chunk_index = self.chunk_index
        return python_chunker._fallback_chunk(content, file_path)


class JavaScriptChunker:
    """Structure-aware chunker for JavaScript/TypeScript code"""
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100,
                 keep_class_together: bool = True):  # Accept param for compatibility
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        # JavaScript has classes, keep them with methods like Python
        self.keep_class_together = keep_class_together
        
    def chunk_file(self, file_path: str) -> List[ASTChunk]:
        """Parse JavaScript/TypeScript file and create structural chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.splitlines(keepends=True)
            chunks = []
            
            # Extract imports
            import_chunk = self._extract_imports(lines, file_path)
            if import_chunk:
                chunks.append(import_chunk)
            
            # Extract classes, functions, and methods
            code_chunks = self._extract_js_structures(lines, file_path)
            chunks.extend(code_chunks)
            
            # If no chunks created, treat as module
            if not chunks:
                chunks.append(self._create_module_chunk(content, file_path))
            
            logger.info(f"Created {len(chunks)} JS/TS chunks from {file_path}", extra={
                "operation": "js_chunk_file",
                "file_path": file_path,
                "chunk_count": len(chunks)
            })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to parse JS/TS file {file_path}: {e}", extra={
                "operation": "js_chunk_file_error",
                "file_path": file_path,
                "error": str(e)
            })
            return self._fallback_chunk(content, file_path)
    
    def _extract_imports(self, lines: List[str], file_path: str) -> Optional[ASTChunk]:
        """Extract imports and requires"""
        import re
        import_lines = []
        last_import_line = 0
        
        # Patterns for different import styles
        import_patterns = [
            re.compile(r'^import\s+.*?from\s+[\'"].*?[\'"]', re.MULTILINE),
            re.compile(r'^import\s+[\'"].*?[\'"]', re.MULTILINE),
            re.compile(r'^import\s+\{.*?\}\s+from\s+[\'"].*?[\'"]', re.MULTILINE | re.DOTALL),
            re.compile(r'^import\s+\*\s+as\s+\w+\s+from\s+[\'"].*?[\'"]', re.MULTILINE),
            re.compile(r'^const\s+.*?\s*=\s*require\s*\([\'"].*?[\'\"]\)', re.MULTILINE),
            re.compile(r'^export\s+.*?from\s+[\'"].*?[\'"]', re.MULTILINE)
        ]
        
        for i, line in enumerate(lines):
            # Check if line contains import/require
            for pattern in import_patterns:
                if pattern.match(line.strip()):
                    import_lines.append(i)
                    last_import_line = i
                    break
            
            # Handle multi-line imports
            if i in import_lines and not line.strip().endswith(';') and not line.strip().endswith('}'):
                # Continue to next line
                j = i + 1
                while j < len(lines) and not lines[j].strip().endswith(';') and not lines[j].strip().endswith('}'):
                    import_lines.append(j)
                    last_import_line = j
                    j += 1
                if j < len(lines):
                    import_lines.append(j)
                    last_import_line = j
        
        if not import_lines:
            return None
        
        # Get content of import lines
        first_import = min(import_lines)
        content = ''.join(lines[first_import:last_import_line + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=first_import + 1,
            line_end=last_import_line + 1,
            chunk_type='imports',
            name='imports',
            hierarchy=['module', 'imports'],
            metadata={
                'language': 'javascript',
                'import_count': len(set(import_lines))
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_js_structures(self, lines: List[str], file_path: str) -> List[ASTChunk]:
        """Extract JavaScript/TypeScript classes, functions, and methods"""
        import re
        chunks = []
        
        # Patterns for JS/TS structures
        class_pattern = re.compile(r'^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)')
        function_pattern = re.compile(r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(')
        arrow_function_pattern = re.compile(r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(.*?\)\s*=>')
        method_pattern = re.compile(r'^\s*(?:async\s+)?(\w+)\s*\(.*?\)\s*\{')
        interface_pattern = re.compile(r'^(?:export\s+)?interface\s+(\w+)')
        type_pattern = re.compile(r'^(?:export\s+)?type\s+(\w+)\s*=')
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for class
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                chunk = self._extract_class(lines, i, file_path, class_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            # Check for interface (TypeScript)
            interface_match = interface_pattern.match(line)
            if interface_match:
                interface_name = interface_match.group(1)
                chunk = self._extract_interface(lines, i, file_path, interface_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            # Check for type alias (TypeScript)
            type_match = type_pattern.match(line)
            if type_match:
                type_name = type_match.group(1)
                chunk = self._extract_type_alias(lines, i, file_path, type_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            # Check for function
            func_match = function_pattern.match(line)
            if func_match:
                func_name = func_match.group(1)
                chunk = self._extract_function(lines, i, file_path, func_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            # Check for arrow function
            arrow_match = arrow_function_pattern.match(line)
            if arrow_match:
                func_name = arrow_match.group(1)
                chunk = self._extract_arrow_function(lines, i, file_path, func_name)
                if chunk:
                    chunks.append(chunk)
                    i = chunk.line_end
                    continue
            
            i += 1
        
        return chunks
    
    def _extract_class(self, lines: List[str], start_idx: int, 
                      file_path: str, class_name: str) -> Optional[ASTChunk]:
        """Extract a JavaScript/TypeScript class"""
        # Find class body
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count == 0 and i > start_idx:
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        # Phase 2: Check if we should analyze class size for smart splitting
        if self.keep_class_together and len(content) > self.max_chunk_size:
            # Determine if class should be kept together up to 1.5x limit
            if len(content) <= self.max_chunk_size * 1.5:
                # Keep together
                pass
            else:
                # Class is too large, extract methods for smart grouping
                return self._smart_split_js_class(lines, start_idx, end_idx, file_path, class_name)
        
        # Check if it's exported
        is_exported = 'export' in lines[start_idx]
        
        # Count methods in the class
        import re
        method_pattern = re.compile(r'^\s*(?:async\s+)?(?:static\s+)?(\w+)\s*\(.*?\)\s*{', re.MULTILINE)
        methods = method_pattern.findall(content)
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='class' if len(methods) <= 1 else 'class_with_methods',
            name=class_name,
            hierarchy=['module', class_name],
            metadata={
                'language': 'javascript',
                'is_exported': is_exported,
                'is_abstract': 'abstract' in lines[start_idx],
                'method_count': len(methods),
                'methods': methods[:10]  # First 10 method names
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_interface(self, lines: List[str], start_idx: int,
                          file_path: str, interface_name: str) -> Optional[ASTChunk]:
        """Extract a TypeScript interface"""
        # Find interface body
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count == 0:
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='interface',
            name=interface_name,
            hierarchy=['module', interface_name],
            metadata={
                'language': 'typescript',
                'is_exported': 'export' in lines[start_idx]
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_type_alias(self, lines: List[str], start_idx: int,
                           file_path: str, type_name: str) -> Optional[ASTChunk]:
        """Extract a TypeScript type alias"""
        # Find the end of the type definition
        end_idx = start_idx
        
        # Simple single-line type
        if ';' in lines[start_idx]:
            end_idx = start_idx
        else:
            # Multi-line type
            for i in range(start_idx + 1, len(lines)):
                if ';' in lines[i] or (i + 1 < len(lines) and not lines[i + 1].startswith(' ')):
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='type',
            name=type_name,
            hierarchy=['module', type_name],
            metadata={
                'language': 'typescript',
                'is_exported': 'export' in lines[start_idx]
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_function(self, lines: List[str], start_idx: int,
                         file_path: str, func_name: str) -> Optional[ASTChunk]:
        """Extract a regular function"""
        # Find function body
        brace_count = 0
        end_idx = start_idx
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            if '{' in line:
                brace_count += line.count('{')
            if '}' in line:
                brace_count -= line.count('}')
                if brace_count == 0 and i > start_idx:
                    end_idx = i
                    break
        
        content = ''.join(lines[start_idx:end_idx + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='function',
            name=func_name,
            hierarchy=['module', func_name],
            metadata={
                'language': 'javascript',
                'is_async': 'async' in lines[start_idx],
                'is_exported': 'export' in lines[start_idx]
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _extract_arrow_function(self, lines: List[str], start_idx: int,
                               file_path: str, func_name: str) -> Optional[ASTChunk]:
        """Extract an arrow function"""
        import re
        
        # Check if it's a single-line arrow function
        line = lines[start_idx]
        if '=>' in line and (';' in line or (start_idx + 1 < len(lines) and not lines[start_idx + 1].startswith(' '))):
            # Single line arrow function
            content = line
            end_idx = start_idx
        else:
            # Multi-line arrow function - find the closing brace
            brace_count = 0
            paren_count = 0
            end_idx = start_idx
            
            for i in range(start_idx, len(lines)):
                line = lines[i]
                # Track parentheses (for parameters)
                paren_count += line.count('(') - line.count(')')
                
                # Once we're past the arrow, track braces
                if '=>' in ''.join(lines[start_idx:i+1]):
                    if '{' in line:
                        brace_count += line.count('{')
                    if '}' in line:
                        brace_count -= line.count('}')
                        if brace_count == 0 and paren_count == 0:
                            end_idx = i
                            break
            
            content = ''.join(lines[start_idx:end_idx + 1])
        
        chunk = ASTChunk(
            content=content.strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=start_idx + 1,
            line_end=end_idx + 1,
            chunk_type='arrow_function',
            name=func_name,
            hierarchy=['module', func_name],
            metadata={
                'language': 'javascript',
                'is_async': 'async' in lines[start_idx],
                'is_exported': 'export' in lines[start_idx]
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _create_module_chunk(self, content: str, file_path: str) -> ASTChunk:
        """Create a chunk for the entire JS/TS file"""
        lines = content.splitlines()
        
        chunk = ASTChunk(
            content=content[:self.max_chunk_size].strip(),
            file_path=file_path,
            chunk_index=self.chunk_index,
            line_start=1,
            line_end=len(lines),
            chunk_type='module',
            name=Path(file_path).stem,
            hierarchy=['module'],
            metadata={
                'language': 'javascript',
                'truncated': len(content) > self.max_chunk_size
            }
        )
        self.chunk_index += 1
        return chunk
    
    def _smart_split_js_class(self, lines: List[str], start_idx: int, end_idx: int,
                              file_path: str, class_name: str) -> List[ASTChunk]:
        """Smart splitting for large JavaScript/TypeScript classes"""
        import re
        chunks = []
        class_hierarchy = ['module', class_name]
        
        # Extract class definition and constructor
        constructor_end = start_idx
        method_groups = []
        current_group = {
            'methods': [],
            'start_line': start_idx,
            'end_line': None,
            'size': 0,
            'has_constructor': False
        }
        
        # Include class definition in first group
        class_def_lines = []
        i = start_idx
        while i <= end_idx:
            line = lines[i]
            # Look for constructor
            if re.match(r'^\s*constructor\s*\(', line.strip()):
                # Find end of constructor
                brace_count = 1 if '{' in line else 0
                constructor_start = i
                for j in range(i + 1, end_idx + 1):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                        if brace_count == 0:
                            constructor_end = j
                            current_group['has_constructor'] = True
                            current_group['end_line'] = j
                            break
                i = constructor_end + 1
                continue
            
            # Look for methods
            method_match = re.match(r'^\s*(?:async\s+)?(?:static\s+)?(\w+)\s*\(.*?\)\s*{', line.strip())
            if method_match and i > start_idx:
                method_name = method_match.group(1)
                # Find method end
                brace_count = 1
                method_start = i
                for j in range(i + 1, end_idx + 1):
                    if '{' in lines[j]:
                        brace_count += lines[j].count('{')
                    if '}' in lines[j]:
                        brace_count -= lines[j].count('}')
                        if brace_count == 0:
                            method_end = j
                            method_content = ''.join(lines[method_start:method_end + 1])
                            method_size = len(method_content)
                            
                            # Check if adding this method would exceed limit
                            if current_group['methods'] and current_group['size'] + method_size > self.max_chunk_size:
                                # Start new group
                                method_groups.append(current_group)
                                current_group = {
                                    'methods': [(method_name, method_start, method_end)],
                                    'start_line': method_start,
                                    'end_line': method_end,
                                    'size': method_size,
                                    'has_constructor': False
                                }
                            else:
                                current_group['methods'].append((method_name, method_start, method_end))
                                current_group['size'] += method_size
                                current_group['end_line'] = method_end
                            
                            i = method_end + 1
                            break
                continue
            i += 1
        
        # Add the last group
        if current_group['methods'] or current_group['has_constructor']:
            method_groups.append(current_group)
        
        # Create chunks from groups
        for idx, group in enumerate(method_groups):
            if idx == 0:
                # First chunk includes class definition
                chunk_end = group['end_line'] if group['end_line'] else start_idx + 1
                content = ''.join(lines[start_idx:chunk_end + 1])
                chunk_type = 'class_with_methods'
                
                chunk = ASTChunk(
                    content=content.strip(),
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=start_idx + 1,
                    line_end=chunk_end + 1,
                    chunk_type=chunk_type,
                    name=class_name,
                    hierarchy=class_hierarchy,
                    metadata={
                        'language': 'javascript',
                        'is_exported': 'export' in lines[start_idx],
                        'methods': [m[0] for m in group['methods']],
                        'has_constructor': group['has_constructor'],
                        'chunk_part': f"1/{len(method_groups)}"
                    }
                )
            else:
                # Subsequent chunks are method groups
                content = f"// Methods from class {class_name}\n"
                for method_name, m_start, m_end in group['methods']:
                    content += ''.join(lines[m_start:m_end + 1]) + "\n"
                
                chunk = ASTChunk(
                    content=content.strip(),
                    file_path=file_path,
                    chunk_index=self.chunk_index,
                    line_start=group['start_line'] + 1,
                    line_end=group['end_line'] + 1,
                    chunk_type='class_methods',
                    name=f"{class_name}_methods_{idx}",
                    hierarchy=class_hierarchy + ['methods'],
                    metadata={
                        'class_name': class_name,
                        'methods': [m[0] for m in group['methods']],
                        'chunk_part': f"{idx+1}/{len(method_groups)}"
                    }
                )
            
            self.chunk_index += 1
            chunks.append(chunk)
        
        return chunks
    
    def _fallback_chunk(self, content: str, file_path: str) -> List[ASTChunk]:
        """Fallback to simple chunking"""
        python_chunker = PythonASTChunker(self.max_chunk_size, self.min_chunk_size)
        python_chunker.chunk_index = self.chunk_index
        return python_chunker._fallback_chunk(content, file_path)


def create_ast_chunker(language: str = 'python', **kwargs) -> Optional[object]:
    """Factory function to create appropriate AST chunker"""
    language = language.lower()
    
    # Handle file extensions
    if language.startswith('.'):
        extension_map = {
            '.py': 'python',
            '.sh': 'shell',
            '.bash': 'shell',
            '.go': 'go',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript'
        }
        language = extension_map.get(language, language)
    
    # Create appropriate chunker
    if language in ['python', 'py']:
        return PythonASTChunker(**kwargs)
    elif language in ['shell', 'sh', 'bash']:
        return ShellScriptChunker(**kwargs)
    elif language in ['go', 'golang']:
        return GoChunker(**kwargs)
    elif language in ['javascript', 'js', 'jsx', 'typescript', 'ts', 'tsx']:
        return JavaScriptChunker(**kwargs)
    else:
        logger.warning(f"AST chunking not yet supported for {language}")
        return None