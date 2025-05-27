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
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        
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
        
        if len(class_content) <= self.max_chunk_size:
            # Small class - keep as single chunk
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
            # Large class - split into class definition and methods
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
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        
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
                        'is_exported': 'export -f' in func_content
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
    
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_index = 0
        
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
            '.go': 'go'
        }
        language = extension_map.get(language, language)
    
    # Create appropriate chunker
    if language in ['python', 'py']:
        return PythonASTChunker(**kwargs)
    elif language in ['shell', 'sh', 'bash']:
        return ShellScriptChunker(**kwargs)
    elif language in ['go', 'golang']:
        return GoChunker(**kwargs)
    else:
        logger.warning(f"AST chunking not yet supported for {language}")
        return None