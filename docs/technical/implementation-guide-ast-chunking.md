# Implementation Guide: AST-Based Hierarchical Chunking

This guide provides concrete implementation details for adding AST-based hierarchical chunking to our Qdrant RAG MCP Server, which promises 40-70% token reduction.

## Overview

AST (Abstract Syntax Tree) based chunking parses code files into their syntactic structure, allowing us to create chunks that respect natural code boundaries (functions, classes, methods) rather than arbitrary character counts.

## Current vs. Proposed Architecture

### Current Architecture
```
File → Character-based splitting → Fixed-size chunks → Embeddings → Qdrant
```

### Proposed Architecture
```
File → AST Parser → Hierarchical Structure → Smart Chunks → Embeddings → Qdrant
         ↓
    Language Detection
         ↓
    Structure Extraction
         ↓
    Context Preservation
```

## Implementation Plan

### Step 1: Create AST Parser Interface

```python
# src/indexers/ast_parser.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ast
import hashlib

@dataclass
class CodeStructure:
    """Represents a code structure (function, class, method)"""
    type: str  # 'function', 'class', 'method'
    name: str
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str]
    imports: List[str]
    parent: Optional[str]  # For nested structures
    children: List['CodeStructure']
    
    def get_context_aware_content(self) -> str:
        """Get content with necessary context (imports, class definition for methods)"""
        context_parts = []
        
        # Add imports if this is a top-level structure
        if not self.parent and self.imports:
            context_parts.append('\n'.join(self.imports))
            context_parts.append('')
        
        # Add the actual content
        context_parts.append(self.content)
        
        return '\n'.join(context_parts)
    
    def get_chunk_id(self, file_path: str) -> str:
        """Generate unique chunk ID"""
        unique_str = f"{file_path}:{self.type}:{self.name}:{self.start_line}"
        return hashlib.md5(unique_str.encode()).hexdigest()

class ASTParser(ABC):
    """Abstract base class for language-specific AST parsers"""
    
    @abstractmethod
    def parse_file(self, file_path: str, content: str) -> List[CodeStructure]:
        """Parse file content and return code structures"""
        pass
    
    @abstractmethod
    def extract_imports(self, content: str) -> List[str]:
        """Extract import statements"""
        pass
```

### Step 2: Implement Python AST Parser

```python
# src/indexers/python_ast_parser.py
import ast
from typing import List, Dict, Any, Optional
from .ast_parser import ASTParser, CodeStructure

class PythonASTParser(ASTParser):
    """AST parser for Python files"""
    
    def parse_file(self, file_path: str, content: str) -> List[CodeStructure]:
        """Parse Python file and extract structures"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fallback to line-based parsing for invalid Python
            return self._fallback_parse(content)
        
        imports = self.extract_imports(content)
        structures = []
        
        # Visit all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                structures.append(self._extract_class(node, content, imports))
            elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                structures.append(self._extract_function(node, content, imports))
        
        return structures
    
    def extract_imports(self, content: str) -> List[str]:
        """Extract all import statements"""
        imports = []
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                imports.append(ast.unparse(node))
        
        return imports
    
    def _extract_class(self, node: ast.ClassDef, content: str, imports: List[str]) -> CodeStructure:
        """Extract class definition with methods"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        class_content = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        # Extract methods
        children = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method = self._extract_method(item, content, node.name)
                children.append(method)
        
        return CodeStructure(
            type='class',
            name=node.name,
            start_line=start_line + 1,
            end_line=end_line,
            content=class_content,
            docstring=docstring,
            imports=imports,
            parent=None,
            children=children
        )
    
    def _extract_function(self, node: ast.FunctionDef, content: str, imports: List[str]) -> CodeStructure:
        """Extract standalone function"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        function_content = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        return CodeStructure(
            type='function',
            name=node.name,
            start_line=start_line + 1,
            end_line=end_line,
            content=function_content,
            docstring=docstring,
            imports=imports,
            parent=None,
            children=[]
        )
    
    def _extract_method(self, node: ast.FunctionDef, content: str, class_name: str) -> CodeStructure:
        """Extract method from class"""
        lines = content.split('\n')
        start_line = node.lineno - 1
        end_line = node.end_lineno
        
        method_content = '\n'.join(lines[start_line:end_line])
        docstring = ast.get_docstring(node)
        
        return CodeStructure(
            type='method',
            name=node.name,
            start_line=start_line + 1,
            end_line=end_line,
            content=method_content,
            docstring=docstring,
            imports=[],  # Methods don't have their own imports
            parent=class_name,
            children=[]
        )
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is a method inside a class"""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _fallback_parse(self, content: str) -> List[CodeStructure]:
        """Fallback parsing for invalid Python"""
        # Simple regex-based parsing as fallback
        import re
        structures = []
        
        # Find functions
        func_pattern = r'^def\s+(\w+)\s*\([^)]*\):'
        class_pattern = r'^class\s+(\w+)\s*(?:\([^)]*\))?:'
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.match(func_pattern, line):
                # Extract function
                match = re.match(func_pattern, line)
                if match:
                    name = match.group(1)
                    # Simple heuristic: function ends at next def/class or dedent
                    end_line = self._find_structure_end(lines, i)
                    structures.append(CodeStructure(
                        type='function',
                        name=name,
                        start_line=i + 1,
                        end_line=end_line + 1,
                        content='\n'.join(lines[i:end_line + 1]),
                        docstring=None,
                        imports=[],
                        parent=None,
                        children=[]
                    ))
        
        return structures
    
    def _find_structure_end(self, lines: List[str], start: int) -> int:
        """Find where a structure ends based on indentation"""
        if start >= len(lines):
            return start
        
        # Get initial indentation
        initial_indent = len(lines[start]) - len(lines[start].lstrip())
        
        for i in range(start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent <= initial_indent and not line.lstrip().startswith(('"""', "'''")):
                    return i - 1
        
        return len(lines) - 1
```

### Step 3: Update Code Indexer to Use AST Parser

```python
# src/indexers/code_indexer.py (updated)
from typing import List, Dict, Any, Optional
from .ast_parser import CodeStructure
from .python_ast_parser import PythonASTParser
# Future: from .javascript_ast_parser import JavaScriptASTParser

class HierarchicalCodeIndexer:
    """Enhanced code indexer using AST-based hierarchical chunking"""
    
    def __init__(self, 
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1500,
                 target_chunk_size: int = 500):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.target_chunk_size = target_chunk_size
        
        # Language-specific parsers
        self.parsers = {
            '.py': PythonASTParser(),
            # '.js': JavaScriptASTParser(),
            # '.ts': TypeScriptASTParser(),
        }
    
    def create_hierarchical_chunks(self, file_path: str, content: str) -> Dict[str, List[Dict[str, Any]]]:
        """Create hierarchical chunks from code file"""
        file_ext = Path(file_path).suffix
        
        if file_ext not in self.parsers:
            # Fallback to traditional chunking
            return self._fallback_chunking(file_path, content)
        
        parser = self.parsers[file_ext]
        structures = parser.parse_file(file_path, content)
        
        # Create three levels of chunks
        chunks = {
            'file_summary': self._create_file_summary(file_path, structures),
            'structure_summaries': self._create_structure_summaries(structures),
            'detailed_implementations': self._create_detailed_chunks(structures, file_path)
        }
        
        return chunks
    
    def _create_file_summary(self, file_path: str, structures: List[CodeStructure]) -> List[Dict[str, Any]]:
        """Create high-level file summary chunk"""
        # Extract key information
        classes = [s for s in structures if s.type == 'class']
        functions = [s for s in structures if s.type == 'function']
        
        summary_parts = [f"File: {file_path}"]
        
        if classes:
            summary_parts.append(f"Classes: {', '.join(c.name for c in classes)}")
        
        if functions:
            summary_parts.append(f"Functions: {', '.join(f.name for f in functions)}")
        
        # Add docstrings if available
        for struct in structures[:3]:  # First 3 structures
            if struct.docstring:
                summary_parts.append(f"{struct.name}: {struct.docstring[:100]}...")
        
        summary = '\n'.join(summary_parts)
        
        return [{
            'content': summary,
            'metadata': {
                'file_path': file_path,
                'chunk_type': 'file_summary',
                'chunk_id': hashlib.md5(f"{file_path}:summary".encode()).hexdigest(),
                'token_estimate': len(summary.split()),
                'classes': [c.name for c in classes],
                'functions': [f.name for f in functions]
            }
        }]
    
    def _create_structure_summaries(self, structures: List[CodeStructure]) -> List[Dict[str, Any]]:
        """Create mid-level structure summaries"""
        summaries = []
        
        for struct in structures:
            if struct.type == 'class':
                # Include class definition and method signatures
                summary_lines = []
                
                # Class definition line
                class_def = struct.content.split('\n')[0]
                summary_lines.append(class_def)
                
                # Docstring
                if struct.docstring:
                    summary_lines.append(f'    """{struct.docstring}"""')
                
                # Method signatures
                for method in struct.children:
                    method_sig = method.content.split('\n')[0]
                    summary_lines.append(f"    {method_sig}")
                    if method.docstring:
                        summary_lines.append(f'        """{method.docstring[:50]}..."""')
                
                summary = '\n'.join(summary_lines)
                
                summaries.append({
                    'content': summary,
                    'metadata': {
                        'structure_type': 'class',
                        'structure_name': struct.name,
                        'chunk_type': 'structure_summary',
                        'chunk_id': f"{struct.get_chunk_id('')}:summary",
                        'token_estimate': len(summary.split()),
                        'methods': [m.name for m in struct.children]
                    }
                })
        
        return summaries
    
    def _create_detailed_chunks(self, structures: List[CodeStructure], file_path: str) -> List[Dict[str, Any]]:
        """Create detailed implementation chunks"""
        chunks = []
        
        for struct in structures:
            # Check if structure is small enough to be a single chunk
            content = struct.get_context_aware_content()
            token_estimate = len(content.split())
            
            if token_estimate <= self.max_chunk_size:
                # Single chunk for this structure
                chunks.append({
                    'content': content,
                    'metadata': {
                        'file_path': file_path,
                        'structure_type': struct.type,
                        'structure_name': struct.name,
                        'chunk_type': 'implementation',
                        'chunk_id': struct.get_chunk_id(file_path),
                        'start_line': struct.start_line,
                        'end_line': struct.end_line,
                        'token_estimate': token_estimate,
                        'parent': struct.parent
                    }
                })
            else:
                # Need to split large structures
                sub_chunks = self._split_large_structure(struct, file_path)
                chunks.extend(sub_chunks)
            
            # Process children (methods in classes)
            for child in struct.children:
                child_content = child.get_context_aware_content()
                chunks.append({
                    'content': child_content,
                    'metadata': {
                        'file_path': file_path,
                        'structure_type': child.type,
                        'structure_name': child.name,
                        'chunk_type': 'implementation',
                        'chunk_id': child.get_chunk_id(file_path),
                        'start_line': child.start_line,
                        'end_line': child.end_line,
                        'token_estimate': len(child_content.split()),
                        'parent': child.parent
                    }
                })
        
        return chunks
    
    def _split_large_structure(self, struct: CodeStructure, file_path: str) -> List[Dict[str, Any]]:
        """Split large structures into smaller chunks while preserving context"""
        # This is a simplified version - in practice, you'd want more sophisticated splitting
        lines = struct.content.split('\n')
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        
        # Always include the structure definition
        signature_lines = []
        for i, line in enumerate(lines):
            signature_lines.append(line)
            if ':' in line and not line.strip().endswith(','):
                break
        
        for i, line in enumerate(lines):
            line_tokens = len(line.split())
            
            if current_tokens + line_tokens > self.target_chunk_size and current_chunk_lines:
                # Create chunk with signature + current lines
                chunk_content = '\n'.join(signature_lines + current_chunk_lines)
                chunks.append({
                    'content': chunk_content,
                    'metadata': {
                        'file_path': file_path,
                        'structure_type': struct.type,
                        'structure_name': struct.name,
                        'chunk_type': 'implementation_part',
                        'chunk_id': f"{struct.get_chunk_id(file_path)}:part{len(chunks)}",
                        'part_number': len(chunks),
                        'token_estimate': current_tokens
                    }
                })
                current_chunk_lines = []
                current_tokens = 0
            
            current_chunk_lines.append(line)
            current_tokens += line_tokens
        
        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(signature_lines + current_chunk_lines)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    'file_path': file_path,
                    'structure_type': struct.type,
                    'structure_name': struct.name,
                    'chunk_type': 'implementation_part',
                    'chunk_id': f"{struct.get_chunk_id(file_path)}:part{len(chunks)}",
                    'part_number': len(chunks),
                    'token_estimate': current_tokens
                }
            })
        
        return chunks
```

### Step 4: Update Qdrant Storage Schema

```python
# Updates to src/qdrant_mcp_context_aware.py

def create_hierarchical_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Create collection with hierarchical chunk support"""
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        ),
        payload_schema={
            # Existing fields
            "file_path": "text",
            "chunk_index": "integer",
            
            # New hierarchical fields
            "chunk_type": "keyword",  # 'file_summary', 'structure_summary', 'implementation'
            "structure_type": "keyword",  # 'class', 'function', 'method'
            "structure_name": "text",
            "parent": "text",  # Parent structure name
            "start_line": "integer",
            "end_line": "integer",
            "token_estimate": "integer",
            
            # For linking
            "chunk_id": "keyword",
            "related_chunks": "text[]",  # IDs of related chunks
            
            # For efficient filtering
            "language": "keyword",
            "imports": "text[]",
            "classes": "text[]",
            "functions": "text[]",
            "methods": "text[]"
        }
    )

@mcp.tool()
def search_with_hierarchy(
    query: str,
    n_results: int = 5,
    detail_level: str = "auto",  # 'summary', 'detailed', 'full', 'auto'
    cross_project: bool = False
) -> Dict[str, Any]:
    """
    Enhanced search with hierarchical awareness
    
    Args:
        query: Search query
        n_results: Number of results
        detail_level: Level of detail to return
        cross_project: If True, search across all projects
    """
    # Determine query intent
    if detail_level == "auto":
        detail_level = classify_query_intent(query)
    
    # Adjust search based on detail level
    if detail_level == "summary":
        # Search only summaries
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchAny(any=["file_summary", "structure_summary"])
                )
            ]
        )
        n_results_internal = n_results * 2  # Get more summaries
    elif detail_level == "detailed":
        # Prefer structure summaries and key implementations
        filter_conditions = models.Filter(
            must=[
                models.FieldCondition(
                    key="chunk_type",
                    match=models.MatchAny(any=["structure_summary", "implementation"])
                )
            ]
        )
        n_results_internal = n_results * 3
    else:  # full
        filter_conditions = None
        n_results_internal = n_results * 5
    
    # Perform search...
    # Then use progressive expansion if needed

def classify_query_intent(query: str) -> str:
    """Classify query to determine appropriate detail level"""
    query_lower = query.lower()
    
    # Summary indicators
    summary_keywords = ['overview', 'structure', 'architecture', 'what', 'which', 'list']
    if any(keyword in query_lower for keyword in summary_keywords):
        return "summary"
    
    # Implementation indicators
    impl_keywords = ['how', 'implementation', 'code', 'function', 'method', 'class']
    if any(keyword in query_lower for keyword in impl_keywords):
        return "detailed"
    
    # Default to progressive
    return "detailed"
```

## Benefits of This Implementation

### 1. Token Efficiency
- **File summaries**: 50-100 tokens capture entire file essence
- **Structure summaries**: 200-300 tokens for class overviews
- **Targeted retrieval**: Only fetch implementation when needed

### 2. Better Context Preservation
- Complete functions/classes in single chunks
- Imports included when relevant
- Parent-child relationships maintained

### 3. Improved Search Accuracy
- Can search at appropriate granularity
- Structure-aware filtering
- Relationship traversal

### 4. Progressive Detail Expansion
```
Query: "How does authentication work?"
1. First: Return AuthManager class summary (200 tokens)
2. If needed: Return login() method implementation (500 tokens)
3. If needed: Return related middleware (800 tokens)
Total: 1,500 tokens vs. 10,000 tokens with current approach
```

## Migration Strategy

1. **Phase 1**: Implement for new indexing only
2. **Phase 2**: Add backward compatibility layer
3. **Phase 3**: Provide migration tool for existing indexes
4. **Phase 4**: Deprecate old chunking method

## Performance Considerations

- **Indexing**: AST parsing adds ~20% overhead
- **Storage**: ~30% more metadata per chunk
- **Search**: Hierarchical filtering improves speed
- **Memory**: AST parsing requires more memory for large files

## Next Steps

1. Implement JavaScript/TypeScript parser
2. Add language detection
3. Create progressive retrieval API
4. Build query intent classifier
5. Add performance benchmarks

This implementation would provide the foundation for achieving the promised 40-70% token reduction while improving code understanding accuracy.