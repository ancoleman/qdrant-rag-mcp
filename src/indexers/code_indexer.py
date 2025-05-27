# src/indexers/code_indexer.py
"""
Code indexer module for processing source code files

Handles extraction of code metadata, intelligent chunking,
and preparation for vector embedding.
"""

import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import AST chunker
try:
    from utils.ast_chunker import create_ast_chunker, ASTChunk
    AST_CHUNKING_AVAILABLE = True
except ImportError:
    AST_CHUNKING_AVAILABLE = False
    ASTChunk = None

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_path: str
    chunk_index: int
    chunk_count: int
    line_start: int
    line_end: int
    metadata: Dict[str, Any]


class CodeIndexer:
    """Handles indexing of source code files"""
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300, use_ast_chunking: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_ast_chunking = use_ast_chunking and AST_CHUNKING_AVAILABLE
        
        # Language-specific settings
        self.language_settings = {
            ".py": {
                "separators": ["\nclass ", "\ndef ", "\n\n", "\n", " ", ""],
                "import_pattern": r'(?:from\s+[\w.]+\s+)?import\s+(?:[\w.]+(?:\s+as\s+\w+)?(?:,\s*)?)+',
                "class_pattern": r'class\s+(\w+)',
                "function_pattern": r'def\s+(\w+)'
            },
            ".js": {
                "separators": ["\nclass ", "\nfunction ", "\nconst ", "\nlet ", "\n\n", "\n", " ", ""],
                "import_pattern": r'(?:import\s+.*?from\s+[\'"].+?[\'"]|require\s*\([\'"].+?[\'\"]\))',
                "class_pattern": r'class\s+(\w+)',
                "function_pattern": r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*)?=>)'
            },
            ".java": {
                "separators": ["\npublic class ", "\nprivate class ", "\npublic static ", "\npublic ", "\n\n", "\n", " ", ""],
                "import_pattern": r'import\s+(?:static\s+)?[\w.]+(?:\.\*)?;',
                "class_pattern": r'(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)',
                "function_pattern": r'(?:public|private|protected)\s+(?:static\s+)?(?:\w+\s+)?(\w+)\s*\('
            }
        }
        
        # Default text splitter
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def index_file(self, file_path: str) -> List[CodeChunk]:
        """Index a single code file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check if we should use AST chunking
        ast_supported = ['.py', '.sh', '.bash', '.go']
        if self.use_ast_chunking and file_path.suffix in ast_supported:
            return self._index_with_ast(file_path)
        else:
            return self._index_with_text_splitter(file_path)
    
    def _index_with_ast(self, file_path: Path) -> List[CodeChunk]:
        """Index using AST-based chunking"""
        # Determine language from extension
        language = file_path.suffix
        ast_chunker = create_ast_chunker(language, 
                                       max_chunk_size=self.chunk_size,
                                       min_chunk_size=100)
        
        if not ast_chunker:
            # Fallback to text splitter
            return self._index_with_text_splitter(file_path)
        
        try:
            # Get AST chunks
            ast_chunks = ast_chunker.chunk_file(str(file_path))
            
            # Convert ASTChunk to CodeChunk
            code_chunks = []
            for ast_chunk in ast_chunks:
                # Extract metadata from AST chunk
                # Get language from metadata or infer from extension
                lang = ast_chunk.metadata.get('language', self.get_language_from_file(str(file_path)))
                
                metadata = {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_type": file_path.suffix,
                    "language": lang,
                    "chunk_type": ast_chunk.chunk_type,
                    "hierarchy": ast_chunk.hierarchy,
                    "name": ast_chunk.name,
                    **ast_chunk.metadata
                }
                
                code_chunk = CodeChunk(
                    content=ast_chunk.content,
                    file_path=str(file_path),
                    chunk_index=ast_chunk.chunk_index,
                    chunk_count=len(ast_chunks),
                    line_start=ast_chunk.line_start,
                    line_end=ast_chunk.line_end,
                    metadata=metadata
                )
                code_chunks.append(code_chunk)
            
            logger.info(f"AST indexed {file_path}: {len(code_chunks)} chunks")
            return code_chunks
            
        except Exception as e:
            logger.warning(f"AST chunking failed for {file_path}, falling back to text splitter: {e}")
            return self._index_with_text_splitter(file_path)
    
    def _index_with_text_splitter(self, file_path: Path) -> List[CodeChunk]:
        """Index using traditional text splitting"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            logger.warning(f"Failed to decode {file_path}, trying latin-1")
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        # Extract metadata
        metadata = self._extract_metadata(content, file_path)
        
        # Create chunks
        chunks = self._create_chunks(content, file_path)
        
        # Create CodeChunk objects
        code_chunks = []
        for i, chunk in enumerate(chunks):
            # Calculate line numbers
            lines_before = content[:chunk["start"]].count('\n')
            lines_in_chunk = chunk["content"].count('\n')
            
            code_chunk = CodeChunk(
                content=chunk["content"],
                file_path=str(file_path),
                chunk_index=i,
                chunk_count=len(chunks),
                line_start=lines_before + 1,
                line_end=lines_before + lines_in_chunk + 1,
                metadata={
                    **metadata,
                    "language": file_path.suffix,
                    "chunk_type": chunk.get("type", "general")
                }
            )
            code_chunks.append(code_chunk)
        
        logger.info(f"Text-split indexed {file_path}: {len(code_chunks)} chunks")
        return code_chunks
    
    def _extract_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from code content"""
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_path.suffix,
            "imports": [],
            "classes": [],
            "functions": [],
            "size": len(content),
            "lines": content.count('\n') + 1
        }
        
        # Get language-specific patterns
        settings = self.language_settings.get(file_path.suffix, {})
        
        if import_pattern := settings.get("import_pattern"):
            metadata["imports"] = re.findall(import_pattern, content)
        
        if class_pattern := settings.get("class_pattern"):
            metadata["classes"] = re.findall(class_pattern, content)
        
        if function_pattern := settings.get("function_pattern"):
            functions = re.findall(function_pattern, content)
            # Flatten tuples from regex groups
            metadata["functions"] = [f for group in functions for f in (group if isinstance(group, tuple) else (group,)) if f]
        
        # Try to get module docstring for Python files
        if file_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    metadata["docstring"] = docstring[:200]  # First 200 chars
            except:
                pass
        
        return metadata
    
    def _create_chunks(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Create intelligent chunks from code content"""
        settings = self.language_settings.get(file_path.suffix, {})
        separators = settings.get("separators", self.default_splitter._separators)
        
        # Create specialized splitter for this file type
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators,
            length_function=len
        )
        
        # Split the content
        text_chunks = splitter.split_text(content)
        
        # Create chunk dictionaries with position information
        chunks = []
        current_pos = 0
        
        for chunk_text in text_chunks:
            # Find the position of this chunk in the original content
            chunk_start = content.find(chunk_text, current_pos)
            if chunk_start == -1:
                # Fallback for overlapping chunks
                chunk_start = current_pos
            
            chunk_end = chunk_start + len(chunk_text)
            
            # Determine chunk type
            chunk_type = self._determine_chunk_type(chunk_text, file_path.suffix)
            
            chunks.append({
                "content": chunk_text,
                "start": chunk_start,
                "end": chunk_end,
                "type": chunk_type
            })
            
            current_pos = chunk_start + 1
        
        return chunks
    
    def _determine_chunk_type(self, chunk_text: str, file_suffix: str) -> str:
        """Determine the type of code chunk (class, function, imports, etc.)"""
        chunk_lower = chunk_text.lower()
        
        # Check for common patterns
        if file_suffix == ".py":
            if chunk_lower.startswith("import ") or chunk_lower.startswith("from "):
                return "imports"
            elif "class " in chunk_lower[:50]:
                return "class"
            elif "def " in chunk_lower[:50]:
                return "function"
        elif file_suffix in [".js", ".jsx", ".ts", ".tsx"]:
            if chunk_lower.startswith("import ") or "require(" in chunk_lower[:50]:
                return "imports"
            elif "class " in chunk_lower[:50]:
                return "class"
            elif "function " in chunk_lower[:50] or "const " in chunk_lower[:50]:
                return "function"
        elif file_suffix == ".java":
            if chunk_lower.startswith("import "):
                return "imports"
            elif "class " in chunk_lower[:100]:
                return "class"
            elif re.search(r'(public|private|protected)\s+', chunk_lower[:100]):
                return "method"
        
        return "general"
    
    def extract_dependencies(self, file_path: str) -> List[str]:
        """Extract dependencies from a code file"""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return []
        
        settings = self.language_settings.get(file_path.suffix, {})
        import_pattern = settings.get("import_pattern")
        
        if not import_pattern:
            return []
        
        imports = re.findall(import_pattern, content)
        
        # Clean up imports to get just the module names
        dependencies = []
        for imp in imports:
            if file_path.suffix == ".py":
                # Extract module name from Python imports
                match = re.search(r'(?:from\s+)?([\w.]+)', imp)
                if match:
                    dependencies.append(match.group(1))
            elif file_path.suffix in [".js", ".jsx", ".ts", ".tsx"]:
                # Extract module name from JS imports
                match = re.search(r'[\'"]([^\'"\s]+)[\'"]', imp)
                if match:
                    dependencies.append(match.group(1))
            elif file_path.suffix == ".java":
                # Extract package name from Java imports
                match = re.search(r'import\s+(?:static\s+)?([\w.]+)', imp)
                if match:
                    dependencies.append(match.group(1))
        
        return list(set(dependencies))  # Remove duplicates
    
    def get_language_from_file(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        suffix = Path(file_path).suffix.lower()
        
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".m": "matlab",
            ".sh": "shell",
            ".bash": "shell",
            ".ps1": "powershell"
        }
        
        return language_map.get(suffix, "unknown")
