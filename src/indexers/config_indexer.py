# src/indexers/config_indexer.py
"""
Configuration file indexer for JSON, XML, YAML files

Handles parsing and intelligent chunking of configuration files
while preserving structure and context.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfigChunk:
    """Represents a chunk of configuration with metadata"""
    content: str
    file_path: str
    chunk_index: int
    path: str  # JSON path or XML path
    value: Any
    metadata: Dict[str, Any]


class ConfigIndexer:
    """Handles indexing of configuration files"""
    
    def __init__(self, chunk_size: int = 2000):
        self.chunk_size = chunk_size
        
        # File type handlers
        self.handlers = {
            ".json": self._index_json,
            ".xml": self._index_xml,
            ".yaml": self._index_yaml,
            ".yml": self._index_yaml,
            ".toml": self._index_toml,
            ".ini": self._index_ini,
            ".env": self._index_env
        }
    
    def index_file(self, file_path: str) -> List[ConfigChunk]:
        """Index a configuration file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        handler = self.handlers.get(suffix)
        
        if not handler:
            logger.warning(f"No handler for file type: {suffix}")
            return self._index_generic(file_path)
        
        try:
            return handler(file_path)
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
            return self._index_generic(file_path)
    
    def _index_json(self, file_path: Path) -> List[ConfigChunk]:
        """Index JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = []
        flattened = self._flatten_json(data)
        
        for i, (path, value, content) in enumerate(flattened):
            chunk = ConfigChunk(
                content=content,
                file_path=str(file_path),
                chunk_index=i,
                path=path,
                value=value,
                metadata={
                    "file_type": "json",
                    "total_keys": len(flattened),
                    "depth": path.count('.') + 1
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Indexed JSON {file_path}: {len(chunks)} chunks")
        return chunks
    
    def _flatten_json(self, data: Any, prefix: str = "") -> List[tuple]:
        """Flatten JSON structure into searchable chunks"""
        results = []
        
        def _flatten(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, (dict, list)):
                        _flatten(value, new_path)
                    else:
                        # Create searchable content
                        content = f"{new_path}: {json.dumps(value, default=str)}"
                        results.append((new_path, value, content))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    if isinstance(item, (dict, list)):
                        _flatten(item, new_path)
                    else:
                        content = f"{new_path}: {json.dumps(item, default=str)}"
                        results.append((new_path, item, content))
        
        _flatten(data, prefix)
        return results
    
    def _index_xml(self, file_path: Path) -> List[ConfigChunk]:
        """Index XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        chunks = []
        elements = self._flatten_xml(root)
        
        for i, (path, element, content) in enumerate(elements):
            chunk = ConfigChunk(
                content=content,
                file_path=str(file_path),
                chunk_index=i,
                path=path,
                value={
                    "tag": element.tag,
                    "attributes": element.attrib,
                    "text": element.text
                },
                metadata={
                    "file_type": "xml",
                    "total_elements": len(elements),
                    "depth": path.count('/') + 1
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Indexed XML {file_path}: {len(chunks)} chunks")
        return chunks
    
    def _flatten_xml(self, root: ET.Element, prefix: str = "") -> List[tuple]:
        """Flatten XML structure into searchable chunks"""
        results = []
        
        def _flatten(element: ET.Element, path: str = ""):
            current_path = f"{path}/{element.tag}" if path else element.tag
            
            # Create content string
            content_parts = [current_path]
            
            if element.attrib:
                attrs = " ".join(f'{k}="{v}"' for k, v in element.attrib.items())
                content_parts.append(f"attributes: {attrs}")
            
            if element.text and element.text.strip():
                content_parts.append(f"text: {element.text.strip()}")
            
            content = " | ".join(content_parts)
            results.append((current_path, element, content))
            
            # Process children
            for child in element:
                _flatten(child, current_path)
        
        _flatten(root, prefix)
        return results
    
    def _index_yaml(self, file_path: Path) -> List[ConfigChunk]:
        """Index YAML file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # YAML can be indexed similar to JSON
        chunks = []
        flattened = self._flatten_json(data)  # Reuse JSON flattening
        
        for i, (path, value, content) in enumerate(flattened):
            chunk = ConfigChunk(
                content=content,
                file_path=str(file_path),
                chunk_index=i,
                path=path,
                value=value,
                metadata={
                    "file_type": "yaml",
                    "total_keys": len(flattened),
                    "depth": path.count('.') + 1
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Indexed YAML {file_path}: {len(chunks)} chunks")
        return chunks
    
    def _index_toml(self, file_path: Path) -> List[ConfigChunk]:
        """Index TOML file"""
        try:
            import toml
            with open(file_path, 'r', encoding='utf-8') as f:
                data = toml.load(f)
            
            # TOML can be indexed similar to JSON
            chunks = []
            flattened = self._flatten_json(data)
            
            for i, (path, value, content) in enumerate(flattened):
                chunk = ConfigChunk(
                    content=content,
                    file_path=str(file_path),
                    chunk_index=i,
                    path=path,
                    value=value,
                    metadata={
                        "file_type": "toml",
                        "total_keys": len(flattened),
                        "depth": path.count('.') + 1
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"Indexed TOML {file_path}: {len(chunks)} chunks")
            return chunks
        except ImportError:
            logger.warning("toml package not installed, treating as text")
            return self._index_generic(file_path)
    
    def _index_ini(self, file_path: Path) -> List[ConfigChunk]:
        """Index INI file"""
        import configparser
        
        config = configparser.ConfigParser()
        config.read(file_path)
        
        chunks = []
        chunk_index = 0
        
        for section in config.sections():
            for key, value in config.items(section):
                path = f"{section}.{key}"
                content = f"{path}: {value}"
                
                chunk = ConfigChunk(
                    content=content,
                    file_path=str(file_path),
                    chunk_index=chunk_index,
                    path=path,
                    value=value,
                    metadata={
                        "file_type": "ini",
                        "section": section,
                        "key": key
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        logger.info(f"Indexed INI {file_path}: {len(chunks)} chunks")
        return chunks
    
    def _index_env(self, file_path: Path) -> List[ConfigChunk]:
        """Index .env file"""
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        chunk_index = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse key=value
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                content = f"{key}: {value}"
                
                chunk = ConfigChunk(
                    content=content,
                    file_path=str(file_path),
                    chunk_index=chunk_index,
                    path=key,
                    value=value,
                    metadata={
                        "file_type": "env",
                        "line_number": line_num,
                        "key": key
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        logger.info(f"Indexed ENV {file_path}: {len(chunks)} chunks")
        return chunks
    
    def _index_generic(self, file_path: Path) -> List[ConfigChunk]:
        """Generic text-based indexing for unknown file types"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple line-based chunking
        lines = content.split('\n')
        chunks = []
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for line_num, line in enumerate(lines, 1):
            current_chunk.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            if current_size >= self.chunk_size:
                chunk_content = '\n'.join(current_chunk)
                
                chunk = ConfigChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    chunk_index=chunk_index,
                    path=f"lines_{chunk_index}",
                    value=chunk_content,
                    metadata={
                        "file_type": file_path.suffix,
                        "start_line": line_num - len(current_chunk) + 1,
                        "end_line": line_num
                    }
                )
                chunks.append(chunk)
                
                current_chunk = []
                current_size = 0
                chunk_index += 1
        
        # Handle remaining lines
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            
            chunk = ConfigChunk(
                content=chunk_content,
                file_path=str(file_path),
                chunk_index=chunk_index,
                path=f"lines_{chunk_index}",
                value=chunk_content,
                metadata={
                    "file_type": file_path.suffix,
                    "start_line": len(lines) - len(current_chunk) + 1,
                    "end_line": len(lines)
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Indexed generic {file_path}: {len(chunks)} chunks")
        return chunks
    
    def extract_schema(self, file_path: str) -> Dict[str, Any]:
        """Extract schema information from configuration file"""
        file_path = Path(file_path)
        
        if file_path.suffix == ".json":
            return self._extract_json_schema(file_path)
        elif file_path.suffix == ".xml":
            return self._extract_xml_schema(file_path)
        else:
            return {"type": "unknown", "file": str(file_path)}
    
    def _extract_json_schema(self, file_path: Path) -> Dict[str, Any]:
        """Extract schema from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        def _infer_type(value: Any) -> str:
            if isinstance(value, bool):
                return "boolean"
            elif isinstance(value, int):
                return "integer"
            elif isinstance(value, float):
                return "number"
            elif isinstance(value, str):
                return "string"
            elif isinstance(value, list):
                return "array"
            elif isinstance(value, dict):
                return "object"
            else:
                return "null"
        
        def _analyze_structure(obj: Any) -> Dict[str, Any]:
            if isinstance(obj, dict):
                properties = {}
                for key, value in obj.items():
                    properties[key] = _analyze_structure(value)
                return {
                    "type": "object",
                    "properties": properties
                }
            elif isinstance(obj, list) and obj:
                # Analyze first item as representative
                return {
                    "type": "array",
                    "items": _analyze_structure(obj[0])
                }
            else:
                return {"type": _infer_type(obj)}
        
        return _analyze_structure(data)
    
    def _extract_xml_schema(self, file_path: Path) -> Dict[str, Any]:
        """Extract schema from XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        def _analyze_element(element: ET.Element) -> Dict[str, Any]:
            schema = {
                "tag": element.tag,
                "attributes": list(element.attrib.keys()),
                "children": {}
            }
            
            # Analyze children
            child_tags = {}
            for child in element:
                if child.tag not in child_tags:
                    child_tags[child.tag] = _analyze_element(child)
            
            schema["children"] = child_tags
            
            return schema
        
        return _analyze_element(root)
