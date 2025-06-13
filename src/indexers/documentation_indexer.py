"""
Documentation Indexer for Markdown and Other Documentation Files

This module provides specialized indexing for documentation files,
particularly markdown, with section-based chunking and metadata extraction.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DocumentationIndexer:
    """
    Specialized indexer for markdown and other documentation files.
    
    Chunks documents by sections (headings) and extracts rich metadata
    including heading hierarchy, code blocks, and links.
    """
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        """
        Initialize the documentation indexer.
        
        Args:
            chunk_size: Target size for chunks (characters)
            chunk_overlap: Overlap between chunks for context
        """
        # Validate parameters to prevent division by zero
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_extensions = {'.md', '.markdown', '.rst', '.txt', '.mdx'}
        
        # Regex patterns for markdown parsing
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```(\w*)\n(.*?)\n```', re.DOTALL)
        self.inline_code_pattern = re.compile(r'`([^`]+)`')
        self.link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.frontmatter_pattern = re.compile(r'^---\n(.*?)\n---\n', re.DOTALL)
        
    def is_supported(self, file_path: str) -> bool:
        """Check if file type is supported for documentation indexing."""
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def index_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Index a documentation file and return chunks with metadata.
        
        Args:
            file_path: Path to the documentation file
            
        Returns:
            List of chunks with metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading documentation file {file_path}: {e}")
            return []
        
        # Handle empty files gracefully
        if not content or not content.strip():
            logger.warning(f"Documentation file {file_path} is empty")
            return []
        
        # Extract metadata
        file_metadata = self._extract_file_metadata(file_path, content)
        
        # Parse markdown structure
        sections = self._parse_sections(content)
        
        # Create chunks
        chunks = []
        for section in sections:
            chunk_metadata = {
                **file_metadata,
                **section['metadata'],
                'chunk_index': len(chunks),
                'chunk_type': section['type']
            }
            
            # Split large sections if needed
            if len(section['content']) > self.chunk_size:
                sub_chunks = self._split_large_section(section['content'], chunk_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append({
                    'content': section['content'],
                    'metadata': chunk_metadata
                })
        
        return chunks
    
    def _extract_file_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract file-level metadata."""
        path = Path(file_path)
        
        # Extract title (first # heading or filename)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else path.stem.replace('-', ' ').title()
        
        # Extract frontmatter if present
        frontmatter = {}
        frontmatter_match = self.frontmatter_pattern.match(content)
        if frontmatter_match:
            # Simple key: value parsing (could be enhanced with yaml parser)
            for line in frontmatter_match.group(1).split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()
        
        # Count code blocks and languages
        code_blocks = self.code_block_pattern.findall(content)
        code_languages = list(set(lang for lang, _ in code_blocks if lang))
        
        # Extract links
        links = self.link_pattern.findall(content)
        internal_links = [link for _, link in links if not link.startswith('http')]
        external_links = [link for _, link in links if link.startswith('http')]
        
        return {
            'file_path': file_path,
            'file_name': path.name,
            'doc_type': path.suffix[1:],  # Remove the dot
            'title': title,
            'frontmatter': frontmatter,
            'has_code_blocks': len(code_blocks) > 0,
            'code_languages': code_languages,
            'code_block_count': len(code_blocks),
            'internal_links': internal_links,
            'external_links': external_links,
            'modified_at': datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        }
    
    def _parse_sections(self, content: str) -> List[Dict[str, Any]]:
        """Parse markdown content into sections based on headings."""
        sections = []
        
        # Find all headings with their positions
        headings = []
        for match in self.heading_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2)
            start = match.start()
            headings.append({
                'level': level,
                'title': title,
                'start': start,
                'full_match': match.group(0)
            })
        
        # If no headings, treat entire content as one section
        if not headings:
            sections.append({
                'content': content.strip(),
                'type': 'document',
                'metadata': {
                    'heading': None,
                    'heading_hierarchy': [],
                    'heading_level': 0
                }
            })
            return sections
        
        # Build heading hierarchy and extract sections
        current_hierarchy = []
        
        for i, heading in enumerate(headings):
            # Update hierarchy
            level = heading['level']
            # Remove deeper levels from hierarchy
            current_hierarchy = current_hierarchy[:level-1]
            # Add current heading
            if len(current_hierarchy) < level:
                current_hierarchy.append(heading['title'])
            else:
                current_hierarchy[level-1] = heading['title']
            
            # Determine section end
            section_start = heading['start']
            section_end = headings[i+1]['start'] if i+1 < len(headings) else len(content)
            
            # Extract section content (including the heading)
            section_content = content[section_start:section_end].strip()
            
            # Extract any code blocks in this section
            section_code_blocks = self.code_block_pattern.findall(section_content)
            section_code_languages = list(set(lang for lang, _ in section_code_blocks if lang))
            
            sections.append({
                'content': section_content,
                'type': 'section',
                'metadata': {
                    'heading': heading['title'],
                    'heading_hierarchy': current_hierarchy.copy(),
                    'heading_level': level,
                    'has_code_blocks': len(section_code_blocks) > 0,
                    'section_code_languages': section_code_languages
                }
            })
        
        # Add any content before the first heading
        if headings and headings[0]['start'] > 0:
            preamble = content[:headings[0]['start']].strip()
            if preamble:
                sections.insert(0, {
                    'content': preamble,
                    'type': 'preamble',
                    'metadata': {
                        'heading': 'Introduction',
                        'heading_hierarchy': [],
                        'heading_level': 0
                    }
                })
        
        return sections
    
    def _split_large_section(self, content: str, base_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a large section into smaller chunks while preserving context."""
        chunks = []
        
        # Try to split at paragraph boundaries
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, start a new chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        **base_metadata,
                        'chunk_index': base_metadata.get('chunk_index', 0) + len(chunks),
                        'is_partial': True
                    }
                })
                # Start new chunk with overlap
                if len(current_chunk) > self.chunk_overlap:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk += ("\n\n" if current_chunk else "") + paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    **base_metadata,
                    'chunk_index': base_metadata.get('chunk_index', 0) + len(chunks),
                    'is_partial': len(chunks) > 0
                }
            })
        
        return chunks
    
    def extract_summary(self, content: str, max_length: int = 200) -> str:
        """Extract a summary from the beginning of the content."""
        # Remove headings and code blocks for summary
        clean_content = re.sub(self.heading_pattern, '', content)
        clean_content = re.sub(self.code_block_pattern, '', clean_content)
        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
        
        if len(clean_content) <= max_length:
            return clean_content
        
        # Try to cut at sentence boundary
        sentences = clean_content.split('. ')
        summary = ""
        for sentence in sentences:
            if len(summary) + len(sentence) + 2 <= max_length:
                summary += sentence + ". "
            else:
                break
        
        return summary.strip() or clean_content[:max_length] + "..."