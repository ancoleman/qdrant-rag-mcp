# Documentation Indexer Implementation Plan - v0.2.3

## Overview

The Documentation Indexer will provide specialized handling for markdown and other documentation files, enabling proper indexing and search of documentation alongside code.

## Problem Statement

Currently:
- Markdown files (*.md) are not indexed at all
- Documentation is crucial for understanding codebases but is ignored
- No way to search README files, CHANGELOG, or docs folders
- Code and Config indexers are not suitable for prose/documentation

## Solution Design

### 1. New DocumentationIndexer Class

Location: `src/indexers/documentation_indexer.py`

```python
class DocumentationIndexer:
    """Specialized indexer for markdown and documentation files."""
    
    def __init__(self):
        self.supported_extensions = ['.md', '.markdown', '.rst', '.txt']
        self.chunk_size = 2000  # Larger chunks for documentation
        self.chunk_overlap = 400
```

### 2. Chunking Strategy

Unlike code/config indexers, documentation should be chunked by:

1. **Heading-based chunking**:
   - Split at major headings (##, ###)
   - Keep heading hierarchy in metadata
   - Preserve context by including parent headings

2. **Smart paragraph chunking**:
   - Keep related paragraphs together
   - Don't break in middle of lists or code blocks
   - Respect markdown structure

### 3. Metadata Extraction

Extract and store:
- Document title (from # heading or filename)
- Heading hierarchy (breadcrumb trail)
- Code block languages
- Internal/external links
- Table of contents structure
- Frontmatter (if present)

### 4. Implementation Steps

#### Phase 1: Basic Implementation (Day 1)
- [ ] Create `documentation_indexer.py`
- [ ] Implement basic markdown parsing
- [ ] Add heading-based chunking
- [ ] Extract basic metadata (title, headings)

#### Phase 2: Enhanced Features (Day 2)
- [ ] Add code block extraction with language detection
- [ ] Implement link extraction and tracking
- [ ] Add frontmatter parsing
- [ ] Handle special markdown elements (tables, lists)

#### Phase 3: Integration (Day 3)
- [ ] Add `index_documentation()` function to main server
- [ ] Create `document_collection` in Qdrant
- [ ] Add `*.md` to default patterns in `index_directory()`
- [ ] Implement `search_docs()` function

#### Phase 4: Testing & Refinement (Day 4)
- [ ] Test with various markdown formats
- [ ] Ensure proper handling of edge cases
- [ ] Add documentation-specific ranking signals
- [ ] Update existing documentation

## Technical Details

### Chunking Algorithm

```python
def chunk_by_sections(self, content: str) -> List[Dict[str, Any]]:
    """
    Split markdown content by sections while preserving context.
    
    Returns chunks with:
    - content: The text content
    - heading: The immediate heading
    - heading_hierarchy: List of parent headings
    - chunk_type: 'heading', 'paragraph', 'code_block', 'list'
    - metadata: Additional info (links, code languages, etc.)
    """
```

### Metadata Schema

```python
{
    "file_path": "/docs/README.md",
    "doc_type": "markdown",
    "title": "Project Documentation",
    "heading": "Installation Guide",
    "heading_hierarchy": ["Project Documentation", "Getting Started", "Installation Guide"],
    "heading_level": 3,
    "chunk_index": 2,
    "chunk_type": "section",
    "has_code_blocks": True,
    "code_languages": ["python", "bash"],
    "internal_links": ["../api/reference.md", "#configuration"],
    "external_links": ["https://example.com"],
    "word_count": 256,
    "modified_at": "2025-05-29T10:00:00Z"
}
```

### Search Enhancements

1. **Documentation-specific search mode**:
   ```python
   @mcp.tool()
   def search_docs(query: str, doc_type: str = None, n_results: int = 5) -> Dict[str, Any]:
       """Search specifically in documentation files."""
   ```

2. **Ranking adjustments**:
   - Boost exact heading matches
   - Prioritize README and main documentation
   - Consider heading hierarchy in relevance

### Collection Configuration

```python
# Separate collection for documentation
document_collection = "project_documentation"

# Schema includes documentation-specific fields
vector_config = VectorParams(
    size=384,  # Same embedding model
    distance=Distance.COSINE
)
```

## Benefits

1. **Better Documentation Discovery**: Find relevant docs quickly
2. **Context-Aware Search**: Understanding section hierarchy
3. **Code Example Finding**: Search for code snippets in docs
4. **Cross-Reference Support**: Track documentation relationships
5. **Migration Guide Access**: Easy access to upgrade instructions

## Success Metrics

- Successfully index all .md files in a project
- Search returns relevant documentation sections
- Code blocks in docs are searchable
- Heading hierarchy provides useful context
- No performance degradation

## Future Enhancements (v0.2.4+)

1. Support for other formats (.rst, .adoc, .textile)
2. Wiki-style link graph analysis
3. Documentation quality scoring
4. Auto-generated documentation detection
5. API documentation special handling