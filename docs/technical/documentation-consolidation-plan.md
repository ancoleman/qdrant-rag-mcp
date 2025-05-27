# Documentation Consolidation Plan

Generated: 2025-05-23

This plan outlines the consolidation of documentation based on semantic search analysis to reduce redundancy and improve user experience.

## Current State
- 17 documentation files in docs/
- Significant overlap between guides
- Outdated references to deprecated features
- No clear hierarchy for different documentation types

## Consolidation Recommendations

### 1. Core Documentation to Keep (Primary Guides)
These should remain in the main `docs/` directory:

- **README.md** - Main entry point ✓
- **complete-setup-and-usage-guide.md** - Primary comprehensive guide
- **context-aware-guide.md** - Essential for understanding the key feature
- **mcp-scope-configuration-guide.md** - Critical for proper global setup
- **rag-usage-examples.md** - Practical examples users need

### 2. Redundant/Overlapping Documentation
These have significant overlap and should be consolidated or moved:

#### Installation Guides (Too Many)
- `complete-global-setup-guide.md` - Overlaps with setup guide
- `global-usage-guide.md` - Redundant with context-aware guide
- `installation-playbook.md` - Too detailed for most users
- `context-awareness-playbook.md` - Too technical/internal

**Action**: Merge essential content into `complete-setup-and-usage-guide.md`

#### Auto-Indexing Guides (Outdated)
- `auto-reindexing-guide.md` - Outdated approach
- `integrated-auto-indexing.md` - References deprecated scripts

**Action**: Update and merge into a single section in main setup guide

### 3. Reference Documentation
Keep but reorganize:

- **qdrant-quick-reference.md** - Useful commands reference
- **mps-quick-reference.md** - Apple Silicon specific
- **claude-code-troubleshooting.md** - Essential troubleshooting

**Action**: Create a `docs/reference/` subdirectory

### 4. Technical/Research Documentation
Move to separate location:

- **qdrant_mcp_research.md** - Research on MCP protocol race conditions
- **enhanced-qdrant-rag-guide.md** - Technical implementation details
- **documentation-accuracy-report.md** - Meta documentation

**Action**: Create `docs/technical/` subdirectory

### 5. Deprecated Documentation
Should be removed or archived:

- **complete-qdrant-setup-guide.md** - Has outdated references and overlaps

## Proposed New Structure

```
docs/
├── README.md                           # Keep as overview
├── complete-setup-and-usage-guide.md   # Main comprehensive guide
├── context-aware-guide.md              # Essential feature guide
├── mcp-scope-configuration-guide.md    # Critical setup info
├── rag-usage-examples.md               # Practical examples
├── reference/                          # Quick reference guides
│   ├── qdrant-quick-reference.md
│   ├── mps-quick-reference.md
│   └── troubleshooting.md
└── technical/                          # Technical/research docs
    ├── qdrant-mcp-research.md
    ├── architecture.md                 # Extract from enhanced guide
    ├── implementation-details.md       # Extract from enhanced guide
    └── documentation-accuracy-report.md
```

## Consolidation Tasks

### Phase 1: Create Directory Structure
1. Create `docs/reference/` directory
2. Create `docs/technical/` directory

### Phase 2: Move Reference Documentation
1. Move `qdrant-quick-reference.md` to `docs/reference/`
2. Move `mps-quick-reference.md` to `docs/reference/`
3. Move `claude-code-troubleshooting.md` to `docs/reference/troubleshooting.md`

### Phase 3: Move Technical Documentation
1. Move `qdrant_mcp_research.md` to `docs/technical/`
2. Move `documentation-accuracy-report.md` to `docs/technical/`
3. Extract technical content from `enhanced-qdrant-rag-guide.md`

### Phase 4: Consolidate Installation Documentation
1. Extract unique content from `complete-global-setup-guide.md`
2. Extract unique content from `global-usage-guide.md`
3. Merge into `complete-setup-and-usage-guide.md`
4. Delete redundant files

### Phase 5: Consolidate Auto-Indexing Documentation
1. Extract current approach from `auto-reindexing-guide.md`
2. Extract current approach from `integrated-auto-indexing.md`
3. Add consolidated section to main guide
4. Delete old auto-indexing files

### Phase 6: Handle Playbooks
1. Extract essential content from `installation-playbook.md`
2. Extract essential content from `context-awareness-playbook.md`
3. Move technical details to `docs/technical/`
4. Delete playbook files

### Phase 7: Final Cleanup
1. Update all internal documentation links
2. Update README.md with new structure
3. Remove `complete-qdrant-setup-guide.md`
4. Create `documentation-consolidation-plan.md` in technical/

## Success Metrics
- Reduce from 17 to ~10 documentation files
- Clear hierarchy with subdirectories
- No redundant information
- All links working correctly
- Easier navigation for users

## Notes
- Preserve git history by using `git mv` for moves
- Test all documentation links after consolidation
- Consider creating redirects for old URLs if needed