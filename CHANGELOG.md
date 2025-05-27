# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-05-27

### Added
- Project-aware logging system that separates logs by project context
- Structured JSON logging with operation timing and metadata
- Log viewer utility `qdrant-logs` for searching and filtering logs
- Logging configuration in `server_config.json` with environment variable support
- Automatic log rotation (daily + size-based at 10MB)
- Operation-level logging for index, search, and reindex operations

### Changed
- Log directories now use user-friendly names (e.g. `qdrant-rag_70e24d/` instead of `project_70e24d3b08fc/`)
- Improved log viewer to support both old and new directory formats

## [0.1.1] - 2025-05-27

### Added
- New `reindex_directory` MCP tool for clean reindexing
- `clear_project_collections()` helper function to remove stale data
- Force flag for automated reindexing workflows
- Documentation explaining when to use reindex vs index
- Test suite for reindex functionality in `tests/reindex-test/`
- Comprehensive reindex testing plan in `docs/technical/reindex-testing-plan.md`

### Fixed
- Stale data persisting after file deletions/renames
- Search results showing non-existent files

### Changed
- Updated `src/__init__.py` with correct author information

## [0.1.0] - 2025-05-27

### Added
- Initial release of Qdrant RAG MCP Server
- Context-aware project detection and isolation
- Specialized indexers for code and configuration files
- Auto-indexing with file watching support via environment variables
- Global installation support with `install_global.sh`
- FastMCP integration for Claude Code
- HTTP API server for testing and debugging
- Support for multiple embedding models
- MPS acceleration for Apple Silicon
- Comprehensive documentation structure:
  - User guides in `docs/`
  - Quick references in `docs/reference/`
  - Technical documentation in `docs/technical/`
- Scripts for common operations:
  - `setup.sh` - Initial project setup
  - `index_project.sh` - Quick project indexing
  - `manage_models.sh` - Unified model management
  - `download_models.sh` - Pre-download embedding models
  - `test_http_api.sh` - API testing suite
- Docker support with docker-compose configurations
- Python 3.12 compatibility
- Tested with latest versions of all dependencies

### Security
- Environment-based configuration to avoid hardcoded secrets
- `.gitignore` configured to exclude sensitive files
- Secure defaults for all configurations

### Dependencies
- langchain 0.3.25 (Pydantic v2 compatible)
- mcp 1.8.1 (Model Context Protocol SDK)
- sentence-transformers 4.1.0
- qdrant-client 1.14.2
- watchdog 6.0.0
- numpy 2.1.0 (NumPy 2.x support)
- torch 2.4.0
- And other supporting libraries with version constraints

[0.1.2]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.2
[0.1.1]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.1
[0.1.0]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.0