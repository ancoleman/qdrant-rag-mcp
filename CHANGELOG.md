# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.0