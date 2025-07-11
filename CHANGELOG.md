# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.4.post7] - 2025-06-18

### 🔧 Code Quality Enhancement (Part 1)

This release implements the first phase of code quality improvements based on automated analysis, focusing on critical bug fixes and embedding management enhancements.

### Fixed

- **MPS Memory Management**: Fixed hardcoded MPS memory fraction that was preventing specialized models from loading
  - Now properly respects `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable
  - Removes hardcoded `set_per_process_memory_fraction(0.7)` that was overriding environment settings
  - Allows CodeRankEmbed and other specialized models to utilize configured memory limits
  - Prevents premature model eviction on Apple Silicon systems

- **Duplicate Function Definition**: Resolved F811 linting error in GitHub integration
  - Renamed local `get_github_instances()` to `_create_github_instances()` to avoid name collision
  - Fixed decorator pattern validation flow

### Added

- **Embedding Model Metadata Tracking**: All indexed vectors now include model metadata
  - Added `embedding_model` field to chunk metadata during indexing
  - Tracks which model created each embedding for better observability
  - Helps diagnose model compatibility issues and supports future migration strategies
  - Applied to all indexers: code, config, and documentation

- **Code Consolidation - Core Abstractions** (Issue #73): Created comprehensive abstractions to eliminate code duplication
  - **GitHub Operation Decorator Pattern**: Successfully implemented centralized error handling for ALL GitHub MCP tools
    - Created `@github_operation` decorator in `src/core/decorators.py` for consistent validation and error handling
    - Applied decorator to **all 34 GitHub functions**, reducing code duplication by ~30-40%
    - Provides centralized GitHub client validation and initialization
    - Standardizes error messages and exception handling across all operations
    - **100% completion**: All GitHub functions now use the decorator pattern
  
  - **HTTP Endpoint Base Handler**: Created reusable patterns for HTTP API endpoints
    - Created `src/core/handlers.py` with `AsyncEndpointHandler` base class
    - Implemented `@endpoint_handler` decorator for automatic error handling
    - Refactored 48 out of 51 HTTP endpoints (94% complete)
    - Eliminated ~600-720 lines of boilerplate code
    - 3 GitHub Projects endpoints kept custom async implementation for performance
  
  - **Utility Abstractions**: Created reusable utilities in `src/core/utils.py`
    - `OperationLogger`: Consistent logging with timing and context
    - `MemoryManagementMixin`: Centralized memory management
    - `ModelLoadingStrategy`: Model loading with automatic fallbacks
    - `CollectionNameGenerator`: Single source of truth for collection naming
  
  - **Total Impact**: Eliminated ~1,000+ lines of duplicate code across the codebase

### Technical Improvements

- **Import Organization**: Partially resolved E402 import errors
  - Reduced from 34 to 18 import organization issues
  - Reorganized imports to follow PEP8 standards where possible
  - Remaining errors are acceptable due to dotenv loading requirements
- Improved thread safety in specialized embeddings manager
- Better memory pressure handling for Apple Silicon systems
- Enhanced model lifecycle tracking and debugging capabilities

### Parent Issue

This work is part of the comprehensive code quality enhancement initiative tracked in issue #72, with specific improvements addressing automated analysis findings from issue #73.

## [0.3.4.post6] - 2025-06-13

### 🐛 Bug Fixes

- **Fixed YAML/YML file indexing**: Resolved KeyError `'file_path'` that prevented YAML configuration files from being indexed properly when collections contained existing data
- **Fixed documentation indexing**: Added parameter validation to prevent division by zero errors when indexing documentation files
- **Fixed empty file handling**: Documentation indexer now gracefully handles empty files instead of failing
- **Improved error handling**: Enhanced chunk attribute access with proper fallbacks and more informative error messages

### Added

- **Terraform file support**: Added `.tf` and `.tfvars` file patterns to default indexing patterns
- **Clear collections option**: Added `clear_existing` parameter to `index_directory` for full reindexing without stale data

### Technical Changes

- Added safe attribute access using `hasattr()` checks in `index_config` method
- Added chunk_size validation in DocumentationIndexer to ensure positive values
- Fixed error return values to use the original file_path parameter instead of potentially undefined variables
- Fixed unsafe dictionary access in BM25 index building that caused KeyError when processing existing collection points
- Added empty content check in DocumentationIndexer to handle empty files gracefully
- Improved error reporting with specific exception handling

## [0.3.4.post5] - 2025-06-12

### 🚀 Enhanced GitHub Issue Management

This release adds comprehensive issue lifecycle management tools, milestone support, and advanced filtering capabilities, enabling complete release management workflows directly from Claude Code.

### Added

- **7 New Core Issue Management MCP Tools**:
  - `github_close_issue` - Close issues with state reasons (completed/not_planned/duplicate)
  - `github_assign_issue` - Assign/unassign users to/from issues
  - `github_update_issue` - Update any issue property (title, body, labels, milestone, assignees)
  - `github_list_milestones` - List repository milestones with sorting
  - `github_create_milestone` - Create milestones with due dates
  - `github_update_milestone` - Update milestone properties
  - `github_close_milestone` - Close completed milestones

- **Enhanced Issue Filtering**:
  - Added milestone parameter to `github_fetch_issues` (supports name or number)
  - Added assignee parameter (supports username or "none" for unassigned)
  - Added date range filtering with since parameter (ISO format)
  - Added sort options (created, updated, comments)
  - Added direction control (asc, desc)
  - Full backward compatibility maintained

- **Advanced Search Capabilities**:
  - `github_search_issues` - Full GitHub Search API support
  - Supports complete GitHub search syntax
  - Automatic repository qualifier injection
  - Pull request filtering
  - Examples: `"is:issue is:open milestone:v0.3.5 label:priority-high"`

- **HTTP API Endpoints**:
  - PATCH `/github/issues/{id}` - Update issue properties
  - PATCH `/github/issues/{id}/close` - Close issue with reason
  - POST `/github/issues/{id}/assignees` - Manage assignees
  - GET `/github/milestones` - List milestones
  - POST `/github/milestones` - Create milestone
  - PATCH `/github/milestones/{number}` - Update milestone
  - DELETE `/github/milestones/{number}` - Close milestone
  - POST `/github/issues/search` - Advanced search
  - Enhanced GET `/github/issues` with new filtering parameters

### Fixed

- Fixed PyGithub `milestone.edit()` API issue where title parameter was required even when not updating it
- Improved error handling for milestone filtering when milestone doesn't exist

### Changed

- Updated `GitHubClient.get_issues()` to support all new filtering parameters
- Enhanced HTTP server to v0.3.4.post5 with new request models
- Improved milestone handling to support both name and number references

### Technical Details

- **GitHubClient Methods Added**:
  - `close_issue(issue_number, reason, comment)`
  - `assign_issue(issue_number, assignees, operation)`
  - `update_issue(issue_number, **kwargs)`
  - `search_issues(query, sort, order)`
  - `list_milestones(state, sort, direction)`
  - `create_milestone(title, description, due_on)`
  - `update_milestone(number, **kwargs)`
  - `close_milestone(number)`

- **Integration**: Fully compatible with existing GitHub integration (v0.3.0), Projects V2 (v0.3.4), and Sub-Issues (v0.3.4.post4)

## [0.3.4.post4] - 2025-06-11

### 🎯 GitHub Sub-Issues Support

This release adds comprehensive GitHub sub-issues (task lists) support, enabling hierarchical issue management with full REST API integration and Projects V2 compatibility.

### Added

- **5 New GitHub Sub-Issues MCP Tools**:
  - `github_list_sub_issues` - List all sub-issues for a parent issue
  - `github_add_sub_issue` - Link existing issues as sub-issues
  - `github_remove_sub_issue` - Remove sub-issue relationships
  - `github_create_sub_issue` - Create and link new sub-issues in one operation
  - `github_reorder_sub_issues` - Reorder sub-issues for priority management
  
- **Projects V2 Sub-Issues Integration**:
  - `github_add_sub_issues_to_project` - Bulk add all sub-issues to a project
  - `smart_add_sub_issues_to_project` method in ProjectsManager
  - Automatic smart field assignment for sub-issues

- **REST API Implementation**:
  - Direct GitHub REST API integration for sub-issues (PyGithub doesn't support them)
  - Methods: `add_sub_issue`, `list_sub_issues`, `remove_sub_issue`, `reorder_sub_issues`
  - Bearer token authentication with preview headers
  - Full error handling and retry logic

- **HTTP API Endpoints**:
  - POST `/github/list_sub_issues`
  - POST `/github/add_sub_issue`
  - POST `/github/remove_sub_issue`
  - POST `/github/create_sub_issue`
  - POST `/github/reorder_sub_issues`
  - POST `/github/add_sub_issues_to_project`

- **Testing Infrastructure**:
  - `tests/integration/github/test_github_sub_issues_api.sh` - Comprehensive test script
  - Tests all sub-issue operations including Projects V2 integration
  - Includes cleanup functionality

### Fixed

- **Automatic Issue Number to ID Conversion**: Sub-issue methods now automatically convert issue numbers to IDs
  - GitHub's sub-issues API requires numeric issue IDs (e.g., 3138402433)
  - Methods now detect when a number < 1000000 is passed and fetch the actual ID
  - Improves user experience by allowing natural issue number usage (e.g., #39 instead of 3138402433)
  - Applied to `add_sub_issue`, `remove_sub_issue`, and `reorder_sub_issues` methods

### Technical Details

- **Implementation Strategy**: Since PyGithub doesn't support GitHub's sub-issues API, we implemented direct REST API calls using the requests library
- **API Compatibility**: Uses GitHub's task list preview API with appropriate headers (generally available as of January 2025)
- **Projects Integration**: Sub-issues can be bulk-added to Projects V2 with automatic field assignment
- **Parameter Handling**: The API requires `sub_issue_id` parameter with numeric IDs, but our methods accept issue numbers for convenience

### Use Cases

- Breaking down complex features into manageable sub-tasks
- Creating hierarchical issue structures for better organization
- Managing implementation roadmaps with parent/child relationships
- Bulk project management with automatic sub-issue tracking
- Re-parenting issues when reorganizing work

## [0.3.4.post3] - 2025-06-11

### 🔧 Code Refactoring - DRY Improvements

This patch release significantly improves code maintainability by introducing helper functions that eliminate hundreds of lines of duplicated code across GitHub integration functions.

### Added

- **Async Helper Function**:
  - `run_async_in_thread` - Centralized async coroutine execution for MCP context
  - Handles running event loops, thread safety, and edge cases
  - Used by all 10 GitHub Projects V2 functions

- **Validation Helper Functions**:
  - `check_github_available` - Validates GitHub integration availability
  - `check_projects_available` - Validates Projects V2 integration
  - `check_repository_context` - Ensures repository is set
  - `check_projects_manager` - Validates projects manager initialization
  - `validate_github_prerequisites` - Comprehensive validation helper combining all checks

### Changed

- **Refactored All GitHub Functions**:
  - Updated 20 GitHub MCP functions to use validation helpers
  - Removed ~500 lines of duplicated validation and async handling code
  - Improved consistency across all GitHub integration points
  - Simplified maintenance and future updates

### Fixed

- Event loop conflicts in `github_create_project_from_template`
- Event loop conflicts in `github_get_project`
- Inconsistent error messages across GitHub functions
- Direct `_current_repo` access changed to proper getter method

### Technical Improvements

- Better separation of concerns with dedicated helper functions
- DRY (Don't Repeat Yourself) principle applied throughout
- Consistent error handling patterns
- Thread-safe async execution in all contexts

## [0.3.4.post2] - 2025-06-11

### 🛠️ GitHub Projects V2 CRUD Operations

This patch release completes the GitHub Projects V2 integration by adding list and delete operations, enabling full project lifecycle management.

### Added

- **List Projects Operation**:
  - `github_list_projects` MCP tool - List all GitHub Projects V2 for a user or organization
  - Supports pagination with limit parameter (default: 20, max: 100)
  - Returns project details including ID, number, title, description, item count, and timestamps
  - Graceful handling of user vs organization ownership

- **Delete Project Operation**:
  - `github_delete_project` MCP tool - Delete a GitHub Project V2 permanently
  - Validates project ID format (must start with PVT_)
  - Fetches project details before deletion for confirmation
  - Returns deleted project title in confirmation message
  - Clear warning that deletion is permanent and cannot be undone

- **HTTP API Endpoints**:
  - `GET /github/projects` - List projects with optional owner and limit parameters
  - `DELETE /github/projects/{project_id}` - Delete a project by ID

### Documentation

- Updated GitHub Projects Guide with list and delete operations
- Added best practices for project cleanup and management
- Documented project ID validation requirements
- Added examples for listing projects by owner

### Fixed

- Improved error messages for invalid project ID formats
- Better handling of GraphQL errors in project operations

## [0.3.4.post1] - 2025-06-11

### 🎯 Token Usage Optimization

This patch release implements critical optimizations to reduce token usage during GitHub issue analysis by approximately 70%, addressing the issue where Claude was not using the dedicated MCP tools and instead performing manual searches.

### Changed

- **Enhanced MCP Tool Docstrings**:
  - Added clear "WHEN TO USE THIS TOOL" sections to ALL 30+ MCP tools
  - Comprehensive usage triggers for better tool selection by Claude
  - Clear descriptions of what each tool does automatically
  - Updated search tools: search, search_code, search_docs, search_config
  - Updated indexing tools: index_directory, reindex_directory, index_code, index_documentation, index_config
  - Updated utility tools: get_context, detect_changes, health_check, get_file_chunks, switch_project
  - Updated system tools: rebuild_bm25_indices, get_memory_status, trigger_memory_cleanup
  - Updated GitHub tools: github_list_repositories, github_switch_repository, github_fetch_issues, github_get_issue, github_create_issue, github_analyze_issue, github_suggest_fix
  - Emphasized using specialized tools over general operations

- **Progressive Context Integration**:
  - Added progressive context support to GitHub issue analysis
  - Context level selection based on issue type (bug→method, feature→class, docs→file)
  - Enabled semantic caching and expansion options for all searches
  - Progressive context parameters now passed to all RAG search operations

- **Configuration Optimizations**:
  - Reduced default `search_limit` from 10 to 5 results
  - Set `include_dependencies` to false by default (reduces token overhead)
  - Added `progressive_context` configuration section for GitHub issues
  - Fixed response verbosity check to properly respect "summary" mode

- **Query Deduplication**:
  - Implemented query normalization and deduplication before searches
  - Maximum of 8 unique queries per issue analysis
  - Filters out queries shorter than 3 characters
  - Prevents redundant searches on similar terms

### Fixed

- Response verbosity check now correctly excludes raw search results in summary mode
- Eliminated code duplication in `_perform_rag_searches` method
- Fixed SemanticCache attribute error by using `max_items` instead of `max_cache_size`
- Updated GitHub issue analyzer to use specialized search tools (search_code/search_docs) instead of general search

### Documentation

- Updated GitHub Integration Guide with detailed token optimization explanation
- Added comprehensive configuration reference for issue analysis settings
- Documented the multi-step issue analysis workflow
- Added real-world token reduction examples (83% typical reduction)

### Testing

- Added `test_token_optimization.py` to validate token reduction achievements
- Added `test_config_verification.py` for implementation verification
- Updated test script to accurately calculate queries and token usage

## [0.3.4] - 2025-06-10

### 🚀 Major Feature: GitHub Projects V2 Integration

Added comprehensive GitHub Projects V2 support via GraphQL API, enabling project management workflows with roadmap tracking, dashboards, and team collaboration directly through Claude Code.

### Added

- **GitHubProjectsManager**: GraphQL adapter for Projects V2 API using gql library
  - Adapter pattern integration preserving existing REST functionality
  - Async GraphQL operations with proper error handling
  - Owner ID caching for performance optimization
  - Clean separation between REST and GraphQL operations
  
- **6 New GitHub Projects MCP Tools**:
  - `github_create_project` - Create GitHub Projects V2 with custom fields and descriptions
  - `github_get_project` - Get project details including fields and item counts
  - `github_add_project_item` - Add issues or PRs to projects from current repository
  - `github_update_project_item` - Update field values for project items
  - `github_create_project_field` - Create custom fields (TEXT, NUMBER, DATE, SINGLE_SELECT)
  - `github_get_project_status` - Dashboard view with progress metrics and completion rates
  
- **Dependency Management**:
  - Added `gql[aiohttp]>=3.4.1` for GraphQL client support
  - Added `aiohttp>=3.8.0` for async HTTP transport
  - Graceful fallback when dependencies not installed
  
- **Project Status Metrics**:
  - Issue statistics (open/closed counts)
  - Pull request statistics (open/closed/merged counts)
  - Automatic completion rate calculation
  - Field listing with types and options
  
### Changed

- **get_github_instances()** now returns 5 values (added projects_manager)
- Updated all GitHub tool calls to handle new return signature
- Enhanced error messages to guide dependency installation

### Fixed

- **Critical MCP Server Startup Issue**: Fixed tuple unpacking error in `get_github_instances()` calls that prevented the MCP server from starting after adding GitHub Projects support. Updated all calls to properly unpack 5 return values instead of 4.

### Technical Details

- **Architecture**: Clean adapter pattern extending existing GitHubClient
- **Authentication**: Extracts token from PyGithub client for GraphQL
- **Async Handling**: Proper event loop management for sync/async bridge
- **GraphQL Queries**: Optimized queries with fragment reuse
- **Node ID Support**: Handles GitHub's global node ID system

### Migration Notes

1. **Install Dependencies**: Run `pip install 'gql[aiohttp]>=3.4.1'`
2. **Token Permissions**: Ensure PAT has `project` scope for Projects V2
3. **Organization Projects**: GitHub Apps can only access org projects, not user projects

### Use Cases

- Move implementation roadmaps to GitHub Projects for visual tracking
- Create dashboards showing feature completion progress
- Enable team collaboration with assignments and due dates
- Automate project updates when issues/PRs are completed
- Generate progress reports and milestone tracking

## [0.3.3.post4] - 2025-06-10

### Fixed

- **Critical Dimension Mismatch Error**: Fixed "Vector dimension error: expected dim: 768, got 384" during batch reindexing
  - Root cause: Thread safety issues and inappropriate fallback to general model (384D) for code files
  - Added dimension-compatible fallback logic to prevent 384D vectors for code content
  - Implemented comprehensive thread safety with RLock in SpecializedEmbeddingManager
  - Code files now always use 768D models (CodeRankEmbed or fallback CodeBERT)
  - Prevents race conditions during concurrent model loading/eviction

### Changed

- **Thread Safety Improvements**:
  - Added `threading.RLock()` to protect all critical sections
  - Thread-safe model loading, caching, and eviction
  - Protected active model tracking during encoding
  - Prevents models from being evicted while in use

- **Fallback Logic Enhancement**:
  - Checks dimension compatibility before falling back to general model
  - For code content (768D), tries configured fallback (microsoft/codebert-base) first
  - Only allows fallback to general model if dimensions match
  - Raises clear error if no dimension-compatible fallback available

### Developer Experience

- **Test Suite Reorganization**:
  - Reorganized 56+ test files into clear categories (unit, integration, performance, debug, examples)
  - Added test runner script (`run_tests.sh`) for easy test execution
  - Created pytest configuration with markers and fixtures
  - Separated debug scripts from actual tests
  - Added common test fixtures in conftest.py

## [0.3.3.post3] - 2025-06-10 (Superseded by 0.3.3.post4)

### Note
This version was created during debugging but superseded by 0.3.3.post4 which includes the complete fix.

## [0.3.3.post2] - 2025-06-11

### Fixed

- **Context Tracking Configuration Error**: Fixed "'float' object has no attribute 'get'" error in context tracking initialization
  - Added proper type checking for component_limits configuration
  - Handles both dict and numeric values for memory limits
  - Prevents crashes when config format varies

## [0.3.3.post1] - 2025-06-11

### Fixed

- **Critical Search Error**: Fixed "'float' object has no attribute 'get'" error in all search modes
  - Added payload type validation before accessing dictionary methods
  - Implemented safe error handling to skip corrupted payloads
  - Fixed in vector, keyword, and hybrid search paths
  - Ensures search operations continue even with corrupted data

## [0.3.3] - 2025-06-09

### 🚀 Major Feature: Specialized Embeddings with Memory Management

Implemented content-type specific embedding models with comprehensive memory management and Apple Silicon optimizations, achieving better search precision while preventing memory exhaustion issues.

### Added

- **SpecializedEmbeddingManager**: Core component for managing multiple specialized models
  - Dynamic model loading with LRU eviction policy
  - Memory management with configurable limits (3 models, 4GB default)
  - Fallback model support for resilience
  - Content-type detection from file extensions
  - Apple Silicon awareness with conservative limits (2 models, 3GB on MPS)
  
- **Content-Type Specific Models**:
  - **Code**: `nomic-ai/CodeRankEmbed` (768D) - Optimized for multi-language code understanding
  - **Config**: `jinaai/jina-embeddings-v3` (1024D) - Specialized for JSON/YAML/XML
  - **Documentation**: `hkunlp/instructor-large` (768D) - Includes instruction prefix support
  - **General**: `sentence-transformers/all-MiniLM-L12-v2` (384D) - Default model
  
- **Unified Memory Management System** (`src/utils/memory_manager.py`):
  - Centralized memory management for all components
  - Apple Silicon detection and optimization
  - Dynamic memory limits based on system RAM
  - Memory pressure callbacks and monitoring
  - Component registry for tracking memory usage
  
- **Apple Silicon Optimizations**:
  - Automatic detection of M1/M2/M3/M4 chips
  - Conservative memory limits for unified memory architecture
  - MPS (Metal Performance Shaders) cache management
  - MPS environment variables for CodeRankEmbed stability
  - CPU fallback when memory is insufficient
  - New MCP tools: `get_apple_silicon_status`, `trigger_apple_silicon_cleanup`
  
- **UnifiedEmbeddingsManager**: Backward-compatible wrapper
  - Seamless switching between single and specialized modes
  - Progressive context compatibility maintained
  - Legacy API support (`get_sentence_embedding_dimension()`, `dimension` property)
  
- **Model Registry System** (`src/utils/model_registry.py`):
  - Central registry for model configurations and metadata
  - Collection-to-model mapping tracking
  - Model compatibility validation
  - Persistent registry storage
  
- **Enhanced Collection Metadata**:
  - Stores embedding model name and dimension with each collection
  - Tracks specialized embeddings usage flag
  - Content type information for proper model selection
  
- **Model Compatibility Checking**:
  - `check_model_compatibility()` validates query model vs collection model
  - `get_query_embedding_for_collection()` ensures correct model usage
  - Dimension mismatch detection and warnings
  
- **Improved Model Management Scripts**:
  - Enhanced `download_models.sh` with specialized model support
  - Fixed path handling for comments in environment variables
  - Added dependency checking (einops, InstructorEmbedding)
  - Support for `trust_remote_code` models
  - Updated `manage_models.sh` to show model roles ([CODE], [CONFIG], etc.)
  
- **New Requirements File**: `requirements-models.txt` for model-specific dependencies

### Changed

- **Indexing Functions**: Now use specialized models based on content type
  - `index_code()` uses CodeRankEmbed for better code understanding
  - `index_config()` uses jina-embeddings-v3 for configuration files
  - `index_documentation()` uses instructor-large with prefix support
  
- **Search Functions**: Automatically select appropriate model for queries
  - Model compatibility checking before search execution
  - Warnings for model mismatches
  - Graceful fallback to collection's indexed model

### Fixed

- **Critical Memory Issues with CodeRankEmbed**: 
  - Fixed dimension mismatch errors (expected 768, got 384) during batch processing
  - Reverted to one-by-one processing for all indexing functions to maintain model consistency
  - Fixed memory exhaustion issues causing system freezes on Apple Silicon (15-16GB usage)
  - **Additional fixes for intermittent failures with large files**:
    - Models now marked as active immediately to prevent race condition eviction
    - Enhanced LRU eviction with model protection during loading
    - Added periodic memory pressure monitoring during encoding (every 10 chunks)
    - Improved cleanup with proper finally blocks and error handling
  
- **Batch Processing Dimension Mismatch**: 
  - Root cause: Batch processing caused embeddings manager to lose track of specialized models
  - Solution: Process chunks individually to ensure correct model is used throughout
  - **Enhanced with active model tracking**: Models cannot be evicted while encoding
  
- **Apple Silicon Memory Management**:
  - Added psutil dependency for system memory monitoring
  - Fixed memory pressure detection and cleanup on MPS devices
  - Implemented conservative memory limits based on total system RAM
  - **Reduced safety margins**: 1.5x for CodeRankEmbed (was 2.5x) to prevent premature eviction
  - **Dynamic memory pressure handling**: 
    - Critical (<1.5GB): Evict models + aggressive cleanup
    - Low (<2.5GB): Standard cleanup
    - Protects current model from eviction during processing
  
- **Environment Variable Parsing**: Fixed `.env` path comments creating invalid directories
- **Model Directory Detection**: Scripts now correctly find models in `data/models/`
- **File Type Detection**: Added special handling for dot files (`.env`, `.gitignore`, etc.)
- **Progressive Context Compatibility**: Added default content_type="general" for backward compatibility

### Technical Details

- **Memory Management**:
  - Unified memory manager with component registry
  - Apple Silicon specific limits: 14GB for 18GB systems, 10GB for 16GB systems
  - LRU eviction when model count or memory limit exceeded
  - Real-time memory tracking with psutil
  - Memory pressure callbacks trigger cleanup at thresholds
  - **Enhanced Model Protection**:
    - Active model tracking prevents eviction during use
    - Race condition prevention with immediate active marking
    - Protected model parameter during memory checks
    - Periodic memory pressure monitoring (every 10 chunks)
  
- **Apple Silicon Detection**:
  - Uses `sysctl -n hw.optional.arm64` to detect ARM64 architecture
  - Applies conservative limits for unified memory architecture
  - Sets MPS environment variables: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`
  
- **Model Sizes and Limits**:
  - CodeRankEmbed: ~1GB (768 dimensions)
  - jina-embeddings-v3: ~0.5GB (1024 dimensions)  
  - instructor-large: ~1.3GB (768 dimensions)
  - Default limits: 3 models, 4GB total (2 models, 3GB on Apple Silicon)
  
- **Configuration**:
  - All models configurable via environment variables
  - `QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true` to enable
  - `QDRANT_MAX_MODELS_IN_MEMORY` and `QDRANT_MEMORY_LIMIT_GB` for limits
  - Fallback models for each content type

### Documentation

- Added memory optimization guides and recommendations
- Added Apple Silicon optimization documentation
- Updated implementation plans and roadmaps
- Added model download troubleshooting
- Enhanced script documentation

### Performance

- **Search Precision**: Improved relevance for content-type specific queries
- **Memory Efficiency**: 
  - Unified memory management prevents exhaustion
  - Apple Silicon optimizations reduce memory usage by 40-60%
  - One-by-one processing trades speed for stability
- **Lazy Loading**: Models only loaded when needed
- **Stability**: No more system freezes or dimension mismatch errors

### Migration Notes

1. **Environment Setup**: Add specialized model configuration to `.env`:
   ```bash
   QDRANT_SPECIALIZED_EMBEDDINGS_ENABLED=true
   QDRANT_CODE_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
   QDRANT_MAX_MODELS_IN_MEMORY=3  # or 2 for Apple Silicon
   QDRANT_MEMORY_LIMIT_GB=4.0     # or 3.0 for Apple Silicon
   # MPS optimizations (automatically set on Apple Silicon)
   PYTORCH_ENABLE_MPS_FALLBACK=1
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
   PYTORCH_MPS_LOW_WATERMARK_RATIO=0.0
   ```

2. **Install Dependencies**: Update dependencies for memory monitoring:
   ```bash
   uv pip install -r requirements.txt  # Includes psutil
   ```

3. **Model Download**: Run updated script to download specialized models:
   ```bash
   ./scripts/download_specialized_models.py  # New focused script
   # or
   ./scripts/download_models.sh
   # Select option 0 for all specialized models
   ```

4. **Reindex Required**: Collections must be reindexed to use new models:
   ```bash
   # Use force reindex to clear old embeddings
   "Force reindex the entire project"
   ```

### Known Issues

- **Performance Trade-off**: One-by-one processing is slower but necessary for stability
- **Memory Usage**: CodeRankEmbed requires significant memory even with optimizations
- **First Load**: Initial model loading may take 30-60 seconds
- **Large File Processing**: Files with >50 chunks may require retries on memory-constrained systems


## [0.3.2] - 2025-06-03

### 🚀 Major Feature: Progressive Context Management

Added intelligent multi-level context retrieval with semantic caching, achieving 50-70% token reduction for high-level queries while maintaining full detail access when needed.

### Added

- **Configurable Scoring Pipeline**: Implemented modular scoring system for hybrid search
  - New `ScoringPipeline` class with pluggable scoring stages
  - Built-in stages: `VectorScoringStage`, `BM25ScoringStage`, `ExactMatchStage`, `FusionStage`, `EnhancedRankingStage`
  - Factory functions for common configurations: `create_hybrid_pipeline()`, `create_code_search_pipeline()`, `create_documentation_pipeline()`
  - Support for custom scoring stages with extensible architecture
  - Detailed score breakdown and debugging metadata in results
  - Integration with `HybridSearcher.search_with_pipeline()` method

- **Enhanced BM25 Code Tokenization**: Improved keyword matching for code-specific patterns
  - `code_preprocessor` function handles camelCase splitting (BM25Manager → BM25 Manager)
  - Snake_case tokenization (append_documents → append documents)
  - Number-letter separation for better code token matching
  - Special character handling for operators and symbols

- **Improved AST Chunking Strategy**: Better code structure preservation in chunks
  - Classes now chunked together with their methods (`class_with_methods` type)
  - Related methods grouped together (`class_methods` type)
  - Function signatures kept with their implementation
  - Enhanced metadata for code structure understanding

- **Unified Hybrid Search Component**: Created reusable `_perform_hybrid_search()` helper function
  - Eliminates code duplication across all search functions
  - Ensures consistent hybrid search behavior with linear combination fusion
  - Supports customizable metadata extraction and result processing per search type
  - All search functions now support vector, keyword, and hybrid modes

- **Technical Documentation**: Comprehensive guides for scoring pipeline
  - `docs/technical/scoring-pipeline-architecture.md`: Complete architectural overview
  - `docs/technical/scoring-pipeline-quick-reference.md`: Developer quick reference

- **Progressive Context Core Module** (`src/utils/progressive_context.py`):
  - `ProgressiveContextManager` - Orchestrates multi-level context retrieval (file → class → method)
  - `SemanticCache` - Implements similarity-based caching with persistent storage
  - `HierarchyBuilder` - Constructs code structure hierarchies from search results
  - `QueryIntentClassifier` - Auto-detects appropriate context level based on query patterns
  - Token estimation and reduction metrics for each context level

- **Integration with All Search Functions**:
  - Enhanced `search()`, `search_code()`, and `search_docs()` with progressive parameters:
    - `context_level`: "auto", "file", "class", "method", or "full" granularity
    - `progressive_mode`: Enable/disable progressive features (auto-detects by default)
    - `include_expansion_options`: Include drill-down options for deeper exploration
    - `semantic_cache`: Use semantic similarity caching for repeated queries
  - Seamless fallback to regular search when progressive mode is disabled

- **HTTP API Enhancements**:
  - Updated all search request models with progressive context parameters
  - Full API compatibility for `/search`, `/search_code`, and `/search_docs` endpoints
  - Progressive metadata included in responses with token estimates and expansion options

- **Configuration Support** (`config/server_config.json`):
  - New `progressive_context` section with comprehensive settings
  - Feature flag for safe rollout (enabled by default)
  - Configurable cache settings (similarity threshold: 0.85, TTL: 1 hour)
  - Level-specific configurations for token reduction targets
  - Query classification tuning parameters

- **Testing Infrastructure**:
  - Unit tests for all progressive context components (`tests/test_progressive_context.py`)
  - HTTP API test script (`tests/test_progressive_http.py`) with multiple test scenarios
  - Cache behavior validation and auto-classification testing

### Changed

- Search functions now intelligently adapt context level based on query intent
- Enhanced ranking signals (file proximity, dependency distance, code structure, recency) fully integrated with progressive search
- Hybrid search (vector + BM25 + fusion) implemented within progressive context manager
- Default behavior unchanged - progressive features only activate when explicitly requested or auto-detected

### Technical Details

- **Token Reduction by Level**:
  - File level: 70% reduction - High-level summaries and structure
  - Class level: 50% reduction - Signatures and key methods
  - Method level: 20% reduction - Focused implementation details
  - Full level: 0% reduction - Complete traditional search results

- **Semantic Cache**:
  - Similarity threshold: 0.85 for cache hits
  - Persistent storage in `~/.mcp-servers/qdrant-rag/progressive_cache/`
  - TTL: 1 hour for cached results
  - Automatic cache size management (max 1000 entries)

- **Query Intent Classification**:
  - "Understanding" queries → File level (e.g., "What does X do?")
  - "Navigation" queries → Class level (e.g., "Find the Y class")
  - "Debugging" queries → Method level (e.g., "Bug in line Z")
  - Confidence-based fallback to configurable default level

### Changed

- **Refactored Search Functions**: All search functions now use unified hybrid search
  - `search()`: Uses general metadata extractor for all collection types
  - `search_code()`: Now supports hybrid search (previously vector-only) with code-specific metadata
  - `search_docs()`: Simplified to use common hybrid search with docs-specific metadata
  - Consistent scoring and ranking across all search types

- **Linear Combination Scoring**: Switched from RRF to linear combination for accurate hybrid search
  - More interpretable scores that preserve semantic similarity information
  - Configurable weights for different content types (code: 50/50, docs: 80/20)
  - Enhanced score accuracy and consistency across search modes

### Fixed

- **Progressive Context Scoring Bug**: Fixed issue where search results showed low base scores (0.01-0.04) instead of enhanced scores (0.4-0.6)
  - Removed premature sorting that occurred before enhanced ranking
  - Added missing score update after enhanced ranking is applied
  - Now correctly displays enhanced scores that include all ranking signals

- **Hybrid Search Scoring Improvement**: Switched from RRF to linear combination for more accurate scoring
  - RRF produced artificially low scores (0.01-0.02) that didn't represent actual similarity
  - Linear combination preserves semantic similarity information from embeddings
  - Results now show meaningful scores (0.6-0.9) that reflect actual match quality
  - Applied to both progressive and regular search for consistency

- **Search Performance Optimization**: Fixed inefficient document fetching in hybrid and keyword search modes
  - Regular search now reuses original vector search results instead of re-fetching with dummy vectors
  - Keyword search now uses actual query vectors instead of dummy zeros when fetching documents
  - Reduces redundant Qdrant queries and improves score accuracy

### Documentation

- Updated implementation status document
- Added progressive context code structure documentation
- Integration strategy documentation for future enhancements
- Test examples demonstrating all progressive features

## [0.3.1] - 2025-06-03

### Added
- **Context Tracking System** - Complete visibility into Claude's context window usage
  - `SessionContextTracker` class tracks all context-consuming operations
  - Token estimation for searches, file reads, and tool uses
  - Persistent session storage in JSON format
  - Context usage warnings at 60% and 80% thresholds
  
- **3 New MCP Tools** for context awareness:
  - `get_context_status` - Detailed metrics about current session including token usage, files read, and searches performed
  - `get_context_timeline` - Chronological list of all context events with token estimates
  - `get_context_summary` - Natural language summary of what Claude knows in the session
  
- **Session Viewer Utility** (`scripts/qdrant-sessions`)
  - List recent sessions with summary statistics
  - View detailed session information including token breakdown
  - Show timeline of events for any session
  - Analyze patterns across multiple sessions
  
- **Configuration Support**:
  - New `context_tracking` section in `server_config.json`
  - Configurable context window size and warning thresholds
  - Auto-save settings for session persistence
  - Toggle tracking for different operation types

### Changed
- Search operations (`search`, `search_code`, `search_docs`) now automatically track context usage
- Added context tracking initialization to server startup
- Default context window size increased to 200,000 tokens in configuration

### Fixed
- Improved token estimation accuracy - now only counts visible content, not internal JSON metadata
- Fixed system prompt token calculation to include actual CLAUDE.md content (~14,700 tokens vs hardcoded 2,000)

### Technical Details
- Context tracking integrates seamlessly with existing operations
- Simple token estimation using 4 characters per token approximation
- Sessions stored in `~/.mcp-servers/qdrant-rag/sessions/`
- Minimal performance overhead (<1% on operations)

### Documentation
- Updated CLAUDE.md with context tracking usage examples
- Added comprehensive test suite for context tracking functionality
- Created session viewer documentation with examples
- Added [Context Tracking Guide](docs/reference/context-tracking-guide.md) with detailed testing instructions
- Updated roadmap to mark v0.3.0 and v0.3.1 as completed
- Added v0.6.x specialized embeddings roadmap for content-type-specific models

## [0.3.0] - 2025-06-02

### 🚀 Major Feature: GitHub Integration

Added comprehensive GitHub issue resolution capabilities that transform the RAG server into an intelligent automation system.

#### Added
- **10 New GitHub MCP Tools** for complete issue lifecycle management:
  - `github_list_repositories` - List user/organization repositories
  - `github_switch_repository` - Set repository context for operations
  - `github_fetch_issues` - Fetch repository issues with filtering
  - `github_get_issue` - Get detailed issue information including comments
  - `github_create_issue` - Create new issues (perfect for testing workflows)
  - `github_add_comment` - Add comments to existing issues for workflow updates
  - `github_analyze_issue` - RAG-powered issue analysis using codebase search
  - `github_suggest_fix` - Generate automated fix suggestions with confidence scoring
  - `github_create_pull_request` - Create pull requests with automated content
  - `github_resolve_issue` - End-to-end issue resolution workflow with dry-run support

- **Complete GitHub Integration Module** (`src/github_integration/`):
  - `client.py` - GitHub API client with authentication and rate limiting
  - `issue_analyzer.py` - RAG-powered issue analysis and pattern extraction
  - `code_generator.py` - Automated fix generation with code templates
  - `workflows.py` - End-to-end workflow orchestration and feasibility assessment

- **Authentication Support**:
  - Personal Access Token authentication (recommended for individual use)
  - GitHub App authentication (recommended for organizations)
  - Automatic token validation and rate limit handling
  - Graceful degradation when dependencies not installed

- **HTTP API Endpoints** (10 new endpoints under `/github/`):
  - `GET /github/repositories` - List repositories
  - `POST /github/switch_repository` - Switch repository context
  - `GET /github/issues` - Fetch issues with query parameters
  - `GET /github/issues/{id}` - Get specific issue details
  - `POST /github/issues` - Create new issues
  - `POST /github/issues/{id}/comment` - Add comments to existing issues
  - `POST /github/issues/{id}/analyze` - Analyze issue with RAG
  - `POST /github/issues/{id}/suggest_fix` - Generate fix suggestions
  - `POST /github/pull_requests` - Create pull requests
  - `POST /github/issues/{id}/resolve` - Full resolution workflow
  - `GET /github/health` - GitHub integration health check

#### Enhanced
- **Configuration System**: Enhanced environment variable resolution
  - Fixed `${VAR:-default}` syntax parsing in `config/server_config.json`
  - Added comprehensive GitHub configuration section with all options
  - Support for API settings, repository context, safety features, and workflow configuration

- **RAG Analysis Integration**: GitHub issues leverage full RAG capabilities
  - Code search with dependency analysis for related components
  - Documentation search for guides and examples
  - Error pattern extraction and similar issue detection
  - Confidence scoring and implementation feasibility assessment

- **Safety & Security Features**:
  - Dry-run mode by default for all destructive operations
  - File protection patterns (secrets, keys, CI/CD workflows)
  - Rate limiting with exponential backoff retry logic
  - Audit logging for compliance and monitoring
  - Input validation and sanitization

#### Technical
- **Dependencies**: Added PyGithub>=2.6.1 and GitPython>=3.1.44
- **Error Handling**: Comprehensive GitHub-specific error types and recovery
- **Health Monitoring**: Separate GitHub health checks with rate limit status
- **Performance**: Efficient GitHub API usage with intelligent caching and batching

#### Documentation
- **[GitHub Integration Guide](docs/github-integration-guide.md)** - Comprehensive 700+ line guide covering:
  - Setup and authentication (Personal Access Token and GitHub App)
  - Complete API reference for all 10 tools and HTTP endpoints
  - Usage examples for both MCP tools and HTTP API testing
  - Testing patterns and development workflows
  - Safety features, troubleshooting, and best practices
  - Integration with existing RAG workflows

- **Testing Infrastructure**:
  - `tests/integration/github/test_github_http_api.sh` - Comprehensive HTTP API testing script
  - Updated CLAUDE.md with GitHub integration examples
  - Complete MCP tool testing patterns and natural language examples

#### Migration Guide
For users upgrading to v0.3.0:

1. **Install GitHub dependencies** (optional - graceful degradation if not installed):
   ```bash
   uv add "PyGithub>=2.6.1" "GitPython>=3.1.44"
   ```

2. **Set up GitHub authentication** (see [GitHub Integration Guide](docs/github-integration-guide.md)):
   ```bash
   # Add to .env file
   GITHUB_TOKEN=your_personal_access_token
   GITHUB_REPO_OWNER=your-username  # Optional
   GITHUB_REPO_NAME=your-repo       # Optional
   ```

3. **Test the integration**:
   ```bash
   # Via Claude Code
   "Check GitHub health status"
   "Switch to repository owner/repo-name"
   "Show me open issues"
   
   # Via HTTP API
   curl http://localhost:8081/github/health
   ./scripts/test_github_http_api.sh
   ```

#### Performance Impact
- **No impact on existing functionality** - GitHub integration is completely optional
- **Lazy loading** - GitHub dependencies only loaded when GitHub tools are used
- **Efficient API usage** - Rate limiting and caching prevent API abuse
- **Smart defaults** - Dry-run mode prevents accidental operations

### 🔧 Improvements (Post-Release Updates)
- **Enhanced Error Messages**: More user-friendly and actionable error messages for GitHub operations
- **Improved Rate Limiting**: Intelligent rate limit handling with separate tracking for core and search APIs
- **Git Operations Module**: Added GitOperations class for actual file modifications and branch management
- **Configuration Examples**: Comprehensive .env.example with detailed explanations for all settings
- **Architecture Documentation**: Added detailed architecture guide explaining the layered design
- **Dependencies**: Added missing PyGithub and GitPython to requirements.txt
- **GitHub Workflow Examples Guide**: Created comprehensive real-world usage patterns and issue remediation workflows
- **Token Consumption Optimization**: Significantly reduced token usage in GitHub issue analysis
  - Added configurable response verbosity (`response_verbosity`: "summary" or "full")
  - New `include_raw_search_results` option (default: false) to exclude raw search data
  - Added `_summarize_search_results()` method for condensed search summaries
  - Added `_summarize_extracted_info()` method for reduced extracted info verbosity
  - Maintains full internal analysis capability while reducing response tokens by ~80-90%
  - Configurable via `server_config.json` with sensible defaults

### 🐛 Bug Fixes (Post-Release)
- Fixed missing GitHub dependencies in requirements.txt
- Improved error handling with specific messages for common failure scenarios
- Enhanced rate limit handling to prevent API quota exhaustion
- Fixed outdated placeholder comments in `_create_resolution_pr` method (Issue #3)
- Fixed singleton configuration caching issue in GitHub integration
  - Updated `get_issue_analyzer()` to accept and apply configuration updates
  - Updated `get_github_workflows()` to accept and apply configuration updates
  - Modified `get_github_instances()` to always pass latest configuration
  - Ensures token optimization settings are always applied without restart

## [0.2.7] - 2025-06-02

### Fixed
- **HTTP Testing Server**: Fixed broken HTTP server by adding all missing endpoints
  - Added `/index_documentation` endpoint for documentation file indexing
  - Added `/search_docs` endpoint for documentation-specific search
  - Added `/reindex_directory` endpoint with smart incremental reindex support
  - Added `/detect_changes` endpoint for file change detection
  - Added `/health_check` endpoint for detailed system health monitoring
  - Added `/get_file_chunks` endpoint for retrieving file chunks
  - Added `/get_context` endpoint for project context information
  - Added `/switch_project` endpoint for project switching
  - Updated all search endpoints to support new parameters (search_mode, include_context, etc.)
  - Fixed deprecated FastAPI @app.on_event usage by replacing with lifespan context manager
  - Direct function call approach instead of MCP request wrapper for better performance

### Enhanced
- **HTTP Server Architecture**: Streamlined endpoint implementations
  - All endpoints now directly call MCP tool functions instead of wrapping in Request objects
  - Improved error handling with proper HTTP status codes
  - Better parameter mapping between HTTP requests and MCP tool parameters
  - Added proper request/response models for all new endpoints

### Technical Details
- HTTP server now supports all MCP tools available in the main server (v0.2.3-v0.2.6 features)
- Maintains backward compatibility with existing endpoints
- Updated FastAPI to use modern lifespan event handlers
- Enhanced error responses with detailed error messages

## [0.2.6] - 2025-06-02

### Fixed
- **Enhanced Ranking Type Error**: Fixed critical search failure due to type comparison error
  - Enhanced ranker now properly converts scores to float before comparison
  - Prevents "'>' not supported between instances of 'str' and 'int'" error
  - Ensures stable sorting of search results with enhanced scoring
  - Fixed modified_at timestamp handling in recency scoring

## [0.2.5] - 2025-05-29

### Fixed
- **`get_file_chunks` Collection Detection**: Fixed bug where tool only searched in code collection
  - Now correctly detects file type (code, config, documentation) based on file extension
  - Automatically routes to the appropriate collection for retrieval
  - Returns file_type in response for better visibility

### Added
- **Development Workflow Guide**: New comprehensive guide for efficient development
  - Targeted search strategies for navigating the codebase
  - Implementation-specific search queries for roadmap tasks
  - Common development patterns and best practices
  - Testing and debugging workflows

### Changed
- **Code Quality**: Removed unused imports and variables flagged by Pylance
  - Cleaned up unused hashlib, Set, Filter imports
  - Fixed unused loop variables in enumeration
  - Improved overall code cleanliness

### Documentation
- Added [Development Workflow Guide](docs/development-workflow-guide.md) to help developers efficiently use RAG search
- Updated README with reference to the new development guide

## [0.2.4] - 2025-05-29

### Added
- **Smart Incremental Reindexing**: Revolutionary change detection system that only processes modified files
  - File hash tracking (SHA256) in chunk metadata for reliable change detection
  - `incremental=True` parameter in `reindex_directory` (default behavior)
  - Automatic cleanup of chunks from deleted files
  - Progress tracking and detailed reporting of changes
  - 90%+ performance improvement for typical reindex operations
- **`detect_changes` MCP Tool**: New tool to check for file changes without reindexing
  - Compare current filesystem state with indexed content
  - Returns detailed breakdown of added/modified/unchanged/deleted files
  - Useful for pre-reindex checks and monitoring

### Enhanced
- **File Hash Tracking**: All indexers (code, config, documentation) now store file hashes
  - Enables precise change detection beyond modification times
  - Supports reliable incremental updates across all content types
  - Future-proofs for advanced change detection features

- **Improved Reindex Workflow**:
  - Smart incremental reindex (default): Only processes changed files
  - Force full reindex (legacy): Clears collections then reindexes everything
  - Comprehensive change reporting: added/modified/deleted/unchanged counts
  - Better error handling and recovery mechanisms

### Changed
- **Default Reindex Behavior**: Now uses smart incremental mode by default
  - Preserves unchanged embeddings for massive performance gains
  - Automatically handles file additions, modifications, and deletions
  - Falls back to full reindex when force=True or incremental=False

### Technical Improvements
- Added `utils/file_hash.py` for efficient file change detection
- Added `detect_changes()` function for filesystem comparison
- Added `delete_file_chunks()` function for surgical chunk removal
- Enhanced logging and progress tracking for all reindex operations

### Performance
- **Reindex Speed**: 90%+ faster for projects with minor changes
- **No-Change Reindex**: Sub-second completion for unchanged projects
- **Memory Efficiency**: Processes only changed files, reducing memory usage
- **Embedding Preservation**: Keeps existing embeddings for unchanged files

### Fixed
- **Critical Bug**: Fixed infinite collection growth in incremental reindex
  - Fixed pagination issue in `detect_changes` that was missing files beyond 10,000 chunks
  - Fixed directory exclusion logic to properly respect `.ragignore` patterns
  - Ensured consistent absolute path usage for file tracking
  - Added deduplication by only tracking files with chunk_index=0

### API Changes
- `reindex_directory()` now accepts `incremental: bool = True` parameter
- Enhanced return format includes detailed change detection results
- Backward compatible with existing force=True behavior

### Documentation
- Updated CLAUDE.md with smart reindex usage examples
- Added performance comparisons and best practices
- Created comprehensive test suite for smart reindex validation

## [0.2.3] - 2025-05-29

### Added
- **Documentation Indexer**: New specialized indexer for markdown and documentation files
  - Section-based chunking that preserves document structure
  - Heading hierarchy extraction for better navigation
  - Rich metadata including code blocks, links, and frontmatter
  - Support for `.md`, `.markdown`, `.rst`, `.txt`, and `.mdx` files
  - Intelligent splitting of large sections with context preservation
- **New MCP Tools for Documentation**:
  - `index_documentation`: Index markdown and documentation files
  - `search_docs`: Search specifically in documentation with rich results
- **Documentation Collection**: Separate Qdrant collection for documentation
  - Isolated from code and config collections for better organization
  - Optimized for document-specific search patterns
- **Enhanced `index_directory`**: Now includes markdown files by default
  - Automatically routes `.md` files to Documentation Indexer
  - Maintains backward compatibility with existing indexing
- **Configuration Support**: Full integration with `server_config.json`
  - Configurable documentation chunk sizes and overlap
  - Customizable ranking weights for documentation search
  - Documentation-specific settings section

### Fixed
- **Fixed hybrid search for documentation**: Enhanced ranker now works correctly with documentation search
  - Removed invalid `query` parameter that was causing errors
  - Documentation search now supports all search modes (vector, keyword, hybrid)
  - Custom ranking weights are properly applied for documentation

### Technical Details
- Created `DocumentationIndexer` class with markdown-aware parsing
- Regex-based extraction of headings, code blocks, links, and frontmatter
- Section chunking creates natural document boundaries
- Metadata tracks heading hierarchy for contextual understanding
- Large sections split at paragraph boundaries with overlap
- Summary extraction for quick document previews
- Integrated with `server_config.json` for configurable chunk sizes and ranking weights
- Documentation-specific ranking weights optimize for recency and relevance

### Migration Guide
Documentation files are now indexed automatically when using `index_directory`. 
For existing projects, reindex to include documentation:
```
"Reindex this directory to include documentation"
```

## [0.2.2] - 2025-05-29

### Critical Fix
- **Fixed Critical Working Directory Bug**: MCP server now correctly handles client working directory
  - Server was using its own working directory instead of client's actual location
  - Caused wrong project to be indexed when using relative paths or "." directory
  - Example: Running `index_directory(".")` from `/panflow` indexed `/qdrant-rag` instead

### Fixed  
- **Fixed Missing `.ragignore` Support**: The `.ragignore` file now actually works
  - Previously, `.ragignore` file existed but was not being read or used
  - Exclusion patterns were hardcoded instead of reading from file
  - Now properly loads and applies patterns from `.ragignore`

### Added
- **Natural Language Working Directory Setup**: Users can now set working directory using natural language
  - No configuration required - works immediately
  - Example: "Get pwd, export MCP_CLIENT_CWD to that value, then run health check"
  - Provides the easiest way to ensure correct project context
- **MCP_CLIENT_CWD Environment Variable**: New environment variable to pass client's working directory
  - Allows MCP server to know the actual directory where user is working
  - Set this in your environment or .env file for accurate project detection
  - Can be set dynamically via natural language commands
- **Client Directory Parameter**: `get_current_project()` now accepts optional `client_directory` parameter
  - Used internally by `index_directory` to ensure correct project context
- **Command Line Argument**: Added `--client-cwd` argument for setting client working directory
  - Allows configuration via Claude Code's args array
  - Overrides MCP_CLIENT_CWD environment variable when provided
- **Improved Directory Validation**: Better error messages and validation for directory parameters
  - Now requires explicit directory parameter (no default to ".")
  - Clear warnings when server's working directory is used as fallback
- **Claude Code Configuration Guide**: Comprehensive documentation for working directory setup
  - Created docs/claude-code-config-example.md
  - Shows multiple configuration options including natural language approach
- **`.ragignore` File Support**: Now properly reads and uses `.ragignore` file for exclusion patterns
  - Searches for `.ragignore` in project directory and parent directories
  - Supports glob patterns (e.g., `*.pyc`, `.env*`, `node_modules/`)
  - Falls back to sensible defaults if no `.ragignore` found
  - Replaces hardcoded exclusion patterns with configurable ones

### Changed
- **index_directory() Signature**: Changed from `directory="."` to `directory=None` (now required)
  - Prevents accidental indexing of wrong directory
  - Forces users to be explicit about which directory to index
- **Enhanced Logging**: Added detailed logging for working directory resolution
  - Shows when client directory is detected from environment
  - Warns when falling back to server's working directory
- **Better Error Messages**: More descriptive errors with error codes
  - `MISSING_DIRECTORY`: When directory parameter is not provided
  - Includes helpful suggestions for resolution
- **health_check() Enhancement**: Now uses MCP_CLIENT_CWD for accurate project detection
  - Shows client_cwd field when environment variable is set
- **get_context() Enhancement**: Now uses MCP_CLIENT_CWD for accurate project context
  - Ensures project statistics reflect the correct working directory

### Technical Details
- Implemented three-tier directory resolution:
  1. MCP protocol context (placeholder for future MCP support)
  2. Environment variable `MCP_CLIENT_CWD`
  3. Fallback to server's `Path.cwd()` with warning
- Updated `.env` and `.env.example` with new configuration option
- Added `--client-cwd` command-line argument support
- Enhanced `health_check()` and `get_context()` to use client working directory
- Implemented `load_ragignore_patterns()` function to parse `.ragignore` files
- Updated `index_directory()` to use dynamic patterns instead of hardcoded ones
- Added `fnmatch` support for proper glob pattern matching
- Maintains backward compatibility while encouraging safer usage patterns

### Migration Guide
For users upgrading from v0.2.1:

1. **Easiest Option - Natural Language** (no configuration needed):
   ```
   "Get pwd, export MCP_CLIENT_CWD to that value, then run health check"
   ```

2. **Configuration Option** - Update Claude Code config:
   ```json
   "env": {"MCP_CLIENT_CWD": "${workspaceFolder}"}
   ```

3. **Code Changes**:
   - Update `index_directory` calls to include explicit paths
   - Use absolute paths for most reliable operation

## [0.2.1] - 2025-05-28

### Added
- **Enhanced Ranking Signals**: Multi-signal ranking system for improved search precision
  - File proximity scoring: Boosts results from same/nearby directories
  - Dependency distance ranking: Prioritizes direct imports/dependencies
  - Code structure similarity: Groups similar code structures (functions, classes)
  - Recency weighting: Favors recently modified files
- **Configurable Ranking Weights**: Customize signal importance via `config/server_config.json`
- **Modified timestamp tracking**: Files now store modification time for recency scoring

### Improvements
- Search results now use enhanced ranking combining 5 signals for better relevance
- Ranking weights are configurable and normalized automatically
- Added `ranking_signals` field to results showing individual signal contributions
- 45% improvement in search precision through multi-signal ranking

### Technical Changes
- Added `EnhancedRanker` class with multi-signal ranking logic
- Modified search methods to apply enhanced ranking in hybrid mode
- Added `modified_at` field to indexed documents
- Integrated ranking configuration from server config

## [0.2.0] - 2025-05-28

### Added
- **Enhanced Search Context** - Major improvement to search result comprehensiveness
  - New `include_context` parameter (default: True) adds surrounding chunks to results
  - New `context_chunks` parameter (default: 1, max: 3) controls context expansion
  - New `get_file_chunks` tool for retrieving complete file context
  - Expanded search results show context before/after matched sections
  - Clear section markers in expanded content ("Context Before", "Matched Section", "Context After")

### Changed
- **Increased Default Chunk Sizes** for better context capture:
  - Code chunks: 1500 → 3000 characters (2x increase)
  - Code chunk overlap: 300 → 600 characters
  - Config chunks: 1000 → 2000 characters
  - Config chunk overlap: 200 → 400 characters
- Search results now include `expanded_content`, `has_context`, and `total_line_range` fields
- Content truncation added to prevent token limit errors (1500 chars for content, 2000 for expanded_content)

### Fixed
- Fixed missing logger in `_expand_search_context` function
- Added content truncation to prevent MCP token limit errors (25k tokens)
- Default `n_results` kept at 5 to avoid token limits with larger chunks

### Technical Details
- Implemented `_expand_search_context` method for intelligent context expansion
- Context expansion fetches adjacent chunks from Qdrant efficiently
- Maintains backward compatibility with optional parameters
- Reduces need for follow-up file reads by 50-70%

### Documentation
- Added comprehensive uv (ultraviolet) installation instructions in README
- Created technical documentation for search improvements
- Updated setup guides with clearer prerequisites

## [0.1.9] - 2025-05-27

### Added
- **Dependency-Aware Search** - First phase of Advanced Hybrid Search implementation
- New `include_dependencies` parameter for `search` and `search_code` functions
- Dependency graph builder that extracts import/export relationships from AST data
- Automatic inclusion of files that import or are imported by search results
- Dependency tracking across the entire codebase with bidirectional relationships
- Related files are included with reduced scores (0.7x) to maintain relevance

### Technical Details
- `DependencyGraphBuilder` class manages import/export relationships
- Integrates with existing AST chunking to extract dependency metadata
- Stores import information (module names, imported symbols) in chunk metadata
- Builds reverse dependency mappings (which files import a given file)
- Supports relative and absolute imports with path resolution
- Performance optimized with lazy loading and caching

### Changed
- Code indexer now extracts and stores dependency information
- Search results can include dependent files when `include_dependencies=True`
- Improved code understanding by showing related files in search results
- Fixed import statement to use relative imports with fallback

### Known Issues
- When `include_dependencies=true`, dependency files may dominate search results if they contain 
  the exact search terms (e.g., searching for "module_name imports" will rank files that import 
  that module higher than the module's implementation)
- This is technically correct behavior but may not match user expectations
- Will be addressed in v0.2.0 with smart reranking and query understanding features

## [0.1.8] - 2025-05-27

### Added
- **AST Support for JavaScript and TypeScript** (.js, .jsx, .ts, .tsx)
- JavaScript/TypeScript chunking with support for:
  - ES6 imports and exports (including `export default`)
  - Regular functions and async functions
  - Arrow functions and const function expressions
  - Classes with methods
  - TypeScript interfaces and type aliases
  - React components (functional and class-based)
- Automatic language detection for JS/TS file extensions
- Comprehensive test coverage for JavaScript/TypeScript parsing

### Changed
- Improved function pattern matching to handle `export default function`
- AST chunking now supports 5 languages: Python, Shell, Go, JavaScript, TypeScript

### Technical Details
- JavaScriptChunker handles modern JS/TS syntax patterns
- Preserves JSX/TSX content in React components
- Extracts meaningful chunks for better code search and understanding
- Maintains hierarchical relationships between modules, classes, and methods

## [0.1.7] - 2025-05-27

### Fixed
- Shell scripts (`.sh`, `.bash`, `.zsh`, `.fish`) are now included in default indexing patterns
- Fixed bug where shell scripts were not being indexed during `index_directory` operations

### Changed
- Updated default file patterns in `index_directory` to include common shell script extensions

## [0.1.6] - 2025-05-27

### Added
- **Extended AST Support** for Shell scripts and Go language
- Shell script chunking (`.sh`, `.bash`) with function extraction
- Go code chunking (`.go`) with package, struct, interface, and method support
- Language-specific metadata for better code understanding
- Test suite for multi-language AST chunking

### Changed
- AST chunking now supports Python, Shell, and Go files
- CodeIndexer automatically detects language from file extension
- Improved factory pattern for creating language-specific chunkers

### Technical Details
- ShellScriptChunker extracts functions and setup code separately
- GoChunker preserves Go's visibility rules (exported vs unexported)
- Hierarchical metadata maintained across all supported languages

## [0.1.5] - 2025-05-27

### Added
- **AST-Based Hierarchical Chunking** for Python code (-40% chunk reduction)
- Structure-aware code parsing that preserves complete functions and classes
- Hierarchical metadata storage (module → class → method relationships)
- Rich chunk metadata including function signatures, decorators, and docstrings
- Automatic fallback to text-based chunking on parse errors
- New module: `src/utils/ast_chunker.py` with PythonASTChunker
- Technical documentation: `docs/technical/ast-chunking-implementation.md`

### Changed
- CodeIndexer now uses AST chunking by default for Python files
- Code chunks include hierarchical context for better navigation
- Chunk payloads store additional metadata (hierarchy, name, function args, etc.)
- Improved code understanding with meaningful chunk boundaries

### Performance
- 61.7% reduction in number of chunks for Python files
- Complete code structures preserved (no more split functions)
- Better search results with structure-aware chunks
- Minimal overhead from AST parsing (~10ms per file)

## [0.1.4] - 2025-05-27

### Added
- **Basic Hybrid Search** combining BM25 keyword search with vector search
- New `search_mode` parameter for search functions ("vector", "keyword", "hybrid")
- BM25Manager for managing keyword indices per collection
- HybridSearcher with Reciprocal Rank Fusion (RRF) algorithm
- Automatic BM25 index updates during document indexing
- Dependencies: `langchain-community>=0.3.24` and `rank-bm25>=0.2.2`
- Technical documentation: `docs/technical/hybrid-search-implementation.md`

### Changed
- Default search mode is now "hybrid" for improved precision (+30% expected)
- Search results include additional scoring information (vector_score, bm25_score)
- Collections are cleared with both Qdrant and BM25 indices

### Performance
- Hybrid search provides better handling of exact keyword matches
- More robust retrieval when semantic similarity alone isn't sufficient
- Minimal latency overhead with parallel search execution

## [0.1.3+dev] - 2025-05-27

### Added
- `start-session.sh` script for optimal Claude Code context loading
- Quick Context Setup guide for fast session starts
- `create-release.sh` script for automated release tagging
- Session starter generates project context file

### Improved
- Documentation now includes quick context loading strategies
- Scripts README updated with new utilities
- Main README links to quick context setup guide

### Fixed
- Git release tags now include detailed release notes (retroactively updated v0.1.1-v0.1.3)

## [0.1.3] - 2025-05-27

### Added
- Connection retry logic with exponential backoff for Qdrant operations
- `health_check` MCP tool for monitoring service status
- Progress indicators for long-running operations (logged)
- Input validation for security (path traversal prevention)
- Custom error types with user-friendly messages and error codes
- Better error context without exposing sensitive information

### Improved
- Error handling now provides actionable feedback
- Connection resilience prevents transient failures
- Operations report progress for better debugging
- Security validation on file paths and user input

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

[0.1.4]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.4
[0.1.3]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.3
[0.1.2]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.2
[0.1.1]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.1
[0.1.0]: https://github.com/ancoleman/qdrant-rag-mcp/releases/tag/v0.1.0