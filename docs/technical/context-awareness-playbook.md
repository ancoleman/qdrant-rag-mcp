# 🎯 Context Awareness Playbook

This playbook explains how the context-aware Python server (`qdrant_mcp_context_aware.py`) works and confirms it's designed specifically for the global installation.

## ✅ Compatibility Check

**Question**: Does the context-aware Python work with the global install?  
**Answer**: YES! It's specifically designed for it. Here's why:

1. The runner script (`qdrant-rag-context.sh`) preserves your working directory
2. The Python script detects the project from that working directory
3. All paths are resolved to absolute paths before processing

---

## 🎬 Context Detection Flow

### Step 1: Starting Point
```
You are here: ~/projects/my-awesome-app
You run: claude (which starts the MCP server)
```

### Step 2: Python Server Receives Working Directory
```python
# When server starts, Python's os.getcwd() returns:
# "/Users/you/projects/my-awesome-app"
# NOT the MCP installation directory!
```

### Step 3: Project Detection Triggered
```python
def get_current_project() -> Optional[Dict[str, str]]:
    """Detect current project based on working directory"""
    global _current_project
    
    # Get current working directory
    cwd = Path.cwd()  # This is YOUR project directory
```

### Step 4: Search for Project Markers
```python
# Starting from current directory, go up the tree
for parent in [cwd] + list(cwd.parents):
    for marker in PROJECT_MARKERS:
        if (parent / marker).exists():
            # Found a project root!
```

**PROJECT_MARKERS searched in order**:
1. `.git` - Git repository
2. `package.json` - Node.js project
3. `pyproject.toml` - Python project
4. `Cargo.toml` - Rust project
5. `go.mod` - Go project
6. `pom.xml` - Java/Maven project
7. `.project` - Generic marker

### Step 5: Project Identification
```python
project_name = parent.name.replace(" ", "_").replace("-", "_")
_current_project = {
    "name": project_name,              # "my_awesome_app"
    "root": str(parent),              # "/Users/you/projects/my-awesome-app"
    "collection_prefix": f"project_{project_name}"  # "project_my_awesome_app"
}
```

---

## 📊 Collection Assignment Flow

### When Indexing a File

```python
@mcp.tool()
def index_code(file_path: str, force_global: bool = False) -> Dict[str, Any]:
```

#### Scenario 1: Indexing File in Current Project
```
Current directory: ~/projects/my-awesome-app
Command: index_code("src/main.py")
```

**Flow**:
1. Resolve path: `src/main.py` → `/Users/you/projects/my-awesome-app/src/main.py`
2. Get current project: `my_awesome_app`
3. Check if file is in project: YES (it's under project root)
4. Assign collection: `project_my_awesome_app_code`

#### Scenario 2: Indexing File Outside Current Project
```
Current directory: ~/projects/my-awesome-app
Command: index_code("/Users/you/projects/other-app/main.py")
```

**Flow**:
1. Resolve path: Already absolute
2. Get current project: `my_awesome_app`
3. Check if file is in project: NO (different path)
4. Detect file's project: Found `.git` in `other-app`
5. Assign collection: `project_other_app_code`

#### Scenario 3: Indexing File with No Project
```
Current directory: ~/projects/my-awesome-app
Command: index_code("/tmp/random-script.py")
```

**Flow**:
1. Resolve path: Already absolute
2. Get current project: `my_awesome_app`
3. Check if file is in project: NO
4. Detect file's project: No markers found
5. Assign collection: `global_code`

---

## 🔍 Search Scoping Flow

### Default Search (Current Project Only)

```python
@mcp.tool()
def search(query: str, n_results: int = 5, cross_project: bool = False) -> Dict[str, Any]:
```

**Command**: `search("database connection")`

```python
# Step 1: Get current project
current_project = get_current_project()
# Returns: {"name": "my_awesome_app", ...}

# Step 2: Determine collections to search
if cross_project:
    search_collections = all_collections  # Search everything
else:
    # Only search current project collections
    search_collections = [
        "project_my_awesome_app_code",
        "project_my_awesome_app_config"
    ]

# Step 3: Execute search only on these collections
```

### Cross-Project Search

**Command**: `search("authentication pattern", cross_project=True)`

```python
# Step 1: Same project detection
# Step 2: But now we search ALL collections
search_collections = [
    "project_my_awesome_app_code",
    "project_my_awesome_app_config",
    "project_other_app_code",
    "project_other_app_config",
    "project_third_app_code",
    "global_code",
    "global_config"
]
```

---

## 🗂️ Complete Example Walkthrough

### Starting State
```
Qdrant Collections:
- project_frontend_code (500 chunks)
- project_frontend_config (50 chunks)
- project_backend_code (800 chunks)
- project_backend_config (100 chunks)
- global_code (20 chunks)
```

### Session 1: Working on Frontend
```bash
cd ~/projects/frontend
claude
```

**In Claude**:
```
> get_context()
{
  "current_project": {
    "name": "frontend",
    "root": "/Users/you/projects/frontend",
    "collection_prefix": "project_frontend"
  },
  "collections": ["project_frontend_code", "project_frontend_config"],
  "total_indexed": 550
}

> search("React component")
# Searches ONLY in: project_frontend_code, project_frontend_config
# Returns: Results only from frontend project

> index_code("src/NewComponent.jsx")
# Indexes to: project_frontend_code
```

### Session 2: Switch to Backend
```bash
cd ~/projects/backend
# Claude still running, but now we're in different directory
```

**In Claude**:
```
> get_context()
{
  "current_project": {
    "name": "backend",
    "root": "/Users/you/projects/backend",
    "collection_prefix": "project_backend"
  },
  "collections": ["project_backend_code", "project_backend_config"],
  "total_indexed": 900
}

> search("React component")
# Searches ONLY in: project_backend_code, project_backend_config
# Returns: No results (backend has no React!)

> search("API endpoint")
# Searches ONLY in: project_backend_code, project_backend_config
# Returns: Backend API endpoints only
```

---

## 🔧 How Global Installation Enables This

### The Critical Path Preservation

```bash
# In qdrant-rag-context.sh:
CURRENT_DIR="$(pwd)"                    # Save: /Users/you/projects/my-app
cd "$MCP_SERVER_DIR"                    # Change to load .env
source .env                             # Load configuration
cd "$CURRENT_DIR"                       # RESTORE: /Users/you/projects/my-app
exec uv run ... qdrant_mcp_context_aware.py  # Python sees correct directory!
```

### Why This Works Globally

1. **No hardcoded paths**: Everything uses relative or detected paths
2. **Dynamic collection names**: Based on detected project name
3. **Working directory awareness**: Python sees where YOU are, not where IT is
4. **Absolute path resolution**: All file operations convert to absolute paths

---

## 🎭 Edge Cases Handled

### Edge Case 1: Nested Projects
```
~/projects/
  big-monorepo/
    .git/
    frontend/
      package.json
    backend/
      pyproject.toml
```

**Behavior**: Uses the first marker found going up (frontend → big-monorepo)

### Edge Case 2: No Project Markers
```
~/random-scripts/
  util.py
  helper.py
```

**Behavior**: Uses directory name as project: `dir_random_scripts`

### Edge Case 3: Symlinked Directories
```
~/projects/my-app -> /actual/location/app
```

**Behavior**: Resolves to real path, uses real directory name

---

## 🚀 Performance Optimizations

### Project Cache
```python
_current_project = None  # Cache current project detection

def get_current_project():
    global _current_project
    
    # Check if we've already detected the project
    cwd = Path.cwd()
    if _current_project and Path(_current_project["root"]) == cwd:
        return _current_project  # Use cached value
```

### Lazy Service Loading
```python
# Services only initialized when first used
_qdrant_client = None
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Load only when needed
        _embedding_model = SentenceTransformer(...)
    return _embedding_model
```

---

## 🎯 Summary

The context-aware Python server is **perfectly designed** for global installation because:

1. **It reads the current working directory** preserved by the runner script
2. **It dynamically detects projects** based on where you are
3. **It scopes operations** to the detected project automatically
4. **It handles cross-project scenarios** when explicitly requested

This design means you can `cd` to any project and Claude will automatically work with just that project's context!