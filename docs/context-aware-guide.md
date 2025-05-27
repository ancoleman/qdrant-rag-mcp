# Context-Aware Qdrant RAG MCP Server Guide

This guide explains how the context-aware version keeps project knowledge separate and provides relevant context based on your current working directory.

## Key Concept: Project Isolation

The context-aware server automatically:
1. **Detects your current project** based on project markers (.git, package.json, etc.)
2. **Scopes searches to the current project** by default
3. **Indexes files into project-specific collections**
4. **Maintains clear boundaries** between different projects

## Installation

```bash
# Install globally with context awareness
cd ~/mcp-servers/qdrant-rag
./install_global.sh
```

This automatically configures Claude Code with context awareness enabled.

## How Context Detection Works

### Project Detection
When you run a command, the server:
1. Checks your current working directory
2. Looks for project markers going up the directory tree:
   - `.git` directory
   - `package.json` 
   - `pyproject.toml`
   - `Cargo.toml`
   - `go.mod`
   - `pom.xml`
   - `.project`
3. Uses the project root directory name as the project identifier
4. Creates project-specific collections: `project_<name>_code` and `project_<name>_config`

### Example Scenarios

#### Scenario 1: Working on a React App
```bash
cd ~/projects/my-react-app
# Claude CLI with MCP server

# This indexes only into my-react-app collections
index_directory .

# This searches ONLY in my-react-app by default
search "useState hook implementation"
# Returns: Results only from my-react-app project
```

#### Scenario 2: Switching to an API Project
```bash
cd ~/projects/my-api-backend
# Claude CLI continues...

# Now indexing goes to my-api-backend collections
index_code src/auth.py

# Searches are scoped to my-api-backend
search "authentication middleware"
# Returns: Results only from my-api-backend project
```

#### Scenario 3: Cross-Project Search When Needed
```bash
# Still in my-api-backend directory

# Search across all projects for a pattern
search "JWT implementation" cross_project=true
# Returns: Results from all indexed projects
```

## Commands and Their Behavior

### Default Project-Scoped Commands

| Command | Behavior | Example |
|---------|----------|---------|
| `index_code` | Indexes to current project | `index_code src/main.py` |
| `index_directory` | Indexes to current project | `index_directory .` |
| `search` | Searches current project only | `search "error handling"` |
| `search_code` | Searches current project code | `search_code "async function"` |

### Cross-Project Options

| Command | Parameter | Example |
|---------|-----------|---------|
| `search` | `cross_project=true` | `search "design pattern" cross_project=true` |
| `search_code` | `cross_project=true` | `search_code "logger" cross_project=true` |
| `index_code` | `force_global=true` | `index_code ~/scripts/util.py force_global=true` |

### Context Management

| Command | Purpose | Example |
|---------|---------|---------|
| `get_context` | Show current project info | `get_context` |
| `switch_project` | Change project context | `switch_project ~/projects/other-app` |

## Real-World Workflow Examples

### 1. Microservices Development

```bash
# Working on user service
cd ~/projects/microservices/user-service
# Index the service
index_directory .

# Search within user service only
search "user validation"

# Need to check how auth service does it?
search "token validation" cross_project=true

# Switch to auth service
switch_project ~/projects/microservices/auth-service
# Now searches default to auth service
search "token generation"
```

### 2. Frontend and Backend Development

```bash
# Morning: Working on React frontend
cd ~/projects/app-frontend
index_directory . patterns=["*.js", "*.jsx", "*.css"]
search "form submission"  # Only frontend results

# Afternoon: Switch to backend
cd ~/projects/app-backend
index_directory . patterns=["*.py", "*.yml"]
search "form handler"  # Only backend results

# Need to see the full flow?
search "form data flow" cross_project=true
```

### 3. Library Development with Examples

```bash
# Working on main library
cd ~/projects/my-awesome-lib
index_directory src
search "public API"  # Only library code

# Checking example usage
cd ~/projects/my-awesome-lib/examples
index_directory .
search "import MyLib"  # Only example code

# See all usages across library and examples
cd ~/projects/my-awesome-lib
search "MyLib usage" cross_project=true
```

## Benefits of Context Awareness

### 1. **Reduced Noise**
- Search results are relevant to your current task
- No confusion from similar code in other projects
- Faster, more focused results

### 2. **Project Privacy**
- Client A's code doesn't appear when working on Client B
- Personal projects stay separate from work projects
- Experimental code doesn't pollute production searches

### 3. **Better Claude Assistance**
- Claude understands your current project context
- Suggestions are relevant to the codebase you're working on
- No accidental mixing of different project patterns

### 4. **Flexible When Needed**
- Cross-project search for finding patterns
- Global search for utility functions
- Easy project switching for multi-project work

## Configuration

### Project Markers
The server recognizes these files/directories as project roots:
- `.git` - Git repositories
- `package.json` - Node.js projects
- `pyproject.toml` - Python projects
- `Cargo.toml` - Rust projects
- `go.mod` - Go projects
- `pom.xml` - Java/Maven projects
- `.project` - Generic project marker

### Excluded Patterns
These are automatically excluded from indexing:
- `.git/`, `__pycache__/`, `node_modules/`
- `.venv/`, `venv/`, `env/`
- `build/`, `dist/`, `.cache/`, `tmp/`

## Troubleshooting

### Project Not Detected
If `get_context` shows no project:
1. Check if you have a project marker in parent directories
2. Create a `.project` file in your project root
3. Use `force_global=true` for standalone scripts

### Wrong Project Detected
If files are going to the wrong project:
1. Check `get_context` to see current detection
2. Ensure you're in the right directory
3. Consider adding a project marker closer to your files

### Need to Reindex After Changes
If project structure changed:
1. Clear the old project: `clear_project "old_name"`
2. Re-index from the new location

## Best Practices

1. **Start Each Session with Context Check**
   ```
   get_context
   ```

2. **Index at Project Start**
   ```
   index_directory . 
   ```

3. **Use Project Defaults**
   - Let the server handle project detection
   - Only use `cross_project=true` when needed

4. **Organize Projects Clearly**
   - Keep projects in separate directories
   - Use standard project markers
   - Avoid nested projects

This context-aware approach ensures Claude Code always has the right context for your current work!