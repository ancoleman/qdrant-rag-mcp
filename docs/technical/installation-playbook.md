# 📚 Global Installation Script Playbook

This playbook breaks down exactly how the `install_context_aware.sh` script works, step by step.

## 🎬 Script Overview

**Purpose**: Install the Qdrant RAG MCP Server globally so it works from any project directory  
**Result**: Context-aware MCP server that automatically scopes to your current project

---

## 📋 Pre-Installation State

```
Your System:
├── ~/mcp-servers/qdrant-rag/          # The MCP server code
├── ~/.claude-code/mcp-servers.json    # Claude's MCP configuration
└── ~/.mcp-servers/                    # May or may not exist
```

---

## 🎯 Step-by-Step Execution

### Step 1: Initial Setup
```bash
#!/bin/bash
set -e  # Exit on any error
```
**What happens**: 
- Script will stop if any command fails
- Ensures clean installation or clear failure

### Step 2: Get Script Location
```bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
```
**What happens**:
- Finds where the install script is running from
- Example: `/Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag`
- This is where all the MCP server code lives

### Step 3: Create Global MCP Directory
```bash
GLOBAL_MCP_DIR="$HOME/.mcp-servers"
mkdir -p "$GLOBAL_MCP_DIR"
```
**What happens**:
- Creates `~/.mcp-servers/` directory if it doesn't exist
- This will be the central location for all MCP servers
- The `-p` flag prevents errors if directory exists

**Result**:
```
~/.mcp-servers/     # Now exists
```

### Step 4: Create Symlink
```bash
ln -sfn "$SCRIPT_DIR" "$GLOBAL_MCP_DIR/qdrant-rag"
```
**What happens**:
- Creates a symbolic link from `~/.mcp-servers/qdrant-rag` → your actual code
- `-s` = symbolic link
- `-f` = force (overwrite if exists)
- `-n` = treat destination as normal file if it's a symlink

**Result**:
```
~/.mcp-servers/
└── qdrant-rag -> /Users/antoncoleman/Documents/repos/mcp-servers/qdrant-rag
```

### Step 5: Create Context-Aware Runner Script
```bash
cat > "$GLOBAL_MCP_DIR/qdrant-rag-context.sh" << 'EOF'
#!/bin/bash
# Context-aware runner for Qdrant RAG MCP Server

# Get the real path of the MCP server
MCP_SERVER_DIR="$(readlink -f "$HOME/.mcp-servers/qdrant-rag")"

# Save current directory (this is important for context detection)
CURRENT_DIR="$(pwd)"

# Change to the server directory to load environment
cd "$MCP_SERVER_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Change back to the original directory so the server detects the correct project
cd "$CURRENT_DIR"

# Run the context-aware version with the current directory as context
exec uv run --directory "$MCP_SERVER_DIR" python "$MCP_SERVER_DIR/src/qdrant_mcp_context_aware.py"
EOF
```

**What this runner script does**:

1. **Resolves the symlink** to find actual code location
2. **Saves your current directory** (critical for project detection!)
3. **Temporarily changes to MCP directory** to load `.env` file
4. **Returns to your original directory** so Python script sees correct context
5. **Runs the Python server** with `uv` from the MCP directory

**Why this matters**: This allows the server to:
- Load its configuration from its install location
- But detect your project from where you're actually working

### Step 6: Make Runner Executable
```bash
chmod +x "$GLOBAL_MCP_DIR/qdrant-rag-context.sh"
```
**What happens**:
- Makes the runner script executable
- Now it can be run as a command

### Step 7: Backup Existing Claude Config
```bash
CLAUDE_CONFIG="$HOME/.claude-code/mcp-servers.json"

if [ -f "$CLAUDE_CONFIG" ]; then
    cp "$CLAUDE_CONFIG" "$CLAUDE_CONFIG.backup.$(date +%Y%m%d_%H%M%S)"
fi
```
**What happens**:
- Locates Claude's MCP configuration file
- If it exists, creates a timestamped backup
- Example: `mcp-servers.json.backup.20240522_143022`

### Step 8: Update Claude Configuration
```bash
cat > "$CLAUDE_CONFIG" << EOF
{
  "servers": [
    {
      "name": "qdrant-rag",
      "command": "$GLOBAL_MCP_DIR/qdrant-rag-context.sh"
    }
  ]
}
EOF
```
**What happens**:
- Overwrites Claude's MCP config
- Points to our context-aware runner script
- Claude will now run this script when starting the MCP server

**Final config looks like**:
```json
{
  "servers": [
    {
      "name": "qdrant-rag",
      "command": "/Users/antoncoleman/.mcp-servers/qdrant-rag-context.sh"
    }
  ]
}
```

---

## 🏁 Post-Installation State

```
Your System:
├── ~/mcp-servers/qdrant-rag/              # Original code (unchanged)
├── ~/.mcp-servers/
│   ├── qdrant-rag -> ~/mcp-servers/...    # Symlink to code
│   └── qdrant-rag-context.sh              # Smart runner script
└── ~/.claude-code/
    ├── mcp-servers.json                    # Updated config
    └── mcp-servers.json.backup.20240522... # Backup
```

---

## 🔄 Runtime Flow

When you start Claude CLI:

1. **Claude reads** `~/.claude-code/mcp-servers.json`
2. **Claude executes** `~/.mcp-servers/qdrant-rag-context.sh`
3. **Runner script**:
   - Remembers your current directory (e.g., `~/projects/my-app`)
   - Loads environment from MCP install directory
   - Returns to your project directory
   - Starts Python server
4. **Python server**:
   - Sees your current directory as `~/projects/my-app`
   - Detects this as the project context
   - Scopes all operations to this project

---

## 🎭 Example Scenario

```bash
# You're in your React project
cd ~/projects/my-react-app
claude  # Start Claude CLI

# Behind the scenes:
# 1. Claude runs: ~/.mcp-servers/qdrant-rag-context.sh
# 2. Script saves: CURRENT_DIR="/Users/you/projects/my-react-app"
# 3. Script loads: .env from MCP install location
# 4. Script returns to: /Users/you/projects/my-react-app
# 5. Python starts and detects: "Current project is my-react-app"
# 6. All searches/indexes now scoped to my-react-app collections
```

---

## 🛠️ Why Each Part Matters

| Component | Purpose | What breaks without it |
|-----------|---------|----------------------|
| **Symlink** | Points to actual code | Can't find MCP server code |
| **Runner script** | Preserves working directory | Wrong project detection |
| **Current dir save/restore** | Enables context detection | Always uses global context |
| **Environment loading** | Gets database config | Can't connect to Qdrant |
| **Claude config** | Tells Claude what to run | Claude can't find MCP server |

---

## 🔍 Debugging the Installation

If something goes wrong, check:

1. **Symlink exists**: `ls -la ~/.mcp-servers/`
2. **Runner is executable**: `ls -la ~/.mcp-servers/qdrant-rag-context.sh`
3. **Claude config is correct**: `cat ~/.claude-code/mcp-servers.json`
4. **Runner script works**: `~/.mcp-servers/qdrant-rag-context.sh` (should start server)

This playbook shows how a seemingly simple install script actually sets up a sophisticated context-aware system!