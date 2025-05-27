#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <project_directory>"
    exit 1
fi

PROJECT_DIR="$1"

echo "Indexing project: $PROJECT_DIR"

# Use the MCP server to index the directory
curl -X POST http://localhost:8080/index_directory \
  -H "Content-Type: application/json" \
  -d "{
    \"directory\": \"$PROJECT_DIR\",
    \"patterns\": [\"*.py\", \"*.js\", \"*.java\", \"*.json\", \"*.xml\", \"*.yaml\", \"*.yml\"]
  }"