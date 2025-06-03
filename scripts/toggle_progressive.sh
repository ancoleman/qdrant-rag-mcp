#!/bin/bash
# Toggle progressive context on/off

CONFIG_FILE="$(dirname "$0")/../config/server_config.json"

# Read current state
CURRENT=$(grep -A1 '"progressive_context"' "$CONFIG_FILE" | grep '"enabled"' | awk -F': ' '{print $2}' | tr -d ',' | tr -d ' ')

if [ "$CURRENT" = "true" ]; then
    echo "Disabling progressive context..."
    # Use sed to toggle the enabled state
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' '/"progressive_context":/,/}/ s/"enabled": true/"enabled": false/' "$CONFIG_FILE"
    else
        # Linux
        sed -i '/"progressive_context":/,/}/ s/"enabled": true/"enabled": false/' "$CONFIG_FILE"
    fi
    echo "Progressive context DISABLED - using regular search by default"
else
    echo "Enabling progressive context..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' '/"progressive_context":/,/}/ s/"enabled": false/"enabled": true/' "$CONFIG_FILE"
    else
        # Linux
        sed -i '/"progressive_context":/,/}/ s/"enabled": false/"enabled": true/' "$CONFIG_FILE"
    fi
    echo "Progressive context ENABLED - using progressive search by default"
fi

echo "Restart Claude Code session for changes to take effect"