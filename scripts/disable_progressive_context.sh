#!/bin/bash
# Disable progressive context in server_config.json

CONFIG_FILE="config/server_config.json"

echo "Disabling progressive context in $CONFIG_FILE..."

# Use jq to update the enabled flag
if command -v jq &> /dev/null; then
    # If jq is available, use it for clean JSON manipulation
    jq '.progressive_context.enabled = false' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
    echo "✓ Progressive context disabled using jq"
else
    # Fallback to sed
    sed -i.bak 's/"enabled": true/"enabled": false/g' "$CONFIG_FILE"
    echo "✓ Progressive context disabled using sed"
fi

# Verify the change
if grep -q '"progressive_context"' "$CONFIG_FILE" && grep -A 5 '"progressive_context"' "$CONFIG_FILE" | grep -q '"enabled": false'; then
    echo "✓ Configuration updated successfully"
    echo ""
    echo "Progressive context is now disabled"
else
    echo "✗ Failed to update configuration"
    exit 1
fi

echo ""
echo "Progressive context has been disabled."
echo "The search functions will use the traditional behavior."