#!/bin/bash
# Enable progressive context in server_config.json

CONFIG_FILE="config/server_config.json"

echo "Enabling progressive context in $CONFIG_FILE..."

# Use jq to update the enabled flag
if command -v jq &> /dev/null; then
    # If jq is available, use it for clean JSON manipulation
    jq '.progressive_context.enabled = true' "$CONFIG_FILE" > tmp.json && mv tmp.json "$CONFIG_FILE"
    echo "✓ Progressive context enabled using jq"
else
    # Fallback to sed
    sed -i.bak 's/"enabled": false/"enabled": true/g' "$CONFIG_FILE"
    echo "✓ Progressive context enabled using sed"
fi

# Verify the change
if grep -q '"enabled": true' "$CONFIG_FILE"; then
    echo "✓ Configuration updated successfully"
    echo ""
    echo "Progressive context settings:"
    grep -A 20 '"progressive_context"' "$CONFIG_FILE" | grep -E '"enabled"|"default_level"|"similarity_threshold"' | head -3
else
    echo "✗ Failed to update configuration"
    exit 1
fi

echo ""
echo "You can now test progressive context with:"
echo "  python src/http_server.py  # Start the server"
echo "  python tests/test_progressive_http.py  # Run tests"