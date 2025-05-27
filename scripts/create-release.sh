#!/bin/bash
# Script to create a release with detailed notes from CHANGELOG.md

set -e

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v0.1.4"
    exit 1
fi

VERSION=$1
VERSION_NO_V=${VERSION#v}  # Remove 'v' prefix for searching CHANGELOG

# Extract release notes from CHANGELOG.md
echo "Extracting release notes for $VERSION from CHANGELOG.md..."

# Create a temporary file for the release notes
TEMP_FILE=$(mktemp)

# Extract the section for this version from CHANGELOG
awk -v version="## \\[$VERSION_NO_V\\]" '
    $0 ~ version { capture = 1; print "Release " $2 " " $3 " " $4; next }
    /^## \[/ && capture { exit }
    capture && /^###/ { print "" }
    capture { print }
' CHANGELOG.md > "$TEMP_FILE"

# Check if we found release notes
if [ ! -s "$TEMP_FILE" ]; then
    echo "Error: No release notes found for version $VERSION_NO_V in CHANGELOG.md"
    rm "$TEMP_FILE"
    exit 1
fi

# Show the release notes
echo ""
echo "Release notes:"
echo "=============="
cat "$TEMP_FILE"
echo "=============="
echo ""

# Confirm before creating tag
read -p "Create tag $VERSION with these notes? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    rm "$TEMP_FILE"
    exit 1
fi

# Create the annotated tag
echo "Creating annotated tag $VERSION..."
git tag -a "$VERSION" -F "$TEMP_FILE"

# Clean up
rm "$TEMP_FILE"

echo "Tag $VERSION created successfully!"
echo ""
echo "To push the tag to remote:"
echo "  git push origin $VERSION"
echo ""
echo "To push all tags:"
echo "  git push origin --tags"