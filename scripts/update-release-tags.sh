#!/bin/bash
# Script to update existing tags with detailed release notes

set -e

echo "This script will update existing tags with detailed release notes from CHANGELOG.md"
echo "WARNING: This will delete and recreate tags, which can cause issues if others have already pulled them."
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Function to update a tag
update_tag() {
    local tag=$1
    local version_no_v=${tag#v}
    
    echo ""
    echo "Updating tag $tag..."
    
    # Get the commit the tag points to
    commit=$(git rev-list -n 1 "$tag")
    
    # Extract release notes from CHANGELOG.md
    temp_file=$(mktemp)
    
    # Use grep and sed to extract the section
    start_line=$(grep -n "## \[$version_no_v\]" CHANGELOG.md | cut -d: -f1)
    if [ -z "$start_line" ]; then
        echo "Warning: No release notes found for $tag in CHANGELOG.md"
        rm "$temp_file"
        return
    fi
    
    # Find the next version header or end of file
    next_line=$(tail -n +$((start_line + 1)) CHANGELOG.md | grep -n "^## \[" | head -1 | cut -d: -f1)
    
    if [ -n "$next_line" ]; then
        end_line=$((start_line + next_line - 1))
        sed -n "${start_line},${end_line}p" CHANGELOG.md > "$temp_file"
    else
        # No next version, take to end of file
        tail -n +$start_line CHANGELOG.md > "$temp_file"
    fi
    
    # Format the first line
    sed -i '' '1s/## \[\(.*\)\] - \(.*\)/Release v\1 - \2\n/' "$temp_file"
    
    if [ ! -s "$temp_file" ]; then
        echo "Warning: No release notes found for $tag in CHANGELOG.md"
        rm "$temp_file"
        return
    fi
    
    # Show what we'll use
    echo "Release notes for $tag:"
    echo "----------------------"
    cat "$temp_file"
    echo "----------------------"
    
    # Delete the old tag locally
    git tag -d "$tag"
    
    # Create new annotated tag at the same commit
    git tag -a "$tag" "$commit" -F "$temp_file"
    
    rm "$temp_file"
    echo "✓ Updated $tag"
}

# Update tags v0.1.1, v0.1.2, and v0.1.3
for tag in v0.1.1 v0.1.2 v0.1.3; do
    update_tag "$tag"
done

echo ""
echo "All tags updated locally!"
echo ""
echo "To push the updated tags to remote (this will overwrite remote tags):"
echo "  git push origin --tags --force"
echo ""
echo "WARNING: Force-pushing tags can cause issues for others who have already pulled them."
echo "Make sure to communicate this change to your team."