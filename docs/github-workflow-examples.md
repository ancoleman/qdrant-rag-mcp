# GitHub Workflow Examples - Real-World Usage Guide

This guide provides practical examples and patterns for using the GitHub integration to work through issues systematically.

## 📋 Table of Contents
- [Initial Setup](#initial-setup)
- [Working Through Issues One by One](#working-through-issues-one-by-one)
- [Batch Processing Multiple Issues](#batch-processing-multiple-issues)
- [Common Workflow Patterns](#common-workflow-patterns)
- [Best Practices](#best-practices)
- [Command Reference](#command-reference)

## 🚀 Initial Setup

Before starting your issue remediation session, set up your repository context:

```bash
# Set repository context (required once per session)
"Switch to repository owner/repo-name"

# Verify connection
"Check GitHub health status"

# Optional: Set default repository in .env
GITHUB_REPO_OWNER=your-username
GITHUB_REPO_NAME=your-repository
```

## 📝 Working Through Issues One by One

### Step-by-Step Issue Remediation

Here's the recommended workflow for addressing individual issues:

#### 1. **Discovery Phase**
```bash
# List all open issues
"Show me all open issues"

# Filter by labels
"Show me open issues with bug label"
"Fetch issues with enhancement label"

# Get specific issue details
"Get details for issue #5"
```

#### 2. **Analysis Phase**
```bash
# Analyze issue with RAG search
"Analyze issue #5"

# This will:
# - Extract key patterns from the issue
# - Search your codebase for relevant files
# - Identify potential root causes
# - Map issue to specific code locations
```

#### 3. **Solution Generation**
```bash
# Generate fix suggestions
"Generate fix suggestions for issue #5"

# Returns:
# - Multiple fix approaches with confidence scores
# - Specific code changes needed
# - Risk assessment for each approach
# - Implementation complexity
```

#### 4. **Preview Phase (Dry Run)**
```bash
# Preview what would happen
"Resolve issue #5 in dry-run mode"

# Shows:
# - Branch name that would be created
# - Files that would be modified
# - Exact changes to be applied
# - Pull request description
# - Safety check results
```

#### 5. **Communication**
```bash
# Keep stakeholders informed
"Add comment to issue #5: 'Analyzing this issue with RAG system. Found relevant files: src/indexers/code_indexer.py'"

# Progress updates
"Add comment to issue #5: 'Root cause identified - regex pattern issue. Testing fix now.'"
```

#### 6. **Implementation**
```bash
# Apply the fix (creates PR)
"Resolve issue #5"

# Or with explicit dry-run disabled
"Resolve issue #5 with dry_run=false"

# This will:
# - Create a new branch
# - Apply the code changes
# - Commit with descriptive message
# - Push to remote
# - Create pull request
# - Link PR to issue
```

## 🔄 Batch Processing Multiple Issues

### Systematic Issue Processing

For handling multiple issues efficiently:

```python
# Example workflow for processing all bugs

1. "Show me all open issues with bug label"
   
2. For each high-priority bug:
   a. "Analyze issue #X"
   b. Review analysis results
   c. "Generate fix suggestions for issue #X"
   d. If confidence > 80%:
      - "Resolve issue #X in dry-run mode"
      - Review changes
      - "Resolve issue #X"
   e. If confidence < 80%:
      - "Add comment to issue #X: 'Needs manual review - confidence too low'"

3. "Show me all open issues with enhancement label"
   # Repeat process for enhancements
```

### Priority-Based Processing

```bash
# Process by severity
"Show me issues with label critical"
# Handle these first

"Show me issues with label high-priority"
# Handle these second

"Show me issues with label good-first-issue"
# Good for testing the workflow
```

## 📊 Common Workflow Patterns

### Pattern 1: Bug Triage and Fix
```bash
# 1. Identify the bug
"Get issue #123 details"

# 2. Understand impact
"Analyze issue #123"

# 3. Verify the fix approach
"Generate fix suggestions for issue #123"

# 4. Test safely
"Resolve issue #123 in dry-run mode"

# 5. Apply if confident
"Resolve issue #123"
```

### Pattern 2: Feature Request Implementation
```bash
# 1. Understand requirements
"Get issue #456 with comments"

# 2. Analyze codebase for integration points
"Analyze issue #456"

# 3. Design implementation
"Generate fix suggestions for issue #456"

# 4. Create draft PR for discussion
"Create pull request titled 'Draft: Feature #456' with body 'Initial implementation for discussion'"
```

### Pattern 3: Documentation Issues
```bash
# 1. Identify documentation gaps
"Show me issues with label documentation"

# 2. For each doc issue:
"Analyze issue #789"

# 3. Generate documentation updates
"Generate fix suggestions for issue #789"

# 4. Apply changes
"Resolve issue #789"
```

### Pattern 4: Security Fixes
```bash
# 1. Handle with extra caution
"Get issue #999 details"

# 2. Thorough analysis
"Analyze issue #999"

# 3. Always dry-run first
"Resolve issue #999 in dry-run mode"

# 4. Add security review comment
"Add comment to issue #999: 'Security fix prepared. Requires security team review before merge.'"

# 5. Create draft PR
"Create pull request as draft for security review"
```

## ✅ Best Practices

### 1. **Always Analyze First**
- Never skip the analysis phase
- RAG search provides crucial context
- Helps avoid duplicate work

### 2. **Use Dry-Run Mode**
- Preview changes before applying
- Verify file modifications are correct
- Check branch naming conventions

### 3. **Communicate Progress**
- Add comments for transparency
- Link related issues
- Update status regularly

### 4. **Monitor Confidence Scores**
```
Confidence > 85%: Usually safe to auto-apply
Confidence 70-85%: Review carefully, test thoroughly
Confidence < 70%: Consider manual intervention
```

### 5. **Batch Similar Issues**
- Group by label or component
- Apply similar fixes together
- Test as a cohesive unit

### 6. **Review Generated PRs**
- Even high-confidence fixes need review
- Check for unintended side effects
- Ensure code style consistency

## 📚 Command Reference

### Repository Management
```bash
"List my repositories"
"Switch to repository owner/repo"
"Check GitHub health status"
```

### Issue Discovery
```bash
"Show me all open issues"
"Show me closed issues"
"Show issues with label [label-name]"
"Show issues assigned to me"
"Get issue #[number] details"
"Get issue #[number] with comments"
```

### Issue Analysis
```bash
"Analyze issue #[number]"
"Search codebase for [pattern]"  # Uses RAG
"Find files related to issue #[number]"
```

### Fix Generation
```bash
"Generate fix suggestions for issue #[number]"
"Suggest improvements for issue #[number]"
```

### Implementation
```bash
"Resolve issue #[number] in dry-run mode"
"Resolve issue #[number]"
"Resolve issue #[number] with dry_run=false"
```

### Communication
```bash
"Add comment to issue #[number]: '[message]'"
"Create issue titled '[title]' with body '[description]'"
"Create issue with labels bug,urgent"
```

### Pull Requests
```bash
"Create pull request titled '[title]' with body '[description]'"
"Create PR from branch [branch-name]"
```

## 🔍 Advanced Examples

### Multi-Issue Dependencies
```bash
# When issues depend on each other
"Get issue #100 details"  # Parent issue
"Analyze issue #100"

"Get issue #101 details"  # Dependent issue
"Add comment to issue #101: 'Blocked by #100. Will address after #100 is resolved.'"

# After fixing #100
"Resolve issue #100"
"Add comment to issue #101: 'Blocker #100 resolved. Starting work on this issue.'"
"Resolve issue #101"
```

### Handling Large Changes
```bash
# For issues requiring multiple files
"Analyze issue #200"
"Generate fix suggestions for issue #200"

# If many files affected
"Resolve issue #200 in dry-run mode"
# Review carefully

# Consider breaking into smaller PRs
"Add comment to issue #200: 'This requires extensive changes. Breaking into smaller PRs for easier review.'"
```

### Emergency Hotfixes
```bash
# For critical production issues
"Get issue #911 details"
"Analyze issue #911"

# Skip dry-run for emergencies (use cautiously)
"Resolve issue #911 with dry_run=false"

# Immediately notify
"Add comment to issue #911: 'Emergency fix applied. PR #X created. Please review ASAP.'"
```

## 📈 Tracking Progress

### Session Summary Pattern
```bash
# At start of session
"Show me all open issues"
# Note the count

# After working through issues
"Show me all open issues"
# Compare count

# Check your work
"Show me closed issues from today"
"Show me my recent pull requests"
```

### Metrics Commands
```bash
# See what's been accomplished
"List pull requests created today"
"Show issues closed this week"
"Show issues with linked PRs"
```

## 🚨 Troubleshooting Common Scenarios

### Issue Not Found
```bash
"Get issue #999 details"
# Error: Issue not found

# Verify repository context
"Check current repository"
"Switch to repository owner/repo"
```

### Low Confidence Fixes
```bash
"Generate fix suggestions for issue #50"
# All suggestions have low confidence

# Get more context
"Add comment to issue #50: 'Automated analysis shows low confidence. Needs manual review for context.'"

# Try more specific analysis
"Search codebase for specific patterns mentioned in issue #50"
```

### PR Creation Fails
```bash
"Resolve issue #75"
# Error: Branch already exists

# Check existing PRs
"Show pull requests for issue #75"

# Use different branch name
"Create PR with branch name 'fix-issue-75-v2'"
```

## 💡 Pro Tips

1. **Start Small**: Test with `good-first-issue` labels
2. **Build Confidence**: Use dry-run extensively at first
3. **Create Templates**: Save common comment patterns
4. **Track Patterns**: Notice which issues resolve well automatically
5. **Iterate**: The system learns from your codebase patterns

Remember: The GitHub integration is a powerful tool, but human oversight ensures quality. Use automation to accelerate your workflow, not replace careful review.