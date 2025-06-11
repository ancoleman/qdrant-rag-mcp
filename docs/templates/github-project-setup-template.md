# GitHub Project Setup Template

## Project Configuration

### Basic Information
- **Project Name**: [Descriptive name for the project]
- **Description**: [Brief description of project purpose and goals]
- **Visibility**: Public / Private
- **Template Type**: roadmap / bugs / features / custom

## Custom Fields Configuration

### Progress Tracking
**Field Name**: Progress  
**Type**: Single Select  
**Options**:
- 📋 Planned (Gray)
- 🚧 In Progress (Yellow)
- ✅ Completed (Green)
- ❌ Cancelled (Red)
- ⏸️ On Hold (Orange)

### Priority Levels
**Field Name**: Priority  
**Type**: Single Select  
**Options**:
- 🔥 Critical (Red)
- ⭐ High (Orange)
- 📌 Medium (Yellow)
- 📎 Low (Blue)

### Epic/Category
**Field Name**: Epic  
**Type**: Single Select  
**Options**:
- 🏗️ Foundation (Purple)
- 🚀 Enhancement (Blue)
- 🔌 Integration (Green)
- ⚡ Optimization (Yellow)
- 📚 Documentation (Gray)
- 🐛 Bug Fix (Red)
- 🔒 Security (Orange)

### Complexity Estimation
**Field Name**: Complexity  
**Type**: Single Select  
**Options**:
- 1️⃣ Trivial (Green) - < 2 hours
- 2️⃣ Simple (Blue) - 2-8 hours
- 3️⃣ Medium (Yellow) - 1-3 days
- 4️⃣ Complex (Orange) - 3-5 days
- 5️⃣ Very Complex (Red) - > 5 days

### Release Planning
**Field Name**: Target Version  
**Type**: Text  
**Format**: v[X.X.X]

### Timeline
**Field Name**: Due Date  
**Type**: Date  
**Format**: YYYY-MM-DD

### Additional Custom Fields

#### For Bug Tracking Template
- **Severity**: Critical / High / Medium / Low
- **Component**: Frontend / Backend / API / Database / Infrastructure
- **Environment**: Production / Staging / Development / All

#### For Feature Development Template
- **Stage**: Research / Design / Implementation / Testing / Deployment
- **Effort**: XS / S / M / L / XL
- **Impact**: High / Medium / Low

#### For Roadmap Template
- **Quarter**: Q1 / Q2 / Q3 / Q4
- **Strategic Goal**: Growth / Stability / Innovation / Technical Debt
- **Dependencies**: Text field for related items

## Views Configuration

### Board View (Default)
**Group By**: Progress  
**Sort By**: Priority (Descending)  
**Filter**: Not Cancelled  

### Roadmap View
**Layout**: Table  
**Visible Fields**:
- Title
- Progress
- Priority
- Epic
- Target Version
- Due Date
- Assignees

**Group By**: Target Version  
**Sort By**: Due Date (Ascending)

### Sprint View
**Filter**: Progress = "In Progress" OR (Progress = "Planned" AND Due Date < 2 weeks)  
**Group By**: Assignees  
**Sort By**: Priority (Descending)

### Backlog View
**Filter**: Progress = "Planned"  
**Group By**: Epic  
**Sort By**: Priority (Descending)

## Automation Rules (Conceptual)

### When Issue is Closed
- Set Progress → Completed
- Add completion date to custom field
- Move to "Done" column

### When PR is Merged
- Update linked issue Progress → Testing
- Add comment with PR link
- Notify assignees

### When High Priority Item Added
- Add to current sprint
- Notify team lead
- Set Due Date to current sprint end

## Issue Templates

### Bug Report Template
```markdown
## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [Expected result]
4. [Actual result]

## Environment
- OS: [e.g., macOS 12.0]
- Version: [e.g., v0.3.4]
- Browser: [if applicable]

## Additional Context
[Any other relevant information]

## Labels
`bug`, `[component]`, `[severity]`
```

### Feature Request Template
```markdown
## Feature Description
[What feature do you want to add?]

## Problem Statement
[What problem does this solve?]

## Proposed Solution
[How should it work?]

## Alternatives Considered
[Other approaches you've thought about]

## Additional Context
[Mockups, examples, or references]

## Labels
`enhancement`, `[component]`, `[priority]`
```

### Task Template
```markdown
## Task Description
[What needs to be done?]

## Acceptance Criteria
- [ ] [Specific criterion 1]
- [ ] [Specific criterion 2]
- [ ] [Specific criterion 3]

## Dependencies
- [Related issue or PR]
- [Required before this task]

## Labels
`task`, `[component]`, `[effort]`
```

## Team Workflow

### Issue Lifecycle
1. **New Issue Created** → Progress: Planned
2. **Sprint Planning** → Assign to developer, set Due Date
3. **Work Begins** → Progress: In Progress
4. **PR Submitted** → Link PR to issue
5. **PR Merged** → Progress: Testing
6. **Verified** → Progress: Completed
7. **Released** → Add release version tag

### Daily Workflow
- **Morning**: Check "My Items" view
- **During Day**: Update Progress as needed
- **End of Day**: Add comments on blockers

### Weekly Workflow
- **Monday**: Sprint planning, assign items
- **Wednesday**: Mid-sprint check-in
- **Friday**: Update progress, identify blockers

## Metrics & Reporting

### Velocity Tracking
- Items completed per sprint
- Story points (if using Effort field)
- Average cycle time by priority

### Quality Metrics
- Bug escape rate
- Time to resolution by severity
- Reopen rate

### Progress Tracking
- Burndown by sprint
- Cumulative flow by Progress state
- Release progress by Target Version

## Best Practices

### For Project Maintainers
1. Keep field options consistent
2. Regularly archive completed items
3. Review and update automation rules
4. Monitor project metrics weekly

### for Developers
1. Update Progress when starting work
2. Link PRs to issues
3. Add context in comments
4. Keep descriptions clear and actionable

### For Product Owners
1. Prioritize backlog weekly
2. Ensure issues have clear acceptance criteria
3. Review completed items for quality
4. Plan releases based on velocity

---

## Quick Setup Commands

```bash
# Create project from template
mcp github_create_project_from_template \
  --title "My Project Roadmap" \
  --template "roadmap" \
  --owner "myusername"

# Add issues to project
mcp github_smart_add_project_item \
  --project-id "PVT_xxxxx" \
  --issue-number 123

# Update item status
mcp github_update_project_item \
  --project-id "PVT_xxxxx" \
  --item-id "PVTI_xxxxx" \
  --field-id "PVTF_xxxxx" \
  --value "In Progress"
```

---

**Template Version**: 1.0  
**Last Updated**: [Date]  
**Compatible With**: GitHub Projects V2