# CLAUDE.md - AI Assistant Guide

This document provides guidance for AI assistants (like Claude) working with this codebase.

## Repository Overview

**Repository:** geg00/test
**Type:** Test/Starter Repository
**Status:** Early Development

This is a minimal test repository currently in its initial stages. It serves as a foundation that can be expanded as the project grows.

## Project Structure

```
/home/user/test/
├── .git/              # Git repository data
├── README.md          # Project readme
└── CLAUDE.md          # This file - AI assistant guidance
```

## Current State

This repository is currently minimal, containing only:
- `README.md` - Basic project description

## Development Workflow

### Git Workflow

1. **Branching Strategy:**
   - Work on feature branches prefixed with `claude/` for AI-assisted development
   - Use descriptive branch names (e.g., `claude/add-feature-name-sessionId`)

2. **Commit Guidelines:**
   - Write clear, descriptive commit messages
   - Use present tense ("Add feature" not "Added feature")
   - Keep commits focused and atomic

3. **Before Pushing:**
   - Verify changes are on the correct branch
   - Review all modified files
   - Ensure no sensitive data is committed

### Commands Reference

```bash
# Check repository status
git status

# View commit history
git log --oneline

# Create and switch to a new branch
git checkout -b <branch-name>

# Push changes
git push -u origin <branch-name>
```

## Conventions for AI Assistants

### General Guidelines

1. **Read Before Modify:** Always read existing files before making changes
2. **Minimal Changes:** Make only the changes necessary to complete the task
3. **Preserve Style:** Match the existing code style and conventions
4. **No Over-Engineering:** Keep solutions simple and focused
5. **Security First:** Never commit sensitive data, credentials, or secrets

### File Handling

- Prefer editing existing files over creating new ones
- Use appropriate file extensions for the content type
- Add meaningful comments only where logic isn't self-evident

### Communication

- Be concise and direct in responses
- Reference specific file paths and line numbers when discussing code
- Ask clarifying questions when requirements are ambiguous

## Future Development

As this repository grows, this document should be updated to include:

- [ ] Technology stack details
- [ ] Build and test commands
- [ ] Code style guidelines
- [ ] Architecture documentation
- [ ] Deployment procedures
- [ ] Environment setup instructions

## Troubleshooting

### Common Issues

1. **Push Failures:**
   - Verify branch name starts with `claude/`
   - Check network connectivity
   - Retry with exponential backoff (2s, 4s, 8s, 16s)

2. **Merge Conflicts:**
   - Pull latest changes before making modifications
   - Resolve conflicts carefully, preserving intended functionality

---

*Last Updated: January 2026*
