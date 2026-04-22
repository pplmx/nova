# Agent Guidelines

## Commit Messages

- **Language:** English only
- **Body:** Always include detailed body explaining the "why", not just the "what"
- **Format:**

```
<type>: <short summary>

<detailed body explaining motivation, changes, and any trade-offs made>

<optional: breaking changes, related issues>
```

## Tools

- Use `fd` instead of `find` for file searches
- Use `rg` instead of `grep` for content searches
- Use `ls` for directory listing

## Git Workflow

- **No worktrees** - work directly on main branch for linear git log
- Create feature branches only when necessary
- Squash/merge commits when appropriate

## Code Style

- C++20 with CUDA 17
- Follow existing patterns in the codebase
- No emojis in code or comments
