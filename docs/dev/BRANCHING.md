# Git and Branching Guide

Branching strategy to maintain stability of the Geodesic Flow Networks framework.

## Main Branches

- **`main`**: The "production-ready" state. Only stable versions (v2.7.x).
- **`dev`**: The main development branch. All features are integrated here first.

## Working Branches (Features/Fixes)

Use prefixes to identify the purpose of the branch:

- `feat/feature-name`: New capabilities.
- `fix/specific-error`: Bug fixes.
- `docs/documented-aspect`: Pure documentation improvements.
- `refactor/optimized-area`: Changes that do not alter behavior.

### Example command flow:

```bash
# 1. Start on dev
git checkout dev
git pull origin dev

# 2. Create working branch
git checkout -b feat/dynamic-integrator

# 3. (Development and commits...)

# 4. Integrate (if not using GitHub PRs)
git checkout dev
git merge feat/dynamic-integrator
git branch -d feat/dynamic-integrator
```

## Commit Messages

Use clear and descriptive messages. We recommend the format:
`type: brief description` (e.g., `fix: fix overflow in bilateral projector`).
