# Human Development Guide (GFN Framework)

This document defines development practices for human contributors to the GFN project. The goal is to maintain a professional, scalable codebase free from avoidable technical errors.

## 1. The Development Cycle

To ensure "slow and controlled" development, we follow these steps:

1. **Synchronization**: Before starting, ensure you are on `dev` and have the latest changes.
2. **Isolation**: Create a branch for your task (`feat/` or `fix/`). Never work directly on `main` or `dev`.
3. **Atomic Development**: Make small, focused changes. If a task is too large, break it down into sub-tasks.
4. **Rigorous Validation**: 
   - **Specific Test**: Run at least one test that directly verifies your change.
   - **Full Suite**: Run the entire test suite (excluding benchmarks) to ensure no regressions.
     - Command: `python -m pytest tests/ --ignore=tests/gssm/benchmarks --ignore=tests/isn/benchmarks`
5. **Review**: In a team environment, open a Pull Request (PR) towards `dev`.

## 2. "Slow and Controlled" Development
- **Quality > Speed**: It is preferable to take an extra day and deliver tested code than to fix bugs in production.
- **Continuous Refactoring**: If you touch a file and see something that can be improved without breaking anything, do it (but in a separate commit if possible).
- **No Blind Hotfixes**: Every fix must be tested in a branch before merging.

## 3. Integration with the AI Agent
The project has an internal memory system in the `/memory/` folder.
- **Note**: As a human developer, **you do not need to edit files in `/memory/`**.
- The AI Agent is responsible for keeping that log updated based on your changes and our pair-programming sessions. Your focus should be on the code in `gfn/` and the documentation in `docs/`.
