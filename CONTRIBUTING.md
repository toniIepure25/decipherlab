# Contributing to DecipherLab

Thank you for helping improve DecipherLab.

## Setup

- Install dependencies with `uv sync`.
- Run the test suite with `uv run pytest`.
- For local validation, use `python3 -m py_compile $(find src -name '*.py')` if you want a quick syntax check.

## Working Guidelines

- Keep changes focused and auditable.
- Prefer small, well-described pull requests.
- Update documentation when behavior changes.
- Do not commit generated data, run outputs, or other artifacts that belong under ignored paths.

## Before Opening a Pull Request

- Ensure tests pass locally.
- Confirm new config files are valid.
- Mention any limitations, caveats, or heuristic behavior in the PR description.
- Include screenshots or output excerpts when they help reviewers understand the change.
