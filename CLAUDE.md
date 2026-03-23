# Claude Code project notes

## Package management
- Use `uv` for all package installs: `uv pip install <package>`
- Do NOT use bare `pip install`

## Running tests
- Use the venv python: `.venv/Scripts/python -m pytest tests/ -v --tb=short`
- System python (`python`) does NOT have project deps installed
