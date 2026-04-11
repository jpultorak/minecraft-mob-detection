# Development Guidelines

## Package Management
- Always use `uv add <package>` instead of `pip install`
- Always run scripts with `uv run python scripts/<script>.py` instead of `python scripts/<script>.py`

## Code Quality
- Run linting: `uv run ruff check .`

## Git
- Never commit secrets, API keys, or credentials
- Always ask before creating commits
