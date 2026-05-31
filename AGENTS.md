# Development Guidelines

## Package Management
- Always use `uv add <package>` instead of `pip install`
- Always run scripts with `uv run python scripts/<script>.py` instead of `python scripts/<script>.py`
- OpenCV: `opencv-python` is excluded in `pyproject.toml` (avoids `libxcb` on Linux). Use `uv sync` so only `opencv-python-headless` is installed.

## Code Quality
- Run linting: `uv run ruff check .`

## Git
- Never commit secrets, API keys, or credentials
- Always ask before creating commits
