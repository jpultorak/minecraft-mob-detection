# Minecraft Mob Detection

Computer vision project for detecting Minecraft mobs.

## Prerequisites
* [uv](https://docs.astral.sh/uv/)

## Setup

1. Clone the repository and navigate into it:
   ```bash
   git clone git@github.com:jpultorak/minecraft-mob-detection.git
   cd minecraft-mob-detection
   ```

2. Install dependencies and create the virtual environment:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Dataset

Dataset from [Roboflow](https://universe.roboflow.com/minecraft-object-detection/minecraft-mob-detection).

To download:
```bash
uv run python scripts/download_data.py
```
