# Minecraft Mob Detection

Computer vision project for detecting Minecraft mobs.

## Prerequisites

* [uv](https://docs.astral.sh/uv/)

## Project setup

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

## Training setup

1. Create an account on [Roboflow](https://roboflow.com/) and generate an API key.
2. Create an account on [Weights & Biases](https://wandb.ai/site) and generate an API key.
3. Copy `.env.example` into `.env`.
4. Copy the API keys into `.env`.
5. Enable Weights and Biases logging for Ultralytics:

   ```bash
   uv run yolo settings wandb=True
   ```

## Dataset

See the dataset at [Roboflow](https://universe.roboflow.com/minecraft-object-detection/minecraft-mob-detection).

To download:

```bash
uv run python scripts/download_data.py
```

## Training

To train the baseline model based on yolov8n run:

```bash
uv run python scripts/train_yolov8n.py
```
