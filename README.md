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

## Training setup
1. Copy `.env.example` into `.env`
2. Create an account on [Roboflow](https://roboflow.com/).
3. Create an account on [Weights & Biases](https://wandb.ai/site).
4. Generate and copy the API keys into `.env`


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
