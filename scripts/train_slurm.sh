#!/bin/bash

# ── Slurm directives ───────────────────────────────────────────────
#SBATCH --job-name=mcdetect-train
#SBATCH --output=output/%j_train.out
#SBATCH --error=output/%j_train.err
#SBATCH --partition=student-nvidia
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────
CLUSTER_USER=${MCDETECT_CLUSTER_USER:-ijakus}
USER_DIRECTORY=/Ziob/$CLUSTER_USER
PROJECT_ROOT=$USER_DIRECTORY/minecraft-mob-detection

source ~/.bashrc
cd "$PROJECT_ROOT"

mkdir -p output

export XDG_RUNTIME_DIR=/tmp/$CLUSTER_USER/runtime
export UV_CACHE_DIR=$USER_DIRECTORY/.cache/uv
export UV_LINK_MODE=copy
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$XDG_RUNTIME_DIR" "$UV_CACHE_DIR"

# ── Virtualenv & deps ─────────────────────────────────────────────
test -d .venv || uv venv .venv --python 3.14
uv sync

export YOLO_CONFIG_DIR="$PROJECT_ROOT/.ultralytics"
uv run yolo settings wandb=True

# ── Defaults (overridable via --export) ────────────────────────────
MODEL=${MODEL:-weights/rtdetr-l.pt}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-100}
RUN_NAME=${RUN_NAME:-rtdetr-l-2gpu}

echo "═══════════════════════════════════════════════"
echo "  Model:  $MODEL"
echo "  Batch:  $BATCH"
echo "  Epochs: $EPOCHS"
echo "  Name:   $RUN_NAME"
echo "  GPUs:   $SLURM_GPUS_ON_NODE"
echo "═══════════════════════════════════════════════"

# ── Train ──────────────────────────────────────────────────────────
uv run python scripts/train.py \
    --model "$MODEL" \
    --batch "$BATCH" \
    --epochs "$EPOCHS" \
    --name "$RUN_NAME" \
    --device 0,1
