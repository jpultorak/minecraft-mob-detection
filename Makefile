.PHONY: help setup data train-yolov8n train-rtdetr train-yolo26m resume detect-video slurm-rtdetr-l slurm-yolov8m slurm-yolo26m

help:
	@echo "Available commands:"
	@echo "  make setup           - Run uv sync and configure Weights & Biases for Ultralytics"
	@echo "  make data            - Download, filter to 5 classes, and rebalance the dataset splits"
	@echo "  make train-yolov8n   - Train YOLOv8n baseline (configured for 8GB system RAM & RTX 3060 VRAM)"
	@echo "  make train-rtdetr    - Train RT-DETR model (configured for 8GB system RAM & RTX 3060 VRAM)"
	@echo "  make train-yolo26m   - Train YOLO26m model (configured for local run)"
	@echo "  make resume          - Resume training from the last checkpoint"
	@echo "  make detect-video VIDEO=path/to/video.mp4  - Run mob detection on a video"
	@echo "  make slurm-rtdetr-l  - Submit RT-DETR-L training on Slurm (2x RTX 3080)"
	@echo "  make slurm-yolov8m   - Submit YOLOv8m training on Slurm (2x RTX 3080)"
	@echo "  make slurm-yolo26m   - Submit YOLO26m training on Slurm (2x RTX 3080)"

setup:
	uv sync
	uv run yolo settings wandb=True

data:
	uv run python scripts/download_data.py

train-yolov8n:
	# Running YOLOv8n training (default batch=8, 50 epochs)
	uv run python scripts/train.py --model weights/yolov8n.pt --batch 8 --epochs 50 --name yolov8n-baseline

train-rtdetr:
	# Running RT-DETR training (default batch=2, 100 epochs)
	uv run python scripts/train.py --model weights/rtdetr-x.pt --batch 2 --epochs 100 --name rtdetr-x

train-yolo26m:
	# Running YOLO26m training (default batch=8, 50 epochs)
	uv run python scripts/train.py --model yolo26m.pt --batch 8 --epochs 50 --name yolo26m-baseline

detect-video:
	uv run python scripts/detect_video.py $(VIDEO)

slurm-rtdetr-l:
	sbatch --export=ALL,MODEL=weights/rtdetr-l.pt,BATCH=16,EPOCHS=100,RUN_NAME=rtdetr-l-2gpu \
		scripts/train_slurm.sh

slurm-yolov8m:
	sbatch --export=ALL,MODEL=weights/yolov8m.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolov8m-2gpu \
		scripts/train_slurm.sh

slurm-yolo26m:
	sbatch --export=ALL,MODEL=yolo26m.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolo26m-2gpu \
		scripts/train_slurm.sh
