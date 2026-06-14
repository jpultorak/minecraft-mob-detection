.PHONY: help setup data resume detect-video slurm-rtdetr-l slurm-yolov8n slurm-yolo26m

help:
	@echo "Available commands:"
	@echo "  make setup           - Run uv sync and configure Weights & Biases for Ultralytics"
	@echo "  make data            - Download, filter to 5 classes, and rebalance the dataset splits"
	@echo "  make resume          - Resume training from the last checkpoint"
	@echo "  make detect-video VIDEO=path/to/video.mp4  - Run mob detection on a video"
	@echo "  make slurm-rtdetr-l  - Submit RT-DETR-L training on Slurm (2x RTX 3080)"
	@echo "  make slurm-yolov8n   - Submit YOLOv8n training on Slurm (2x RTX 3080)"
	@echo "  make slurm-yolo26m   - Submit YOLO26m training on Slurm (2x RTX 3080)"

setup:
	uv sync
	uv run yolo settings wandb=True

data:
	uv run python scripts/download_data.py

detect-video:
	uv run python scripts/detect_video.py $(VIDEO)

slurm-rtdetr-l:
	sbatch --export=ALL,MODEL=weights/rtdetr-l.pt,BATCH=16,EPOCHS=100,RUN_NAME=rtdetr-l-2gpu \
		scripts/train_slurm.sh

slurm-yolov8n:
	sbatch --export=ALL,MODEL=weights/yolov8n.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolov8n-2gpu \
		scripts/train_slurm.sh

slurm-yolo26m:
	sbatch --export=ALL,MODEL=yolo26m.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolo26m-2gpu \
		scripts/train_slurm.sh
