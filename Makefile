# Makefile for Minecraft Mob Detection training and data preparation

.PHONY: help data train-yolov8n train-rtdetr detect-video

help:
	@echo "Available commands:"
	@echo "  make data            - Download, filter to 5 classes, and rebalance the dataset splits"
	@echo "  make train-yolov8n   - Train YOLOv8n baseline (configured for 8GB system RAM & RTX 3060 VRAM)"
	@echo "  make train-rtdetr    - Train RT-DETR model (configured for 8GB system RAM & RTX 3060 VRAM)"
	@echo "  make detect-video VIDEO=path/to/video.mp4  - Run mob detection on a video"

data:
	uv run python scripts/download_data.py

train-yolov8n:
	# Running YOLOv8n training (default batch=8, 50 epochs)
	uv run python scripts/train.py --model weights/yolov8n.pt --batch 8 --epochs 50 --name yolov8n-baseline

train-rtdetr:
	# Running RT-DETR training (default batch=2, 100 epochs)
	uv run python scripts/train.py --model weights/rtdetr-x.pt --batch 2 --epochs 100 --name rtdetr-x

detect-video:
	uv run python scripts/detect_video.py $(VIDEO)
