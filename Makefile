# Fix UV_CACHE_DIR if it points to incorrect /ziob mount path on cluster
ifneq ($(findstring /ziob/,$(UV_CACHE_DIR)),)
    export UV_CACHE_DIR := $(subst /ziob/,/Ziob/,$(UV_CACHE_DIR))
endif

.PHONY: help setup data resume detect-video fetch-results presentation slurm-rtdetr-l slurm-rtdetr-x slurm-yolov8n slurm-yolo26m slurm-submit-all

help:
	@echo "Available commands:"
	@echo "  make setup           - Run uv sync and configure Weights & Biases for Ultralytics"
	@echo "  make data            - Download, filter to 5 classes, and rebalance the dataset splits"
	@echo "  make resume          - Resume training from the last checkpoint"
	@echo "  make detect-video VIDEO=path/to/video.mp4  - Run mob detection on a video"
	@echo "  make fetch-results   - Fetch W&B run metrics into presentation/results.json"
	@echo "  make presentation    - Fetch results and compile presentation/slides.pdf"
	@echo "  make slurm-rtdetr-l  - Submit RT-DETR-L training on Slurm (2x RTX 3090)"
	@echo "  make slurm-rtdetr-x  - Submit RT-DETR-X training on Slurm (2x RTX 3090)"
	@echo "  make slurm-yolov8n   - Submit YOLOv8n training on Slurm (2x RTX 3090)"
	@echo "  make slurm-yolo26m   - Submit YOLO26m training on Slurm (2x RTX 3090)"
	@echo "  make slurm-submit-all - Submit YOLOv8n, YOLO26m, RT-DETR-L, and RT-DETR-X training jobs to Slurm"

setup:
	uv sync
	uv run yolo settings wandb=True

data:
	uv run python scripts/download_data.py

detect-video:
	uv run python scripts/detect_video.py $(VIDEO)

fetch-results:
	uv run python scripts/fetch_results.py

presentation: fetch-results
	cd presentation && typst compile slides.typ
	@echo "Built presentation/slides.pdf"

slurm-rtdetr-l:
	sbatch --export=ALL,MODEL=weights/rtdetr-l.pt,BATCH=16,EPOCHS=100,RUN_NAME=rtdetr-l-2gpu \
		scripts/train_slurm.sh

slurm-rtdetr-x:
	sbatch --export=ALL,MODEL=rtdetr-x.pt,BATCH=8,EPOCHS=100,RUN_NAME=rtdetr-x-2gpu \
		scripts/train_slurm.sh

slurm-yolov8n:
	sbatch --export=ALL,MODEL=weights/yolov8n.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolov8n-2gpu \
		scripts/train_slurm.sh

slurm-yolo26m:
	sbatch --export=ALL,MODEL=yolo26m.pt,BATCH=32,EPOCHS=100,RUN_NAME=yolo26m-2gpu \
		scripts/train_slurm.sh

slurm-submit-all:
	$(MAKE) slurm-yolov8n
	$(MAKE) slurm-yolo26m
	$(MAKE) slurm-rtdetr-l
	$(MAKE) slurm-rtdetr-x
