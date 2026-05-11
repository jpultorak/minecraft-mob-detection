import torch
import wandb
from dotenv import load_dotenv
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


def main():
    load_dotenv()
    wandb.login()

    if torch.cuda.is_available():
        compute_device = "cuda"
    elif torch.backends.mps.is_available():
        compute_device = "mps"
    else:
        compute_device = "cpu"

    model = YOLO("weights/yolov8n.pt")

    model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        device=compute_device,
        project="mcdetect",
        name="yolov8n-baseline",
    )


if __name__ == "__main__":
    main()
