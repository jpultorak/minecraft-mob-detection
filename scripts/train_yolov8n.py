import torch
import wandb
from dotenv import load_dotenv
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


def main():
    load_dotenv()
    wandb.login()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = YOLO("yolov8n.pt")

    model.train(
        data="data/data.yaml",
        epochs=50,
        imgsz=640,
        device=device,
        project="mcdetect",
        name="yolov8n-baseline",
    )


if __name__ == "__main__":
    main()
