import argparse

import torch
from dataset import ensure_dataset
from dotenv import load_dotenv
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on the Minecraft mob dataset.")
    parser.add_argument(
        "--model",
        default="weights/rtdetr-x.pt",
        help="Ultralytics RT-DETR weights (e.g. weights/rtdetr-l.pt, weights/rtdetr-x.pt).",
    )
    parser.add_argument("--data", default="data/data.yaml", help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument(
        "--patience", type=int, default=20, help="Early-stopping patience (epochs)."
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (lower if OOM).")
    parser.add_argument("--project", default="mcdetect", help="WandB / Ultralytics project name.")
    parser.add_argument("--name", default="rtdetr-x", help="Run name.")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cpu', or device index.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str | int:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()
    load_dotenv()
    wandb.login()
    ensure_dataset()

    device = resolve_device(args.device)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
