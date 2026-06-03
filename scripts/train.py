import argparse

from dataset import ensure_dataset
from dotenv import load_dotenv
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO or RT-DETR on the Minecraft mob dataset."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to Ultralytics model weights (e.g. weights/yolov8n.pt or weights/rtdetr-x.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--name", required=True, help="Run name for tracking.")
    return parser.parse_args()


def log_best_model(trainer) -> None:
    """Callback to upload only the best model weights at the end of training."""
    if trainer.best and trainer.best.exists():
        artifact = wandb.Artifact(name=f"{trainer.args.name}_best", type="model")
        artifact.add_file(str(trainer.best))
        wandb.log_artifact(artifact)
        print(f"Uploaded best model checkpoint to WandB: {trainer.best}")


def main() -> None:
    args = parse_args()
    load_dotenv()
    wandb.login()
    ensure_dataset()

    model = YOLO(args.model)
    model.add_callback("on_train_end", log_best_model)
    model.train(
        data="data/data.yaml",
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project="mcdetect",
        workers=2,  # Keep low to prevent system RAM OOM
        imgsz=640,
        device="auto",
    )


if __name__ == "__main__":
    main()
