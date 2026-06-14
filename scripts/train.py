import argparse
from pathlib import Path

from dataset import ensure_dataset
from dotenv import load_dotenv
from ultralytics import YOLO, settings  # pyright: ignore[reportPrivateImportUsage]

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
    parser.add_argument(
        "--device",
        default="0",
        help="Device(s) to train on, e.g. '0' or '0,1' for multi-GPU.",
    )
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

    # Dynamically configure Ultralytics paths relative to the current working directory.
    # This prevents PermissionError on HPC clusters where global configuration files
    # might point to incorrect/outdated path structures (e.g. /ziob instead of /Ziob).
    current_project_dir = Path.cwd()
    updates = {}
    if settings.get("runs_dir") != str(current_project_dir / "runs"):
        updates["runs_dir"] = str(current_project_dir / "runs")
    if settings.get("weights_dir") != str(current_project_dir / "weights"):
        updates["weights_dir"] = str(current_project_dir / "weights")

    if updates:
        print(f"Updating Ultralytics settings: {updates}")
        settings.update(updates)

    wandb.login()
    ensure_dataset()

    # Parse device string: "0" -> 0, "0,1" -> [0, 1]
    if "," in args.device:
        device = [int(d) for d in args.device.split(",")]
    else:
        device = int(args.device)

    model = YOLO(args.model)
    model.add_callback("on_train_end", log_best_model)
    model.train(
        data="data/data.yaml",
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        project="mcdetect",
        workers=4 if isinstance(device, list) else 2,
        imgsz=640,
        device=device,
    )


if __name__ == "__main__":
    main()
