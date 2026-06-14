# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path


def sanitize_path(path_str: str) -> str:
    """Force lowercase ziob to capitalized Ziob to prevent permission errors on cluster nodes."""
    if path_str.startswith("/ziob/"):
        return "/Ziob/" + path_str[6:]
    return path_str


# Set up local/isolated Ultralytics configuration directory.
# This prevents environment-specific path issues across different machines/clusters,
# and avoids loading stale paths (like /ziob) from the global settings.json.
# We must configure this BEFORE importing ultralytics.
project_dir = Path(__file__).absolute().parent.parent
project_dir_str = sanitize_path(str(project_dir))
config_dir = Path(project_dir_str) / ".ultralytics"
config_dir.mkdir(exist_ok=True)

settings_path = config_dir / "settings.json"
settings_data = {}
if settings_path.exists():
    try:
        with open(settings_path) as f:
            settings_data = json.load(f)
    except Exception:
        pass

# Force correct runs and weights paths for the current environment
settings_data["runs_dir"] = sanitize_path(str(Path(project_dir_str) / "runs"))
settings_data["weights_dir"] = sanitize_path(str(Path(project_dir_str) / "weights"))
settings_data["wandb"] = True

with open(settings_path, "w") as f:
    json.dump(settings_data, f, indent=2)

os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

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
