import argparse
from pathlib import Path

import torch
import torchvision
import yaml
from dataset import DEFAULT_DATA_DIR, PROJECT_DIR
from PIL import Image
from torchvision.utils import draw_bounding_boxes
from train_faster_rcnn_pl import MCDetModule

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Faster R-CNN inference and save visualizations."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "weights" / "faster_rcnn_pl_50.ckpt",
        help="Path to a Lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_DATA_DIR / "test" / "images",
        help="Image file or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs" / "faster-rcnn-predictions",
        help="Directory where annotated images are saved.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_DIR / "data.yaml",
        help="Dataset yaml with class names.",
    )
    parser.add_argument(
        "--max-size", type=int, default=640, help="Longest image side (same as training)."
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, help="Minimum detection score."
    )
    parser.add_argument("--limit", type=int, default=0, help="Max images to process (0 = all).")
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', or 'cpu'.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_class_names(data_yaml: Path) -> list[str]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        msg = f"Source not found: {source}"
        raise FileNotFoundError(msg)
    images = sorted(
        p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    if not images:
        msg = f"No images found in {source}"
        raise FileNotFoundError(msg)
    return images


def load_image_tensor(path: Path, max_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h), Image.BILINEAR)
    return torchvision.transforms.functional.to_tensor(img)


def predict_image(
    module: MCDetModule,
    image: torch.Tensor,
    device: torch.device,
    score_threshold: float,
) -> dict[str, torch.Tensor]:
    module.eval()
    with torch.inference_mode():
        outputs = module.model([image.to(device)])[0]

    keep = outputs["scores"] >= score_threshold
    return {
        "boxes": outputs["boxes"][keep].cpu(),
        "labels": outputs["labels"][keep].cpu(),
        "scores": outputs["scores"][keep].cpu(),
    }


def draw_predictions(
    image: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    class_names: list[str],
) -> Image.Image:
    label_names = {idx + 1: name for idx, name in enumerate(class_names)}
    captions = [
        f"{label_names.get(int(label), str(int(label)))} {score:.2f}"
        for label, score in zip(prediction["labels"], prediction["scores"], strict=True)
    ]
    annotated = draw_bounding_boxes(
        (image * 255).to(torch.uint8),
        prediction["boxes"],
        labels=captions if captions else None,
        width=2,
    )
    return torchvision.transforms.functional.to_pil_image(annotated)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.checkpoint.is_file():
        msg = f"Checkpoint not found: {args.checkpoint}"
        raise FileNotFoundError(msg)

    class_names = load_class_names(args.data_yaml)
    image_paths = collect_images(args.source)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)

    module = MCDetModule.load_from_checkpoint(args.checkpoint, map_location=device)
    module.to(device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Saving {len(image_paths)} image(s) to {args.output}")

    for image_path in image_paths:
        image = load_image_tensor(image_path, args.max_size)
        prediction = predict_image(module, image, device, args.score_threshold)
        annotated = draw_predictions(image, prediction, class_names)
        out_path = args.output / image_path.name
        annotated.save(out_path)
        n_det = len(prediction["scores"])
        print(f"  {image_path.name}: {n_det} detection(s) -> {out_path}")


if __name__ == "__main__":
    main()
