"""Run object detection on a video and save annotated output."""

import argparse
from pathlib import Path

from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]


def main() -> None:
    p = argparse.ArgumentParser(description="Detect Minecraft mobs in a video.")
    p.add_argument("source", help="Path to input video (or 0 for webcam).")
    p.add_argument("--model", default="weights/rtdetr-x.pt", help="Model weights path.")
    p.add_argument("--conf", type=float, default=0.4, help="Confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--device", default="auto", help="Device: auto, cuda, cpu, mps.")
    args = p.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    device = args.device if args.device != "auto" else None  # let ultralytics pick

    model = YOLO(args.model)
    results = model.predict(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        save=True,
        stream=True,  # memory-efficient frame-by-frame generator
    )

    # Consume the generator so all frames are processed
    n = sum(1 for _ in results)

    predictor = model.predictor
    if predictor is None:
        raise RuntimeError("YOLO predictor was not initialized after predict()")
    out_dir = Path(predictor.save_dir)
    print(f"Done — {n} frames processed. Output saved to {out_dir}")


if __name__ == "__main__":
    main()
