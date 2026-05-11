import torch
from dotenv import load_dotenv
from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]

load_dotenv()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = YOLO("weights/yolov8n.pt")

model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    project="mcdetect",
    name="yolov8n-baseline",
    device=device,
)
