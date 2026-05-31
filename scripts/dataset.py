import shutil
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
ROBOFLOW_FORMAT = "yolov11"


def is_yolo_dataset(data_dir: Path) -> bool:
    return (
        (data_dir / "data.yaml").is_file()
        and (data_dir / "train" / "images").is_dir()
        and (data_dir / "train" / "labels").is_dir()
    )


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    load_dotenv()
    if is_yolo_dataset(data_dir):
        return data_dir

    if data_dir.exists():
        print(f"Removing incompatible dataset at {data_dir} (expected YOLO layout with data.yaml)")
        shutil.rmtree(data_dir)

    version = (
        Roboflow()
        .workspace("minecraft-object-detection")
        .project("minecraft-mob-detection")
        .version(10)
    )
    # Roboflow skips download when `location` already exists; do not mkdir beforehand.
    version.download(ROBOFLOW_FORMAT, location=str(data_dir), overwrite=True)

    if not is_yolo_dataset(data_dir):
        msg = f"Download finished but {data_dir} is missing data.yaml or train/images|labels"
        raise RuntimeError(msg)

    return data_dir
