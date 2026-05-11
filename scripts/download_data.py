import zipfile
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent


def main() -> None:
    rf = Roboflow()
    project = rf.workspace("minecraft-object-detection").project("minecraft-mob-detection")
    version = project.version(10)

    data_dir = PROJECT_DIR / "data"
    version.download("yolov8", location=str(data_dir))

    zip_path = data_dir / "roboflow.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        zip_path.unlink()


if __name__ == "__main__":
    main()
