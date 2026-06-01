import random
import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
ROBOFLOW_FORMAT = "yolov11"

TARGET_CLASSES = ["pig", "chicken", "cow", "creeper", "sheep"]


def is_yolo_dataset(data_dir: Path) -> bool:
    return (
        (data_dir / "data.yaml").is_file()
        and (data_dir / "train" / "images").is_dir()
        and (data_dir / "train" / "labels").is_dir()
        and (data_dir / ".balanced").exists()
    )


def _rebalance_dataset(data_dir: Path) -> None:
    with open(data_dir / "data.yaml") as f:
        cfg = yaml.safe_load(f)

    old_names = cfg.get("names", [])
    id_map = {
        old_names.index(name): i for i, name in enumerate(TARGET_CLASSES) if name in old_names
    }

    # Rename existing splits to avoid collision during redistribution
    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            split_dir.rename(data_dir / f"{split}_old")

    dataset_items = []
    for split in ["train", "valid", "test"]:
        old_img_dir = data_dir / f"{split}_old" / "images"
        if not old_img_dir.exists():
            continue

        for img_path in old_img_dir.glob("*"):
            lbl_path = data_dir / f"{split}_old" / "labels" / f"{img_path.stem}.txt"
            new_labels = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and int(parts[0]) in id_map:
                            parts[0] = str(id_map[int(parts[0])])
                            new_labels.append(" ".join(parts) + "\n")

            # Keep the image only if it contains at least one of the target classes
            if new_labels:
                dataset_items.append((img_path, new_labels))

    # Shuffle for a uniform distribution
    random.seed(42)
    random.shuffle(dataset_items)

    n = len(dataset_items)
    splits = {
        "train": dataset_items[: int(n * 0.7)],
        "valid": dataset_items[int(n * 0.7) : int(n * 0.85)],
        "test": dataset_items[int(n * 0.85) :],
    }

    for split_name, items in splits.items():
        img_dir = data_dir / split_name / "images"
        lbl_dir = data_dir / split_name / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for img_path, labels in items:
            shutil.move(str(img_path), img_dir / img_path.name)
            with open(lbl_dir / f"{img_path.stem}.txt", "w") as f:
                f.writelines(labels)

    # Clean up the original unaligned directories
    for split in ["train", "valid", "test"]:
        old_split = data_dir / f"{split}_old"
        if old_split.exists():
            shutil.rmtree(old_split)

    with open(data_dir / "data.yaml", "w") as f:
        yaml.dump(
            {
                "names": TARGET_CLASSES,
                "nc": len(TARGET_CLASSES),
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
            },
            f,
        )

    (data_dir / ".balanced").touch()


def ensure_dataset(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    load_dotenv()
    if is_yolo_dataset(data_dir):
        return data_dir

    if data_dir.exists():
        print(f"Removing incompatible or unaligned dataset at {data_dir}")
        shutil.rmtree(data_dir)

    version = (
        Roboflow()
        .workspace("minecraft-object-detection")
        .project("minecraft-mob-detection")
        .version(10)
    )
    version.download(ROBOFLOW_FORMAT, location=str(data_dir), overwrite=True)

    print("Rebalancing dataset and filtering to the target classes...")
    _rebalance_dataset(data_dir)

    if not is_yolo_dataset(data_dir):
        msg = f"Download finished but {data_dir} failed validation after rebalance."
        raise RuntimeError(msg)

    return data_dir
