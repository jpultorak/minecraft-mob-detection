from collections import Counter
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent.parent / "data"


def count_class_instances():
    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.is_file():
        print("Missing data.yaml")
        return

    with open(yaml_path) as f:
        config = yaml.safe_load(f)
    class_names = config.get("names", [])

    print(f"{'Split':<10} | {'Class Name':<20} | {'Instances':<10}")
    print("-" * 48)

    for split in ["train", "valid", "test"]:
        label_dir = DATA_DIR / split / "labels"
        if not label_dir.is_dir():
            continue

        counts = Counter()
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1

        # Print sorted by frequency
        for class_id, count in counts.most_common():
            name = class_names[class_id] if class_id < len(class_names) else f"ID-{class_id}"
            print(f"{split:<10} | {name:<20} | {count:<10}")
        print("-" * 48)


if __name__ == "__main__":
    count_class_instances()
