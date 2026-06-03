from dataset import ensure_dataset


def main() -> None:
    data_dir = ensure_dataset()
    print(f"Dataset ready at {data_dir}")


if __name__ == "__main__":
    main()
