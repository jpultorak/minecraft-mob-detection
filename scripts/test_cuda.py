import os

import torch


def main() -> None:
    print(f"PyTorch version: {torch.__version__}")
    print(f"Built with CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        total_gb = props.total_memory / 1024**3
        print(f"Device {index}: {props.name} ({total_gb:.1f} GiB)")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        tensor = torch.ones((1024, 1024), device=device)
        result = tensor @ tensor
        torch.cuda.synchronize()
        print(f"CUDA smoke test: ok, result sum={result.sum().item():.1f}")
    else:
        print("CUDA smoke test: skipped because PyTorch cannot see a CUDA device.")


if __name__ == "__main__":
    main()
