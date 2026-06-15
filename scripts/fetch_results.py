"""Fetch final metrics for W&B runs and write them to presentation/results.json.

Reads the run list from presentation/runs.json (entity, project, run paths),
queries the W&B public API, and stores a compact JSON the Typst presentation
reads directly. Run with: uv run python scripts/fetch_results.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

import wandb

PRESENTATION_DIR = Path(__file__).absolute().parent.parent / "presentation"
RUNS_CONFIG = PRESENTATION_DIR / "runs.json"
RESULTS_OUT = PRESENTATION_DIR / "results.json"

# Ultralytics -> W&B summary keys. (B) = box metrics.
METRIC_KEYS = {
    "map50": ["metrics/mAP50(B)", "metrics/mAP_0.5", "metrics/mAP50"],
    "map5095": ["metrics/mAP50-95(B)", "metrics/mAP_0.5:0.95", "metrics/mAP50-95"],
    "precision": ["metrics/precision(B)", "metrics/precision"],
    "recall": ["metrics/recall(B)", "metrics/recall"],
    "inference_ms": ["model/speed_PyTorch(ms)"],
}


def pick(summary: dict, candidates: list[str]) -> float | None:
    for key in candidates:
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def runtime_minutes(run) -> float | None:
    runtime = run.summary.get("_runtime")
    if isinstance(runtime, (int, float)):
        return round(runtime / 60.0, 1)
    return None


def inference_fps(summary: dict) -> float | None:
    ms = pick(summary, METRIC_KEYS["inference_ms"])
    if ms and ms > 0:
        return round(1000.0 / ms, 1)
    return None


def main() -> None:
    load_dotenv()
    config = json.loads(RUNS_CONFIG.read_text())
    api = wandb.Api()

    results = []
    for entry in config["runs"]:
        path = entry["path"]
        label = entry["label"]
        try:
            run = api.run(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not fetch {path}: {exc}")
            results.append({"label": label, "path": path, "error": str(exc)})
            continue

        summary = dict(run.summary)
        metrics = {name: pick(summary, keys) for name, keys in METRIC_KEYS.items()}
        fps = inference_fps(summary)
        results.append(
            {
                "label": label,
                "path": path,
                "state": run.state,
                "runtime_min": runtime_minutes(run),
                "epochs": run.config.get("epochs"),
                "fps": fps,
                **{k: v for k, v in metrics.items() if k != "inference_ms"},
            }
        )
        print(
            f"[ok] {label}: mAP50={metrics['map50']} "
            f"mAP50-95={metrics['map5095']} fps={fps} runtime={runtime_minutes(run)}m"
        )

    RESULTS_OUT.write_text(json.dumps({"runs": results}, indent=2) + "\n")
    print(f"\nWrote {RESULTS_OUT}")


if __name__ == "__main__":
    main()
