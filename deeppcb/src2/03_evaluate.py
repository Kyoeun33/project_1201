"""Score the sealed test originals once per trained model and reuse the saved predictions."""
from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import tempfile
from pathlib import Path

import numpy as np
from filelock import FileLock

from .common import DEFAULT_CONFIG, EXPERIMENTS, config, load_prepared, read_json, save_json, sha256
from .inputs import TileDataset, input_cache, loader
from .metrics import image_scores, save_predictions
from .model import device_for, restore, score_loader
protocol_id = import_module(".02_train", __package__).protocol_id


def evaluate(cfg, experiments):
    _, originals, tiles, report = load_prepared(cfg)
    parents = originals[originals.split == "test"]
    grid = tiles[tiles.split == "test"]
    ids = "\n".join(sorted(parents.image_id)).encode()
    test_key = hashlib.sha256(ids).hexdigest()
    identity = protocol_id(cfg, report)
    device = device_for(cfg["training"]["device"])
    batches = None
    for experiment in experiments:
        run = Path(cfg["output_dir"]) / "runs" / experiment
        if not (run / "run.json").is_file():
            raise FileNotFoundError(f"First run python -m src2.02_train --experiments {experiment}")
        with FileLock(str(run / "test.lock")):
            info = read_json(run / "run.json")
            weights = run / "best.pt"
            if info["protocol_id"] != identity or sha256(weights) != info["weights_sha256"]:
                raise ValueError(f"Model or protocol differs from the sealed run: {run}")
            dest = run / "test"
            if dest.exists():
                done = read_json(dest / "evaluation.json")
                if (done["test_ids_sha256"] != test_key or done["weights_sha256"] != info["weights_sha256"]
                        or any(sha256(dest / name) != value for name, value in done["files"].items())):
                    raise ValueError(f"Saved evaluation does not match: {dest}")
                print(f"{experiment}: using saved final test results in {dest}")
                continue
            model, checkpoint = restore(weights, device)
            if checkpoint["protocol_id"] != identity:
                raise ValueError("Checkpoint was trained with a different manifest or protocol")
            if batches is None:
                cache = input_cache(cfg, originals, tiles, report)
                batches = loader(TileDataset(grid, originals, cfg, cache), cfg, device)
            probs = score_loader(model, batches, device, checkpoint["training"]) if len(grid) else np.empty(0)
            pred = image_scores(parents, grid, probs)
            with tempfile.TemporaryDirectory(dir=run) as tmp:
                directory = Path(tmp) / "test"
                directory.mkdir()
                save_predictions(directory, pred, checkpoint["threshold"])
                save_json(directory / "evaluation.json", dict(protocol_id=identity, test_ids_sha256=test_key,
                          weights_sha256=info["weights_sha256"], originals=len(parents), all_tiles=len(grid),
                          operating_threshold=checkpoint["threshold"], threshold_source="validation",
                          files={name: sha256(directory / name) for name in ("predictions.csv", "metrics.csv")}))
                directory.rename(dest)
            print(f"{experiment}: evaluated {len(pred)} test originals ({len(grid)} tiles), saved to {dest}",
                  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    evaluate(config(args.config), args.experiments)
