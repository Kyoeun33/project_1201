"""Configuration, manifest identity, and portable artifact I/O."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
CLASSES = {"0": "normal", "1": "defect"}
SOURCES = ("DeepPCB", "PKU", "VisA")
BOARDS = ("pcb1", "pcb2", "pcb3", "pcb4")
SPLITS = ("train", "validation", "test")
EXPERIMENTS = dict(A=SOURCES[:1], B=SOURCES[:2], C=SOURCES)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                          + "\n", encoding="utf-8")


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    cfg = read_json(path)
    cfg["paths"] = {k: str((path.parent / v).resolve()) for k, v in cfg["paths"].items()}
    cfg["output_dir"] = str((path.parent / cfg["output_dir"]).resolve())
    sources = cfg["sources"]
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("sources must be a nonempty selection of DeepPCB, PKU, VisA")
    for source in sources:
        if source not in cfg["paths"]:
            raise ValueError(f"Set paths.{source} in {path}")
    fractions = cfg["split_fractions"]
    if len(fractions) != 3 or min(fractions) <= 0 or not math.isclose(sum(fractions), 1):
        raise ValueError("split_fractions must be three positive fractions summing to 1")
    tile = cfg["tiling"]
    if any(type(tile[k]) is not int for k in ("size", "overlap")):
        raise ValueError("Tile size and overlap must be integer pixel counts")
    if not 0 <= tile["overlap"] < tile["size"] or not 0 <= tile["min_defect_coverage"] <= 1:
        raise ValueError("Require 0 <= overlap < size and 0 <= min_defect_coverage <= 1")
    tr = cfg["training"]
    if min(tr[k] for k in ("image_size", "batch_size", "epochs", "learning_rate")) <= 0:
        raise ValueError("image_size, batch_size, epochs and learning_rate must be positive")
    if tr["workers"] < 0 or tr["weight_decay"] < 0 or not 0 <= tr["binary_cutoff"] <= 255:
        raise ValueError("Invalid workers, weight_decay or binary_cutoff")
    for key, options in dict(model=("resnet18", "efficientnet_b0"), optimizer=("adamw", "sgd"),
                             preprocessing=("rgb", "binary"), device=("auto", "cpu", "cuda"),
                             imbalance=("none", "class_weight", "weighted_sampler")).items():
        if tr[key] not in options:
            raise ValueError(f"training.{key} must be one of {options}")
    if not all(0 <= cfg["threshold"][k] <= 1 for k in ("min_recall", "max_fpr")):
        raise ValueError("min_recall and max_fpr must be probabilities")
    return cfg


def preparation_settings(cfg):
    return {k: cfg[k] for k in ("sources", "seed", "split_fractions", "tiling")}


def load_prepared(cfg):
    directory = Path(cfg["output_dir"]) / "prepared"
    if not (directory / "summary.json").is_file():
        raise FileNotFoundError(f"Run python -m src2.01_prepare first. Missing {directory}")
    report = read_json(directory / "summary.json")
    if preparation_settings(cfg) != report["settings"]:
        raise ValueError("Prepared settings differ. Use the original config or a new output_dir.")
    for name, expected in report["manifest_sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError(f"Manifest changed: {directory / name}")
    originals = pd.read_csv(directory / "originals.csv", keep_default_na=False, dtype={"board": str})
    tiles = pd.read_csv(directory / "tiles.csv", keep_default_na=False)
    return directory, originals, tiles, report
