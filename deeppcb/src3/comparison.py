"""Compare saved image-level results for matching ResNet18 and EfficientNet-B0 runs."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from filelock import FileLock

from src2.common import EXPERIMENTS, preparation_settings, read_json, sha256

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "fpr", "fnr"]
COLUMNS = ["experiment", "architecture", "scope", "mode", "threshold", "n", "n_normal", "n_defect",
           "tn", "fp", "fn", "tp", *METRICS, "undefined"]


def read_run(directory, experiment, architecture):
    cfg = read_json(directory / "config.json")
    run = read_json(directory / "run.json")
    test = directory / "test"
    evaluation = read_json(test / "evaluation.json")
    if cfg["training"]["model"] != architecture or run["experiment"] != experiment:
        raise ValueError(f"Expected {architecture} experiment {experiment} at {directory}")
    if any(evaluation[k] != run[k] for k in ("protocol_id", "weights_sha256", "operating_threshold")):
        raise ValueError(f"Evaluation does not match the saved training run: {directory}")
    for filename in ("metrics.csv", "predictions.csv"):
        if sha256(test / filename) != evaluation["files"][filename]:
            raise ValueError(f"Saved evaluation changed: {test / filename}")

    # The architectures have distinct protocol IDs. Compare the conditions that must be shared.
    conditions = dict(**preparation_settings(cfg), threshold_policy=cfg["threshold"],
                      train_sources=run["train_sources"], checkpoint_rule=run["checkpoint_rule"],
                      manifests=run["manifest_sha256"], splits=sha256(directory / "splits.csv"),
                      test_ids=evaluation["test_ids_sha256"])
    # Cache use and loader workers change I/O only. Accept auto/cuda when both actually used CUDA.
    conditions["training.device"] = run["device"]
    for key, value in cfg["training"].items():
        if key not in {"model", "device", "cache_inputs", "workers"}:
            conditions[f"training.{key}"] = value
    identity = ["image_id", "source", "board", "group_id", "label", "n_tiles"]
    originals = pd.read_csv(test / "predictions.csv", usecols=identity, dtype={"board": str})
    originals = originals.set_index("image_id", verify_integrity=True).sort_index()
    if len(originals) != evaluation["originals"]:
        raise ValueError(f"Unexpected number of test originals: {directory}")
    metrics = pd.read_csv(test / "metrics.csv")
    return conditions, originals, metrics.assign(experiment=experiment, architecture=architecture)[COLUMNS]


def paired_metrics(cfg):
    """Return all completed, comparable A/B/C pairs and explain missing or differing pairs."""
    roots = {"resnet18": Path(cfg["resnet_output_dir"]), "efficientnet_b0": Path(cfg["output_dir"])}
    frames, notes = [], []
    for experiment in EXPERIMENTS:
        paths = {model: root / "runs" / experiment for model, root in roots.items()}
        missing = [model for model, path in paths.items() if not (path / "test/evaluation.json").is_file()]
        if missing:
            notes.append(f"{experiment} · 저장된 test 결과가 필요합니다 ({', '.join(missing)}).")
            continue
        ref_conditions, ref_images, ref_metrics = read_run(paths["resnet18"], experiment, "resnet18")
        conditions, images, metrics = read_run(paths["efficientnet_b0"], experiment, "efficientnet_b0")
        differences = [k for k in sorted(conditions.keys() | ref_conditions.keys())
                       if conditions.get(k) != ref_conditions.get(k)]
        if not images.equals(ref_images):
            differences.append("test image IDs, labels, sources or tile counts")
        if differences:
            notes.append(f"{experiment} · 비교에서 제외한 조건 차이 ({', '.join(differences)}).")
            continue
        frames.extend([ref_metrics, metrics])
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)), notes


def save_comparison(cfg):
    root = Path(cfg["output_dir"])
    destination = root / "model_comparison.csv"
    with FileLock(str(destination) + ".lock"):
        table, notes = paired_metrics(cfg)
        with TemporaryDirectory(dir=root) as tmp:
            staged = Path(tmp) / destination.name
            table.to_csv(staged, index=False)
            staged.replace(destination)
    for note in notes:
        print(note)
    print(f"Saved {len(table)} comparison rows to {destination}")
    return table
