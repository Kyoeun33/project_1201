"""Compare VisA EfficientNet-B0 with and without patch augmentation in 5-fold CV."""
from __future__ import annotations

import sys
import tempfile
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import ConcatDataset, WeightedRandomSampler

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src2.common import CLASSES, load_prepared, read_json, save_json
from src2.inputs import COORDS, TileDataset, input_cache, loader
from src2.metrics import choose_threshold, image_scores, log_loss, metrics, save_predictions
from src2.model import device_for, network, score_loader, seed_all
from src2.tiles import regions, tile_target, windows
from src3.settings import load_config

trainer = import_module("src2.02_train")
FOLDS = 5
CALIBRATION_FRACTION = 0.15
DEFECT_SAMPLE_WEIGHT = 2.0
CONDITIONS = ("baseline", "patch_oversample")
MEASURES = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "fpr", "fnr")
OUTPUT = Path(__file__).resolve().with_name("artifacts_visa_kfold")


def partitions(originals, fractions, seed):
    """Stratify by board and label, keeping identical image bytes together."""
    if originals.groupby("group_id")["sha256"].nunique().gt(1).any():
        raise ValueError("VisA fold assignment expects each original group to have one image digest.")

    rng = np.random.default_rng(seed)
    assignment = {}
    unique = originals.drop_duplicates("sha256")
    for stratum, rows in unique.groupby(["board", "label"], sort=True):
        if len(rows) < len(fractions):
            raise ValueError(f"Too few originals in {stratum} for {len(fractions)} partitions.")
        hashes = rng.permutation(sorted(rows["sha256"]))
        order = rng.permutation(len(fractions))
        targets = len(rows) * np.asarray(fractions)
        counts = np.zeros(len(fractions), dtype=int)
        for index, digest in enumerate(hashes):
            empty = order[counts[order] == 0]
            choices = empty if len(hashes) - index <= len(empty) else order
            part = max(choices, key=lambda i: targets[i] - counts[i])
            assignment[digest] = part
            counts[part] += 1
    return originals["sha256"].map(assignment)


def fold_assignments(originals, seed):
    pool = originals.loc[originals["split"].isin(["train", "validation"])].copy()
    if pool.empty:
        raise ValueError("No VisA train or validation originals available.")
    outer = partitions(pool, [1 / FOLDS] * FOLDS, seed)
    columns = ["image_id", "board", "label", "group_id", "sha256"]
    for fold in range(1, FOLDS + 1):
        evaluation = outer == fold - 1
        inner = partitions(pool.loc[~evaluation], [1 - CALIBRATION_FRACTION, CALIBRATION_FRACTION],
                           seed + fold)
        roster = pool[columns].assign(fold=fold, role="evaluation")
        roster.loc[inner.index, "role"] = inner.map({0: "train", 1: "calibration"})
        yield fold, roster


def extra_defect_patches(train_images, base_tiles, cfg, overlap):
    """Crop additional positive patches at a smaller stride in the training split."""
    tiling = {**cfg["tiling"], "overlap": overlap}
    existing = set(base_tiles[["image_id", *COORDS]].itertuples(index=False, name=None))
    rows = []
    for item in train_images.loc[train_images["label"] == 1].itertuples():
        defects = regions(item, cfg["paths"]["VisA"])
        for box in windows(item.width, item.height, item.source, tiling):
            key = (item.image_id, *box)
            if key in existing:
                continue
            label, _, _, _ = tile_target(box, defects, tiling["min_defect_coverage"])
            if label == 1:
                rows.append((len(rows), *key, label))
    if not rows:
        raise ValueError("No additional defect patches were found in this fold.")
    return pd.DataFrame(rows, columns=["tile_id", "image_id", *COORDS, "label"])


def fit_fold(directory, cfg, condition, roster, originals, tiles, cache, patch_overlap, device):
    tr = cfg["training"]
    fold = int(roster["fold"].iloc[0])
    seed_all(cfg["seed"])
    role_by_image = roster.set_index("image_id")["role"]
    image_roles = originals["image_id"].map(role_by_image)
    tile_roles = tiles["image_id"].map(role_by_image)
    train_images = originals.loc[image_roles == "train"]
    train_tiles = tiles.loc[(tile_roles == "train") & (tiles["label"] >= 0)]
    calibration_images = originals.loc[image_roles == "calibration"]
    calibration_tiles = tiles.loc[tile_roles == "calibration"]
    evaluation_images = originals.loc[image_roles == "evaluation"]
    evaluation_tiles = tiles.loc[tile_roles == "evaluation"]

    base_labels = train_tiles["label"].to_numpy()
    counts = np.bincount(base_labels, minlength=2)
    if not counts.all() or any(frame["label"].nunique() != 2
                               for frame in (calibration_images, evaluation_images)):
        raise ValueError(f"Both classes are required for training, calibration and evaluation in fold {fold}.")
    weights = torch.tensor(len(train_tiles) / (2 * counts), dtype=torch.float32, device=device)
    train_data = TileDataset(train_tiles, originals, cfg, cache)
    sampler = None
    n_extra = 0
    if condition == "patch_oversample":
        extra = extra_defect_patches(train_images, train_tiles, cfg, patch_overlap)
        n_extra = len(extra)
        train_data = ConcatDataset((train_data, TileDataset(extra, train_images, cfg, None)))
        labels = np.concatenate((base_labels, extra["label"].to_numpy()))
        sampling_weights = np.where(labels == 1, DEFECT_SAMPLE_WEIGHT, 1.0)
        sampler = WeightedRandomSampler(sampling_weights, len(train_tiles), replacement=True,
                                        generator=torch.Generator().manual_seed(cfg["seed"]))

    train_batches = loader(train_data, cfg, device, shuffle=sampler is None, sampler=sampler)
    calibration_batches = loader(TileDataset(calibration_tiles, originals, cfg, cache), cfg, device)
    model = network(tr["model"], tr["pretrained"]).to(device, memory_format=torch.channels_last)
    opt_args = dict(lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    optimizer = (torch.optim.AdamW(model.parameters(), **opt_args) if tr["optimizer"] == "adamw"
                 else torch.optim.SGD(model.parameters(), momentum=0.9, **opt_args))
    scaler = torch.amp.GradScaler("cuda", enabled=tr["amp"] and device.type == "cuda")
    print(f"Fold {fold}/{FOLDS} {condition} on {device} | base tiles {len(train_tiles):,} "
          f"| extra defect patches {n_extra:,}", flush=True)

    history = []
    best_loss = float("inf")
    best_epoch = 0
    usable = calibration_tiles["label"].to_numpy() >= 0
    for epoch in range(1, tr["epochs"] + 1):
        train_loss, _ = trainer.train_epoch(model, train_batches, device, tr, optimizer, scaler, weights)
        probs = score_loader(model, calibration_batches, device, tr)
        pred = image_scores(calibration_images, calibration_tiles, probs)
        image_loss = log_loss(pred["label"], pred["prob_defect"])
        tile_loss = log_loss(calibration_tiles.loc[usable, "label"], probs[usable]) if usable.any() else None
        scores = metrics(pred["label"], pred["prob_defect"], 0.5)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_image_loss=image_loss,
                            val_tile_loss=tile_loss, **{f"val_{k}": scores[k] for k in MEASURES}))
        print(f"Fold {fold} {condition} epoch {epoch}/{tr['epochs']} | train={train_loss:.4f} "
              f"| validation image loss={image_loss:.4f}", flush=True)
        if image_loss < best_loss:
            best_epoch, best_loss, best_predictions = epoch, image_loss, pred
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= tr["early_stopping_patience"]:
            print(f"Fold {fold} {condition} stopped at epoch {epoch} after no validation improvement.",
                  flush=True)
            break
    if not best_epoch:
        raise ValueError(f"No finite validation loss in fold {fold} ({condition}).")

    decision, curve = choose_threshold(best_predictions, cfg["threshold"])
    threshold = decision["threshold"]
    model.load_state_dict(best_state)
    evaluation_batches = loader(TileDataset(evaluation_tiles, originals, cfg, cache), cfg, device)
    probabilities = score_loader(model, evaluation_batches, device, tr)
    predictions = image_scores(evaluation_images, evaluation_tiles, probabilities)

    save_predictions(directory, predictions, threshold)
    save_json(directory / "threshold.json", decision)
    curve.to_csv(directory / "threshold_curve.csv", index=False)
    frame = pd.DataFrame(history)
    frame.to_csv(directory / "history.csv", index=False)
    trainer.save_curves(directory, frame)
    torch.save(dict(state_dict=best_state, training=tr, tiling=cfg["tiling"], classes=CLASSES,
                    pooling="max", threshold=threshold, calibration=decision,
                    fold=fold, condition=condition, best_epoch=best_epoch, seed=cfg["seed"]),
               directory / "best.pt")
    save_json(directory / "run.json", dict(
        fold=fold, condition=condition, seed=cfg["seed"], best_epoch=best_epoch, epochs_run=len(history),
        best_validation_image_log_loss=best_loss, train_originals=len(train_images),
        calibration_originals=len(calibration_images), evaluation_originals=len(evaluation_images),
        train_normal_tiles=int(counts[0]), train_defect_tiles=int(counts[1]), extra_defect_patches=n_extra,
        draws_per_epoch=len(train_tiles), threshold=threshold, calibration_targets_met=decision["targets_met"],
        calibration_recall=decision["validation"]["recall"], calibration_fpr=decision["validation"]["fpr"]))


def save_summary(output, folds):
    all_metrics, run_info = [], []
    for fold, roster in folds:
        expected = roster.loc[roster["role"] == "evaluation"].set_index("image_id")["label"].sort_index()
        for condition in CONDITIONS:
            directory = output / "folds" / f"fold_{fold:02d}" / condition
            predictions = pd.read_csv(directory / "predictions.csv")
            actual = predictions.set_index("image_id")["label"].sort_index()
            if not actual.equals(expected):
                raise ValueError(f"Saved evaluation differs from the assigned fold: {directory}")
            table = pd.read_csv(directory / "metrics.csv")
            scopes = table["scope"].eq("all") | table["scope"].str.startswith("VisA/")
            all_metrics.append(table.loc[scopes].assign(fold=fold, condition=condition))
            run_info.append(read_json(directory / "run.json"))
    metrics_by_fold = pd.concat(all_metrics, ignore_index=True)
    long = metrics_by_fold.melt(id_vars=["fold", "scope", "mode", "condition"], value_vars=list(MEASURES),
                                var_name="metric", value_name="score")
    keys = ["scope", "mode", "metric"]
    summary = (long.groupby([*keys, "condition"])["score"]
               .agg(mean="mean", std="std", folds="count").reset_index())
    paired = long.pivot(index=[*keys, "fold"], columns="condition", values="score")
    delta = paired["patch_oversample"] - paired["baseline"]
    deltas = delta.groupby(level=keys).agg(mean="mean", std="std", folds="count").reset_index()

    # The summary is published last so the notebook sees complete companion tables.
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        tables = {"run_details.csv": pd.DataFrame(run_info), "fold_metrics.csv": metrics_by_fold,
                  "paired_deltas.csv": deltas, "summary.csv": summary}
        for name, table in tables.items():
            stage = Path(tmp) / name
            table.to_csv(stage, index=False)
            stage.replace(output / name)


def main():
    cfg = load_config()
    cfg["sources"] = ["VisA"]
    cfg["paths"] = {"VisA": cfg["paths"]["VisA"]}
    cfg["output_dir"] = str(OUTPUT)
    cfg["training"].update(epochs=25, early_stopping_patience=7, learning_rate=0.0002,
                           augment=False, imbalance="class_weight")
    patch_overlap = (cfg["tiling"]["size"] + cfg["tiling"]["overlap"]) // 2
    if patch_overlap <= cfg["tiling"]["overlap"]:
        raise ValueError("Base tile stride must be at least 2 pixels to build a denser patch grid.")

    OUTPUT.mkdir(exist_ok=True)
    with FileLock(str(OUTPUT / "cv.lock")):
        import_module("src2.01_prepare").prepare(cfg)
        _, originals, tiles, report = load_prepared(cfg)
        experiment = dict(config=cfg, manifest_sha256=report["manifest_sha256"], folds=FOLDS,
                          conditions=list(CONDITIONS), calibration_fraction=CALIBRATION_FRACTION,
                          patch_overlap=patch_overlap, defect_sample_weight=DEFECT_SAMPLE_WEIGHT,
                          excluded_test_originals=int((originals["split"] == "test").sum()))
        record = OUTPUT / "experiment.json"
        if record.exists():
            if read_json(record) != experiment:
                raise ValueError(f"Different cross-validation settings are already saved in {OUTPUT}.")
        else:
            with tempfile.TemporaryDirectory(dir=OUTPUT) as tmp:
                staged = Path(tmp) / record.name
                save_json(staged, experiment)
                staged.replace(record)

        folds = list(fold_assignments(originals, cfg["seed"]))
        pd.concat([roster for _, roster in folds], ignore_index=True).to_csv(
            OUTPUT / "fold_assignments.csv", index=False)
        cache = input_cache(cfg, originals, tiles, report)
        device = device_for(cfg["training"]["device"])
        for fold, roster in folds:
            fold_dir = OUTPUT / "folds" / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_cfg = {**cfg, "seed": cfg["seed"] + fold}
            for condition in CONDITIONS:
                dest = fold_dir / condition
                if dest.is_dir():
                    print(f"Reusing completed fold: {dest}", flush=True)
                    continue
                with tempfile.TemporaryDirectory(dir=fold_dir) as tmp:
                    stage = Path(tmp) / condition
                    stage.mkdir()
                    fit_fold(stage, fold_cfg, condition, roster, originals, tiles, cache, patch_overlap, device)
                    stage.rename(dest)
        save_summary(OUTPUT, folds)
    print(f"Cross-validation results: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
