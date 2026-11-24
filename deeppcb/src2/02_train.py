"""Fit A/B/C with a shared original-image validation set and no test inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import WeightedRandomSampler

from .common import (CLASSES, DEFAULT_CONFIG, EXPERIMENTS, config, load_prepared,
                     read_json, save_json, sha256)
from .inputs import TileDataset, input_cache, loader
from .metrics import choose_threshold, image_scores, log_loss, metrics, save_predictions
from .model import device_for, input_batch, network, score_loader, seed_all


def protocol_id(cfg, report):
    conditions = dict(manifests=report["manifest_sha256"], settings=report["settings"],
                      training=cfg["training"], threshold=cfg["threshold"],
                      pooling="maximum", checkpoint="min validation image log loss")
    return hashlib.sha256(json.dumps(conditions, sort_keys=True).encode()).hexdigest()


def train_epoch(model, batches, device, settings, optimizer, scaler, weights):
    model.train()
    loss_total = weight_total = 0.0
    targets, probabilities = [], []
    for images, labels in batches:
        images = input_batch(images, device, settings, augment=settings["augment"])
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            logits = model(images)
            summed = F.cross_entropy(logits.float(), labels, weight=weights, reduction="sum")
            denominator = weights[labels].sum() if weights is not None else len(labels)
            loss = summed / denominator
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total += summed.detach().item()
        weight_total += float(denominator)
        targets.append(labels.cpu().numpy())
        probabilities.append(logits.detach().float().softmax(1)[:, 1].cpu().numpy())
    return loss_total / weight_total, metrics(np.concatenate(targets), np.concatenate(probabilities), 0.5)


def save_curves(directory, frame):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 3.4))
    for field, label in [("train_loss", "Train tiles (objective)"), ("val_tile_loss", "Validation tiles"),
                         ("val_image_loss", "Validation originals")]:
        ax.plot(frame.epoch, frame[field], marker="o", label=label)
    ax.set(title="Cross-entropy by epoch", xlabel="Epoch", ylabel="Loss")
    for field in ("precision", "recall", "f1", "fpr"):
        bx.plot(frame.epoch, frame["val_" + field], marker="o", label=field.upper())
    bx.set(title="Validation originals at 0.5", xlabel="Epoch", ylabel="Fraction", ylim=(-0.03, 1.03))
    for panel in (ax, bx):
        panel.set_xticks(frame.epoch if len(frame) <= 12 else frame.epoch.iloc[::len(frame)//8])
        panel.legend(fontsize=8)
        panel.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "learning_curves.png", dpi=130)
    plt.close(fig)


def fit(directory, experiment, cfg, originals, tiles, report, cache):
    tr = cfg["training"]
    device = device_for(tr["device"])
    seed_all(cfg["seed"])
    train_images = originals[(originals.split == "train") & originals.source.isin(EXPERIMENTS[experiment])]
    train_tiles = tiles[(tiles.split == "train") & tiles.image_id.isin(train_images.image_id) & (tiles.label >= 0)]
    val_images = originals[originals.split == "validation"]
    val_tiles = tiles[tiles.split == "validation"]
    counts = np.bincount(train_tiles.label.to_numpy(), minlength=2)
    if not counts.all() or val_images.empty:
        raise ValueError(f"Need both training classes and validation images. Training counts: {counts}")
    weights = None
    sampler = None
    balanced = len(train_tiles) / (2 * counts)
    if tr["imbalance"] == "class_weight":
        weights = torch.tensor(balanced, dtype=torch.float32, device=device)
    elif tr["imbalance"] == "weighted_sampler":
        sampler = WeightedRandomSampler(balanced[train_tiles.label.to_numpy()], len(train_tiles),
                                        replacement=True, generator=torch.Generator().manual_seed(cfg["seed"]))
    train_batches = loader(TileDataset(train_tiles, originals, cfg, cache), cfg, device,
                           shuffle=sampler is None, sampler=sampler)
    val_batches = loader(TileDataset(val_tiles, originals, cfg, cache), cfg, device)
    model = network(tr["model"], tr["pretrained"]).to(device, memory_format=torch.channels_last)
    opt_args = dict(lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    optimizer = (torch.optim.AdamW(model.parameters(), **opt_args) if tr["optimizer"] == "adamw"
                 else torch.optim.SGD(model.parameters(), momentum=0.9, **opt_args))
    scaler = torch.amp.GradScaler("cuda", enabled=tr["amp"] and device.type == "cuda")
    print(f"{experiment} on {device}: train tiles normal={counts[0]}, defect={counts[1]}; "
          f"validation originals={len(val_images)}", flush=True)
    history = []
    best_loss = float("inf")
    usable_val = val_tiles.label.to_numpy() >= 0
    for epoch in range(1, tr["epochs"] + 1):
        loss, training = train_epoch(model, train_batches, device, tr, optimizer, scaler, weights)
        probabilities = score_loader(model, val_batches, device, tr)
        pred = image_scores(val_images, val_tiles, probabilities)
        image_loss = log_loss(pred.label, pred.prob_defect)
        val_tile_loss = (log_loss(val_tiles.label[usable_val], probabilities[usable_val])
                         if usable_val.any() else None)
        validation = metrics(pred.label, pred.prob_defect, 0.5)
        row = dict(epoch=epoch, train_loss=loss, val_tile_loss=val_tile_loss, val_image_loss=image_loss)
        for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "fpr", "fnr"):
            row["train_" + key] = training[key]
            row["val_" + key] = validation[key]
        history.append(row)
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(f"{experiment} epoch {epoch}/{tr['epochs']} train_loss={loss:.4f} "
              f"val_image_loss={image_loss:.4f} val_recall={validation['recall']}", flush=True)
        if image_loss < best_loss:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss, best_epoch, best_predictions = image_loss, epoch, pred
    decision, curve = choose_threshold(best_predictions, cfg["threshold"])
    validation_dir = directory / "validation"
    validation_dir.mkdir()
    save_predictions(validation_dir, best_predictions, decision["threshold"])
    curve.to_csv(validation_dir / "threshold_curve.csv", index=False)
    save_json(directory / "threshold.json", decision)
    save_json(directory / "classes.json", CLASSES)
    save_json(directory / "config.json", cfg)
    split_list = originals[["image_id", "source", "board", "group_id", "label", "split"]].copy()
    split_list["in_training"] = split_list.image_id.isin(train_images.image_id)
    split_list.to_csv(directory / "splits.csv", index=False)
    save_curves(directory, pd.DataFrame(history))
    identity = protocol_id(cfg, report)
    torch.save(dict(state_dict=best_state, training=tr, tiling=cfg["tiling"], classes=CLASSES,
                    pooling="max", threshold=decision["threshold"], calibration=decision,
                    best_epoch=best_epoch, experiment=experiment, protocol_id=identity), directory / "best.pt")
    save_json(directory / "run.json", dict(experiment=experiment, protocol_id=identity,
              manifest_sha256=report["manifest_sha256"], weights_sha256=sha256(directory / "best.pt"),
              seed=cfg["seed"], train_sources=list(EXPERIMENTS[experiment]),
              validation_sources=sorted(val_images.source.unique()), validation_originals=len(val_images),
              train_tiles={"normal": int(counts[0]), "defect": int(counts[1])},
              imbalance=tr["imbalance"], inverse_frequency_weights=balanced.tolist(),
              best_epoch=best_epoch, best_validation_image_log_loss=best_loss,
              checkpoint_rule="minimum validation original-image log loss; first epoch on a tie",
              operating_threshold=decision["threshold"], device=str(device), torch_version=str(torch.__version__)))


def train(cfg, experiments):
    _, originals, tiles, report = load_prepared(cfg)
    parent = Path(cfg["output_dir"]) / "runs"
    parent.mkdir(exist_ok=True)
    missing = {s for exp in experiments for s in EXPERIMENTS[exp]} - set(originals.source)
    if missing:
        raise ValueError(f"Requested experiments need these missing sources: {sorted(missing)}")
    cache = input_cache(cfg, originals, tiles, report)
    for exp in experiments:
        dest = parent / exp
        with FileLock(str(dest) + ".lock"):
            if dest.exists():
                info = read_json(dest / "run.json")
                if info["protocol_id"] != protocol_id(cfg, report) or sha256(dest / "best.pt") != info["weights_sha256"]:
                    raise ValueError(f"Different run or changed weights at {dest}. Choose a new output_dir.")
                print(f"Already trained: {dest}")
                continue
            with tempfile.TemporaryDirectory(dir=parent) as tmp:
                directory = Path(tmp) / exp
                directory.mkdir()
                fit(directory, exp, cfg, originals, tiles, report, cache)
                directory.rename(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    args = parser.parse_args()
    train(config(args.config), args.experiments)
