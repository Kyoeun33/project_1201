"""Shared training data, reproducibility and checkpoint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from candidates import candidate_labels, find_candidates, patch_tensor
from dataset import augment_pair, load_pair, read_split_csv
from models import FastPatchCNN, make_accurate_model


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("data_root", "split_csv", "runs_dir", "model", "training"):
        if key not in config:
            raise ValueError(f"설정 누락: {key}")
    project_root = path.parents[1]
    for key in ("data_root", "split_csv", "runs_dir"):
        value = Path(config[key])
        config[key] = value if value.is_absolute() else project_root / value
    return config


def select_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cuda", "cpu") or (name == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"사용할 수 없는 장치: {name}")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def require_full_candidate_check(config: dict) -> None:
    report_path = Path(config["runs_dir"]) / "candidate_report.json"
    if not report_path.exists():
        raise FileNotFoundError("학습 전에 python src/02_check_candidates.py를 실행하세요")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    split_path = Path(config["split_csv"])
    expected = hashlib.sha256(split_path.read_bytes()).hexdigest()
    count = len(read_split_csv(split_path, "val"))
    if (report["split_sha256"] != expected or report["candidate_config"] != config["candidate"]
            or report["evaluated_images"] != count):
        raise ValueError("후보 실험 이후 설정이나 분할이 변경되었거나 val 일부만 점검함; 02를 다시 실행하세요")


class PairDataset(Dataset):
    def __init__(self, config: dict, split: str, *, augment: bool = False, limit: int = 0):
        self.records = read_split_csv(Path(config["split_csv"]), split)
        if limit:
            self.records = self.records[:limit]
        if not self.records:
            raise ValueError(f"{split} 이미지가 없음")
        self.root = Path(config["data_root"])
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def get_arrays(self, index: int):
        record = self.records[index]
        temp, test, boxes, labels = load_pair(self.root, record)
        if self.augment:
            temp, test, boxes = augment_pair(temp, test, boxes, random)
        return temp, test, boxes, labels

    def __getitem__(self, index: int):
        temp, test, boxes, labels = self.get_arrays(index)
        pair = np.stack((test, temp)).copy()
        image = torch.from_numpy(pair).float() / 255.0
        target = {"boxes": torch.from_numpy(boxes.copy()).float(),
                  "labels": torch.from_numpy(labels.copy()).long()}
        return image, target


class FastCandidateDataset(PairDataset):
    def __init__(self, config: dict, split: str, *, augment: bool = False, limit: int = 0):
        super().__init__(config, split, augment=augment, limit=limit)
        self.candidate_cfg = config["candidate"]
        self.patch_size = int(config["model"]["patch_size"])
        self.max_background = int(config["training"]["max_background_per_image"])
        self.sample_background = split == "train"

    def __getitem__(self, index: int):
        temp, test, boxes, classes = self.get_arrays(index)
        candidates, added, missing = find_candidates(temp, test, self.candidate_cfg)
        labels = candidate_labels(candidates, boxes, classes)
        foreground = np.flatnonzero(labels > 0).tolist()
        background = np.flatnonzero(labels == 0).tolist()
        if self.sample_background:
            background = random.sample(background, min(self.max_background, len(background)))
        else:
            background = background[:self.max_background]
        chosen = foreground + background
        patches = [patch_tensor(temp, test, added, missing, candidates[i].patch_box,
                                self.patch_size) for i in chosen]
        if patches:
            return torch.from_numpy(np.stack(patches)), torch.from_numpy(labels[chosen])
        return torch.empty((0, 4, self.patch_size, self.patch_size)), torch.empty(0, dtype=torch.long)


def collate_fast(batch):
    return torch.cat([item[0] for item in batch]), torch.cat([item[1] for item in batch])


def collate_pairs(batch):
    return tuple(zip(*batch))


def save_checkpoint(path: Path, model: torch.nn.Module, config: dict, kind: str,
                    epoch: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config, "kind": kind,
                "epoch": epoch, "validation_loss": loss,
                "split_sha256": hashlib.sha256(Path(config["split_csv"]).read_bytes()).hexdigest()}, path)


def add_training_history(path: Path, train_losses: list[float], val_losses: list[float],
                         stopped_early: bool) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["training_history"] = {
        "train_loss": train_losses,
        "validation_loss": val_losses,
        "stopped_early": stopped_early,
        "early_stopping_patience": int(checkpoint["config"]["training"].get(
            "early_stopping_patience", 7)),
    }
    checkpoint["overfitting"] = assess_overfitting(train_losses, val_losses)
    torch.save(checkpoint, path)


def assess_overfitting(train_losses: list[float], val_losses: list[float],
                       consecutive_epochs: int = 3) -> dict:
    """Flag sustained falling train loss alongside rising validation loss."""
    streak = 0
    start_epoch = None
    for i in range(1, min(len(train_losses), len(val_losses))):
        if train_losses[i] < train_losses[i - 1] and val_losses[i] > val_losses[i - 1]:
            streak += 1
            if streak == 1:
                start_epoch = i + 1
            if streak >= consecutive_epochs:
                return {"detected": True, "start_epoch": start_epoch,
                        "rule": f"train loss fell while validation loss rose for {consecutive_epochs} consecutive epochs"}
        else:
            streak = 0
            start_epoch = None
    return {"detected": False, "start_epoch": None,
            "rule": f"train loss fell while validation loss rose for {consecutive_epochs} consecutive epochs"}


def load_checkpoint(path: Path, expected_kind: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["kind"] != expected_kind:
        raise ValueError(f"체크포인트 종류 오류: {path}")
    config = checkpoint["config"]
    actual = hashlib.sha256(Path(config["split_csv"]).read_bytes()).hexdigest()
    if checkpoint["split_sha256"] != actual:
        raise ValueError("체크포인트 학습 분할과 현재 분할 CSV가 다름")
    model = FastPatchCNN() if expected_kind == "fast" else make_accurate_model(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, config
