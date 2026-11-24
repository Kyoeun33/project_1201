"""Shared inference, matching, AP, false-positive calibration and timing."""

from __future__ import annotations

import csv
import hashlib
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from candidates import box_iou, candidate_labels, find_candidates, patch_tensor
from dataset import CLASS_NAMES, load_pair

PREDICTION_FIELDS = ("sample_id", "x1", "y1", "x2", "y2", "class_id", "score")
MATCH_IOU = 0.33
SMALL_AREA = 32 * 32


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _class_nms(detections: list[dict], overlap: float) -> list[dict]:
    kept = []
    for cls in CLASS_NAMES:
        ordered = sorted((d for d in detections if d["class_id"] == cls),
                         key=lambda d: d["score"], reverse=True)
        while ordered:
            chosen = ordered.pop(0)
            kept.append(chosen)
            ordered = [p for p in ordered if box_iou(
                (p["x1"], p["y1"], p["x2"], p["y2"]),
                (chosen["x1"], chosen["y1"], chosen["x2"], chosen["y2"])) <= overlap]
    return sorted(kept, key=lambda d: d["score"], reverse=True)


@torch.inference_mode()
def predict_one(kind: str, model, config: dict, temp: np.ndarray, test: np.ndarray,
                sample_id: str, device: torch.device) -> list[dict]:
    predictions = []
    if kind == "fast":
        candidates, added, missing = find_candidates(temp, test, config["candidate"])
        patch_size = int(config["model"]["patch_size"])
        chunk_size = 128
        for start in range(0, len(candidates), chunk_size):
            group = candidates[start:start + chunk_size]
            patches = np.stack([patch_tensor(temp, test, added, missing, candidate.patch_box, patch_size)
                                for candidate in group])
            probabilities = model(torch.from_numpy(patches).to(device)).softmax(dim=1).cpu().numpy()
            for candidate, probs in zip(group, probabilities):
                cls = int(np.argmax(probs))
                if cls == 0:
                    continue
                x1, y1, x2, y2 = candidate.output_box  # never the larger classifier crop
                predictions.append(dict(sample_id=sample_id, x1=x1, y1=y1, x2=x2, y2=y2,
                                        class_id=cls, score=float(probs[cls])))
    elif kind == "accurate":
        pair = np.stack((test, temp)).copy()
        image = torch.from_numpy(pair).float().div_(255).to(device)
        output = model([image])[0]
        for box, cls, score in zip(output["boxes"].cpu().tolist(),
                                   output["labels"].cpu().tolist(), output["scores"].cpu().tolist()):
            if cls not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = box
            predictions.append(dict(sample_id=sample_id, x1=x1, y1=y1, x2=x2, y2=y2,
                                    class_id=int(cls), score=float(score)))
    else:
        raise ValueError(f"알 수 없는 모델: {kind}")
    return _class_nms(predictions, float(config["model"]["nms_iou"]))


def infer_records(kind: str, model, config: dict, device: torch.device,
                  records: list[dict[str, str]]) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    all_predictions = []
    truths = {}
    for number, record in enumerate(records, 1):
        temp, test, boxes, labels = load_pair(Path(config["data_root"]), record)
        sample_id = record["sample_id"]
        truths[sample_id] = (boxes, labels)
        all_predictions.extend(predict_one(kind, model, config, temp, test, sample_id, device))
        if number % 50 == 0:
            print(f"{kind}: {number}/{len(records)} 이미지 추론")
    return all_predictions, truths


def filter_predictions(predictions: list[dict], threshold: float) -> list[dict]:
    return [prediction for prediction in predictions if prediction["score"] >= threshold]


def write_predictions(path: Path, predictions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for item in predictions:
            writer.writerow({**item, **{key: f"{item[key]:.3f}" for key in ("x1", "y1", "x2", "y2")},
                             "score": f"{item['score']:.8f}"})


def operating_metrics(predictions: list[dict], truths: dict, *, threshold: float,
                      match_iou: float = MATCH_IOU) -> dict:
    by_sample = defaultdict(list)
    for prediction in filter_predictions(predictions, threshold):
        by_sample[prediction["sample_id"]].append(prediction)
    counts = defaultdict(int)
    true_positive_class = defaultdict(int)
    small_total = small_tp = 0
    for sample_id, (boxes, labels) in truths.items():
        matched = set()
        counts["gt"] += len(boxes)
        small_total += int(np.sum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) < SMALL_AREA))
        for cls in labels:
            counts[f"gt_{int(cls)}"] += 1
        for prediction in sorted(by_sample.get(sample_id, []), key=lambda d: d["score"], reverse=True):
            pred_box = [prediction[key] for key in ("x1", "y1", "x2", "y2")]
            best, best_iou = -1, match_iou
            for j, (box, cls) in enumerate(zip(boxes, labels)):
                if j in matched or int(cls) != prediction["class_id"]:
                    continue
                iou = box_iou(pred_box, box)
                if iou >= best_iou:
                    best, best_iou = j, iou
            if best < 0:
                counts["fp"] += 1
            else:
                matched.add(best)
                counts["tp"] += 1
                true_positive_class[prediction["class_id"]] += 1
                if (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1]) < SMALL_AREA:
                    small_tp += 1
    tp, fp, gt = counts["tp"], counts["fp"], counts["gt"]
    fn = gt - tp
    return {
        "match_iou": match_iou, "score_threshold": threshold,
        "images": len(truths), "tp": tp, "fp": fp, "fn": fn,
        "false_positives_per_image": fp / len(truths),
        # Object detection has no meaningful finite true-negative count. This
        # accuracy treats unmatched GT and predictions as errors (Jaccard score).
        "accuracy": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "accuracy_definition": "TP / (TP + FP + FN), object-level detection accuracy at the configured IoU",
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / gt if gt else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "class_recall": {CLASS_NAMES[c]: true_positive_class[c] / counts[f"gt_{c}"]
                         if counts[f"gt_{c}"] else None for c in CLASS_NAMES},
        "small_box_area_lt": SMALL_AREA,
        "small_box_recall": small_tp / small_total if small_total else None,
        "small_box_count": small_total,
    }


def evaluate_loss(kind: str, model, config: dict, device: torch.device,
                  records: list[dict]) -> float:
    """Compute a held-out loss using the same target construction as training."""
    total_loss = 0.0
    total_items = 0
    if kind == "fast":
        model.eval()
        weights = torch.tensor([0.5] + [1.0] * 6, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction="sum")
        for record in records:
            temp, test, boxes, classes = load_pair(Path(config["data_root"]), record)
            candidates, added, missing = find_candidates(temp, test, config["candidate"])
            labels = candidate_labels(candidates, boxes, classes)
            foreground = np.flatnonzero(labels > 0).tolist()
            background = np.flatnonzero(labels == 0).tolist()[:int(config["training"]["max_background_per_image"])]
            chosen = foreground + background
            if not chosen:
                continue
            patch_size = int(config["model"]["patch_size"])
            patches = np.stack([patch_tensor(temp, test, added, missing,
                                             candidates[i].patch_box, patch_size) for i in chosen])
            inputs = torch.from_numpy(patches).to(device)
            targets = torch.from_numpy(labels[chosen]).to(device)
            with torch.inference_mode():
                loss = criterion(model(inputs), targets)
            total_loss += float(loss)
            total_items += float(weights[targets].sum())
    elif kind == "accurate":
        prior_mode = model.training
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        try:
            with torch.no_grad():
                for record in records:
                    temp, test, boxes, labels = load_pair(Path(config["data_root"]), record)
                    pair = np.stack((test, temp)).copy()
                    image = torch.from_numpy(pair).float().div_(255).to(device)
                    target = {"boxes": torch.from_numpy(boxes.copy()).float().to(device),
                              "labels": torch.from_numpy(labels.copy()).long().to(device)}
                    losses = model([image], [target])
                    total_loss += float(sum(losses.values()))
                    total_items += 1
        finally:
            model.train(prior_mode)
    else:
        raise ValueError(f"unknown model kind: {kind}")
    if not total_items:
        raise ValueError("cannot calculate loss: no candidates or records")
    return total_loss / total_items


def calibrate_threshold(predictions: list[dict], truths: dict, max_fp_per_image: float) -> dict:
    if max_fp_per_image < 0:
        raise ValueError("오탐 상한은 음수가 될 수 없음")
    candidates = np.append(np.linspace(0.0, 1.0, 101), 1.000001)
    feasible = []
    for threshold in candidates:
        metrics = operating_metrics(predictions, truths, threshold=float(round(threshold, 2)))
        if metrics["false_positives_per_image"] <= max_fp_per_image + 1e-12:
            feasible.append(metrics)
    # The final threshold > 1 is feasible even if numerical saturation yields 1.0.
    if not feasible:
        raise RuntimeError("오탐 상한을 만족하는 임계값이 없음")
    return max(feasible, key=lambda m: (m["f1"], m["recall"], -m["false_positives_per_image"],
                                        -m["score_threshold"]))


def _class_ap(predictions: list[dict], truths: dict, cls: int, match_iou: float) -> float | None:
    gt = {sample_id: [box for box, label in zip(boxes, labels) if int(label) == cls]
          for sample_id, (boxes, labels) in truths.items()}
    total = sum(map(len, gt.values()))
    if total == 0:
        return None
    used = defaultdict(set)
    tp, fp = [], []
    for prediction in sorted((p for p in predictions if p["class_id"] == cls),
                             key=lambda p: p["score"], reverse=True):
        sample_id = prediction["sample_id"]
        pred_box = [prediction[key] for key in ("x1", "y1", "x2", "y2")]
        best, best_iou = -1, match_iou
        for j, box in enumerate(gt[sample_id]):
            if j not in used[sample_id]:
                iou = box_iou(pred_box, box)
                if iou >= best_iou:
                    best, best_iou = j, iou
        good = best >= 0
        if good:
            used[sample_id].add(best)
        tp.append(int(good))
        fp.append(int(not good))
    if not tp:
        return 0.0
    recall = np.cumsum(tp) / total
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1)
    return float(np.mean([np.max(precision[recall >= r], initial=0.0)
                          for r in np.linspace(0, 1, 101)]))


def mean_average_precision(predictions: list[dict], truths: dict) -> dict:
    thresholds = [0.33] + [round(0.5 + 0.05 * i, 2) for i in range(10)]
    by_iou = {}
    for iou in thresholds:
        by_class = {CLASS_NAMES[cls]: _class_ap(predictions, truths, cls, iou) for cls in CLASS_NAMES}
        present = [ap for ap in by_class.values() if ap is not None]
        by_iou[f"{iou:.2f}"] = {"mean": float(np.mean(present)) if present else None,
                                  "class_ap": by_class}
    return {
        "mAP_0_33": by_iou["0.33"]["mean"],
        "mAP_0_50_to_0_95": float(np.mean([by_iou[f"{iou:.2f}"]["mean"]
                                               for iou in thresholds[1:]
                                               if by_iou[f"{iou:.2f}"]["mean"] is not None])),
        "by_iou": by_iou,
        "ap_interpolation": "101-point recall interpolation; per-class, score-ranked; pre-operating-threshold",
    }


def benchmark(kind: str, model, config: dict, device: torch.device,
              records: list[dict], *, warmup: int = 3, sample_count: int = 20) -> dict:
    if sample_count < 1 or warmup < 0:
        raise ValueError("sample_count >= 1, warmup >= 0 필요")
    # Read every image first; include binarization, crops, transfer and postprocess.
    loaded = [(record["sample_id"], *load_pair(Path(config["data_root"]), record)[:2])
              for record in records[:sample_count]]
    if not loaded:
        raise ValueError("속도 측정용 이미지가 없음")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for i in range(warmup):
        sample_id, temp, test = loaded[i % len(loaded)]
        predict_one(kind, model, config, temp, test, sample_id, device)
    sync()
    durations = []
    for sample_id, temp, test in loaded:
        sync()
        start = time.perf_counter()
        predict_one(kind, model, config, temp, test, sample_id, device)
        sync()
        durations.append((time.perf_counter() - start) * 1000)
    return {"device": str(device), "warmup_pairs": warmup, "measured_pairs": len(durations),
            "disk_read_included": False, "median_ms": float(np.median(durations)),
            "p95_ms": float(np.percentile(durations, 95))}
