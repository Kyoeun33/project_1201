"""Evaluate a saved src2 ResNet18 on the Mendeley PCB-Defect positive-only cohort."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from time import perf_counter

from src2.common import CLASSES, read_json, save_json, sha256, write_rows
from src2.mendeley_data import audit_dataset


DEFAULT_CONFIG = Path(__file__).with_name("mendeley_external_config.json")
LIMITATION = (
    "The dataset contains only defect images. This run measures defect recall on readable "
    "images and cannot establish how well the model separates normal and defective images."
)
UNDEFINED = {
    "TN": "There are no ground-truth normal images.",
    "FP": "There are no ground-truth normal images.",
    "specificity": "There are no ground-truth normal images.",
    "normal_false_positive_rate": "There are no ground-truth normal images.",
    "roc_auc": "Both ground-truth classes are required.",
}


class MissingCheckpointSettings(ValueError):
    def __init__(self, keys):
        self.keys = keys
        super().__init__(
            f"Checkpoint metadata missing: {', '.join(keys)}. "
            "Supply the original training values in checkpoint_metadata_overrides or a "
            "validation threshold and provenance via --threshold and --threshold-source."
        )


def settings_from_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--annotation-format", choices=("obb", "coco"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-source", help="Original validation record for a manual threshold")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    settings = read_json(config_path)
    for key in ("image_dir", "annotations", "checkpoint", "output_dir"):
        value = getattr(args, key)
        settings[key] = value.resolve() if value is not None else (config_path.parent / settings[key]).resolve()
    for key in ("annotation_format", "threshold", "threshold_source", "device", "batch_size"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    settings["config_path"] = config_path
    settings["audit_only"] = args.audit_only
    if settings["annotation_format"] not in ("obb", "coco"):
        raise ValueError("annotation_format must be obb or coco")
    if settings["device"] not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu")
    return settings


def checkpoint_settings(saved, config):
    overrides = config.get("checkpoint_metadata_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("checkpoint_metadata_overrides must be an object with dotted keys")
    applied = {}
    missing = []

    def read_setting(name, optional=False):
        value = saved
        for key in name.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            return value
        if name in overrides and overrides[name] is not None:
            applied[name] = overrides[name]
            return applied[name]
        if not optional:
            missing.append(name)
        return None

    training = {
        key: read_setting(f"training.{key}")
        for key in ("model", "image_size", "preprocessing")
    }
    if training["preprocessing"] == "binary":
        training["binary_cutoff"] = read_setting("training.binary_cutoff")
    tiling = {key: read_setting(f"tiling.{key}") for key in ("size", "overlap")}
    pooling = read_setting("pooling")
    classes = read_setting("classes")
    calibration = saved.get("calibration") or {}
    threshold = config.get("threshold")
    provenance = config.get("threshold_source")
    if threshold is None:
        threshold = saved.get("threshold")
        if threshold is None:
            missing.append("threshold")
        if not str(calibration.get("selected_on", "")).startswith("validation"):
            missing.append("calibration.selected_on")
        calibrated_threshold = calibration.get("threshold")
        if calibrated_threshold is not None and calibrated_threshold != threshold:
            raise ValueError("Checkpoint threshold disagrees with its validation calibration")
        provenance = "checkpoint calibration (" + str(calibration.get("selected_on")) + ")"
    elif not isinstance(provenance, str) or not provenance.strip():
        raise ValueError("A manual threshold requires --threshold-source identifying the existing validation record")
    if missing:
        raise MissingCheckpointSettings(missing)

    if training["model"] != "efficientnet_b0" or classes != CLASSES or pooling != "max":
        raise ValueError("Expected src3 EfficientNet-B0, classes 0=normal/1=defect and max image pooling")
    if type(training["image_size"]) is not int or training["image_size"] <= 0:
        raise ValueError("Checkpoint training.image_size must be a positive integer")
    if training["preprocessing"] not in ("rgb", "binary"):
        raise ValueError("Unknown checkpoint preprocessing mode")
    if training["preprocessing"] == "binary" and (
        type(training["binary_cutoff"]) is not int or not 0 <= training["binary_cutoff"] <= 255
    ):
        raise ValueError("Checkpoint binary_cutoff must be an integer in [0, 255]")
    if any(type(value) is not int for value in tiling.values()) or not 0 <= tiling["overlap"] < tiling["size"]:
        raise ValueError("Require integer tile pixel counts with 0 <= overlap < size")
    if type(threshold) not in (float, int) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The validation threshold must be a finite number in [0, 1]")

    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = (saved.get("training") or {}).get("batch_size", 16)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Use a positive integer for --batch-size")
    training["batch_size"] = batch_size
    return training, tiling, float(threshold), provenance, applied


def load_model(config, summary):
    import torch

    from src2.model import device_for, network

    checkpoint = config["checkpoint"]
    summary["checkpoint"]["sha256"] = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training, tiling, threshold, provenance, applied = checkpoint_settings(saved, config)
    summary["checkpoint"].update(
        architecture=training["model"],
        experiment=saved.get("experiment"),
        best_epoch=saved.get("best_epoch"),
        protocol_id=saved.get("protocol_id"),
        metadata_overrides_applied=applied,
    )
    summary["preprocessing"] = {
        "decode": "Pillow convert RGB",
        "resize": "crop native tile then resize with Pillow bilinear",
        "input_size": [training["image_size"]] * 2,
        "tensor_layout": "NCHW",
        "scale": "uint8 / 255",
        "mode": training["preprocessing"],
        "binary_cutoff": training.get("binary_cutoff"),
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
        "augmentation": False,
        "implementation": "src2.inputs.rgb_tile and collate, src2.model.input_batch and batch_scores",
    }
    summary["tiling"] = {
        **tiling,
        "stride": tiling["size"] - tiling["overlap"],
        "edge_policy": "src2.tiles.windows grid: anchor final tile to each edge, clip dimensions smaller than tile size",
        "coordinates": "original pixels, half-open (x0, y0, x1, y1)",
    }
    summary["scoring"] = {
        "tile_score": "float32 softmax(logits)[1]",
        "image_score": "max of all tile scores",
        "threshold": threshold,
        "threshold_source": provenance,
        "validation_calibration": saved.get("calibration"),
        "positive_if": "score >= threshold",
    }
    device = device_for(config["device"])
    summary["device"].update(
        used=str(device), torch_version=str(torch.__version__), batch_size=training["batch_size"]
    )
    if device.type == "cuda":
        summary["device"]["name"] = torch.cuda.get_device_name(device)
    model = network(training["model"], pretrained=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model = model.to(device, memory_format=torch.channels_last).eval()
    summary["checkpoint"]["strict_state_dict_loaded"] = True
    return model, device, training, tiling, threshold


def predict_images(config, images, model, device, training, tiling, threshold):
    from PIL import Image

    from src2.inputs import collate, rgb_tile
    from src2.model import batch_scores
    from src2.tiles import windows

    image_rows = []
    tile_rows = []
    for relative, info in images.items():
        if info["status"] != "ok":
            continue
        path = config["image_dir"] / relative
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Image changed after audit: {path}")
        with Image.open(path) as raw:
            image = raw.convert("RGB")
        grid = windows(*image.size, "grid", tiling)
        image_tiles = []
        while True:
            boxes = list(islice(grid, training["batch_size"]))
            if not boxes:
                break
            tiles = [(rgb_tile(image, box, training["image_size"]), 0) for box in boxes]
            batch, _ = collate(tiles)
            scores = batch_scores(model, batch, device, training)
            if getattr(scores, "shape", None) != (len(boxes),):
                raise ValueError(f"Expected one defect probability per tile for {relative}")
            for box, score in zip(boxes, scores):
                score = float(score)
                if not 0 <= score <= 1:
                    raise ValueError(f"Invalid tile probability for {relative}: {score}")
                image_tiles.append({
                    "image_path": relative, "tile_number": len(image_tiles) + 1,
                    "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3],
                    "prob_defect": score, "prob_normal": 1 - score,
                    "prediction": CLASSES[str(int(score >= threshold))],
                })
        if not image_tiles:
            raise ValueError(f"No tiles produced for {relative}")
        best = max(image_tiles, key=lambda row: row["prob_defect"])
        score = best["prob_defect"]
        image_rows.append({
            "image_path": relative, "width": info["width"], "height": info["height"],
            "tile_count": len(image_tiles), "best_tile_number": best["tile_number"],
            "max_prob_defect": score, "threshold": threshold,
            "prediction": best["prediction"], "ground_truth": "defect",
            "annotation_linked": info["annotation_linked"],
            "defect_box_count": info["defect_box_count"],
            "original_classes": info["original_classes"],
        })
        tile_rows.extend(image_tiles)
    return image_rows, tile_rows


def evaluate(config, output):
    start = perf_counter()
    summary = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started",
        "config_path": str(config["config_path"]),
        "checkpoint": {"path": str(config["checkpoint"]), "strict_state_dict_loaded": False},
        "requested_threshold": config.get("threshold"),
        "requested_threshold_source": config.get("threshold_source"),
        "device": {"requested": config["device"]},
        "limitation": LIMITATION,
        "unavailable_metrics": UNDEFINED,
    }
    try:
        images, boxes, audit = audit_dataset(
            config["image_dir"], config["annotations"], config["annotation_format"],
            config.get("obb_classes"), output,
        )
        summary["audit"] = audit
        save_json(output / "audit.json", audit)
        observed = audit["observed"]
        print(f"Audit: {observed['image_files']} image files, "
              f"{observed['annotation_rows']} annotation rows, "
              f"{audit['readable_images']} readable images")
        differences = audit["difference_from_published_v2"]
        if any(value != 0 for value in differences.values()):
            print(f"Difference from published v2 counts: {differences}", file=sys.stderr)
        if audit["issue_counts"]:
            print(f"Audit issues: {audit['issue_counts']}. See audit_issues.csv.", file=sys.stderr)
        if config["audit_only"]:
            summary["status"] = "audit_only"
            return
        if not audit["readable_images"]:
            raise ValueError("No readable images. See audit_images.csv and audit_issues.csv.")

        model, device, training, tiling, threshold = load_model(config, summary)
        image_rows, tile_rows = predict_images(config, images, model, device, training, tiling, threshold)
        write_rows(output / "image_predictions.csv", image_rows)
        write_rows(output / "tile_predictions.csv", tile_rows)

        import matplotlib
        import numpy as np

        matplotlib.use("Agg")
        from src2.mendeley_figures import save_figures

        tp = sum(row["prediction"] == "defect" for row in image_rows)
        total = len(image_rows)
        scores = [row["max_prob_defect"] for row in image_rows]
        summary["metrics"] = {
            "total_image_files": len(images), "evaluated_defect_images": total,
            "unreadable_excluded": len(images) - total,
            "detected_defect_images": tp, "missed_defect_images": total - tp,
            "confusion_counts": {"TP": tp, "FN": total - tp},
            "defect_recall": tp / total, "defect_miss_rate": (total - tp) / total,
            "always_defect_baseline_recall": 1.0,
            "recall_gap_vs_baseline": tp / total - 1.0,
            "tile_count": len(tile_rows),
            "score_distribution": {
                str(key): float(value)
                for key, value in zip(
                    ("min", "p25", "median", "p75", "max"), np.quantile(scores, [0, .25, .5, .75, 1])
                )
            },
        }
        summary["figures"] = save_figures(output, config["image_dir"], image_rows, tile_rows, boxes, threshold)
        summary["status"] = "completed"
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, MissingCheckpointSettings):
            summary["missing_checkpoint_settings"] = error.keys
        raise
    finally:
        summary["runtime_seconds"] = round(perf_counter() - start, 3)
        save_json(output / "summary.json", summary)


def main():
    from filelock import FileLock

    try:
        config = settings_from_args()
        output = config["output_dir"]
        annotation_path = config["annotations"]
        protected = (
            config["image_dir"], annotation_path.parent if annotation_path.suffix.lower() in {".json", ".txt"}
            else annotation_path,
            config["checkpoint"].parent,
            Path(__file__).resolve().parent.parent / "out_data",
        )
        if any(output.is_relative_to(path) for path in protected):
            raise ValueError("Use an output directory outside the data folders and checkpoint directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(output) + ".lock", timeout=0):
            output.mkdir(exist_ok=True)
            if any(output.iterdir()):
                raise ValueError(f"Output directory already contains files. Choose a new --output-dir: {output}")
            evaluate(config, output)
        print(f"Saved artifacts to {output}")
    except Exception as error:
        print(f"Mendeley external evaluation: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

