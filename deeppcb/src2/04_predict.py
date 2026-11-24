"""Classify one original PCB image using a trained checkpoint."""
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image

from .common import SOURCES
from .inputs import collate, rgb_tile
from .model import batch_scores, device_for, restore
from .tiles import windows


def predict(checkpoint, path, device="auto", source="auto"):
    device = device_for(device)
    model, saved = restore(checkpoint, device)
    settings = saved["training"]
    with Image.open(path) as raw:
        image = raw.convert("RGB")
    if source == "auto":
        source = "DeepPCB" if image.size == (640, 640) else "grid"
    if source == "DeepPCB" and image.size != (640, 640):
        raise ValueError("DeepPCB whole-image mode expects a 640 x 640 input.")
    grid = windows(*image.size, source, saved["tiling"])
    maximum, count = 0.0, 0
    while boxes := list(islice(grid, settings["batch_size"])):
        patches = [(rgb_tile(image, box, settings["image_size"]), 0) for box in boxes]
        batch, _ = collate(patches)
        probabilities = batch_scores(model, batch, device, settings)
        maximum = max(maximum, float(np.max(probabilities)))
        count += len(boxes)
    threshold = saved["threshold"]
    label = int(maximum >= threshold)
    return dict(label=label, class_name=saved["classes"][str(label)], prob_normal=1 - maximum,
                prob_defect=maximum, threshold=threshold, tiles=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["auto", *SOURCES], default="auto")
    args = parser.parse_args()
    print(json.dumps(predict(args.checkpoint, args.image, args.device, args.source), indent=2))
