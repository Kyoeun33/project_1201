"""XOR-like binary difference candidates and the fast classifier's input crops."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class Candidate:
    output_box: tuple[int, int, int, int]
    patch_box: tuple[int, int, int, int]
    area: int


def box_iou(a: np.ndarray | tuple, b: np.ndarray | tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - intersection
    return float(intersection / union) if union else 0.0


def _pad(box: tuple[int, int, int, int], margin: int, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return max(0, x1 - margin), max(0, y1 - margin), min(width, x2 + margin), min(height, y2 + margin)


def find_candidates(temp: np.ndarray, test: np.ndarray, cfg: dict) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    if temp.shape != test.shape or temp.ndim != 2:
        raise ValueError("템플릿과 검사 영상은 크기가 같은 단일 채널 영상이어야 함")
    threshold = int(cfg["binary_threshold"])
    min_area = int(cfg["min_area"])
    output_margin = int(cfg["output_margin"])
    context_margin = int(cfg["context_margin"])
    if not 0 <= threshold <= 255 or min_area <= 0 or min(output_margin, context_margin) < 0:
        raise ValueError("후보 임계값·면적·여백 설정이 올바르지 않음")
    # JPEG intensities are thresholded only for candidate generation. Images retain
    # their full grayscale values in the four-channel classifier input.
    temp_white = temp >= threshold
    test_white = test >= threshold
    polarity = cfg["copper_polarity"]
    if polarity == "dark":
        added = temp_white & ~test_white
        missing = ~temp_white & test_white
    elif polarity == "light":
        added = ~temp_white & test_white
        missing = temp_white & ~test_white
    else:
        raise ValueError("copper_polarity는 dark 또는 light")
    changed = added | missing
    connected, count = ndimage.label(changed, structure=np.ones((3, 3), dtype=np.uint8))
    slices = ndimage.find_objects(connected)
    areas = np.bincount(connected.ravel(), minlength=count + 1)
    h, w = temp.shape
    candidates = []
    for component_id, region in enumerate(slices, 1):
        if region is None or areas[component_id] < min_area:
            continue
        ys, xs = region
        raw = (xs.start, ys.start, xs.stop, ys.stop)
        output = _pad(raw, output_margin, w, h)
        patch = _pad(output, context_margin, w, h)
        candidates.append(Candidate(output, patch, int(areas[component_id])))
    return candidates, added.astype(np.uint8), missing.astype(np.uint8)


def patch_tensor(temp: np.ndarray, test: np.ndarray, added: np.ndarray,
                 missing: np.ndarray, box: tuple[int, int, int, int],
                 patch_size: int) -> np.ndarray:
    """Return CHW float32 [test, template, addition, removal] with separate ROI."""
    x1, y1, x2, y2 = box
    if not (x1 < x2 and y1 < y2 and patch_size > 0):
        raise ValueError("빈 분류 패치 또는 잘못된 patch_size")
    channels = []
    for arr, nearest in ((test, False), (temp, False), (added, True), (missing, True)):
        cropped = Image.fromarray(arr[y1:y2, x1:x2])
        interpolation = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        resized = cropped.resize((patch_size, patch_size), interpolation)
        channels.append(np.asarray(resized, dtype=np.float32) / (1 if nearest else 255))
    return np.stack(channels)


def candidate_labels(candidates: list[Candidate], boxes: np.ndarray, classes: np.ndarray,
                     match_iou: float = 0.33) -> np.ndarray:
    labels = np.zeros(len(candidates), dtype=np.int64)
    for i, candidate in enumerate(candidates):
        if len(boxes):
            ious = [box_iou(candidate.output_box, gt) for gt in boxes]
            best = int(np.argmax(ious))
            if ious[best] > match_iou:
                labels[i] = int(classes[best])
    return labels
