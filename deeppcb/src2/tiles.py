"""Native-resolution windows and per-defect coverage; coordinates are half-open."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def starts(length, size, overlap):
    last = max(0, length - size)
    values = list(range(0, last + 1, size - overlap))
    if values[-1] != last:
        values.append(last)
    return values


def windows(width, height, source, tiling):
    if source == "DeepPCB":
        yield (0, 0, width, height)
        return
    size, overlap = tiling["size"], tiling["overlap"]
    for y in starts(height, size, overlap):
        for x in starts(width, size, overlap):
            yield (x, y, min(x + size, width), min(y + size, height))


def read_mask(path, size):
    with Image.open(path) as im:
        if im.size != size:
            raise ValueError(f"Mask size {im.size} differs from image {size}: {path}")
        raw = np.asarray(im)
        mask = np.any(raw[:, :, :3] != 0, axis=2) if raw.ndim == 3 else raw != 0
    if not mask.any():
        raise ValueError(f"Empty mask for image marked Anomaly: {path}")
    return mask


@dataclass
class Region:
    bounds: tuple
    area: int
    pixels: np.ndarray | None = None


def regions(item, root):
    if not item.label:
        return []
    if item.source != "VisA":
        return [Region((x, y, r, b), (r - x) * (b - y)) for x, y, r, b in item.boxes]
    mask = read_mask(Path(root) / item.annotation, (item.width, item.height))
    ids, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    components = []
    for index, (ys, xs) in enumerate(ndimage.find_objects(ids), start=1):
        pixels = ids[ys, xs] == index
        components.append(Region((xs.start, ys.start, xs.stop, ys.stop), int(pixels.sum()), pixels))
    return components


def tile_target(window, defects, minimum):
    """Store the intersection and denominator of the best-covered individual defect."""
    x0, y0, x1, y1 = window
    inside = area = 0
    for defect in defects:
        left, top, right, bottom = defect.bounds
        x, y, r, b = max(x0, left), max(y0, top), min(x1, right), min(y1, bottom)
        if r <= x or b <= y:
            continue
        count = (r - x) * (b - y)
        if defect.pixels is not None:
            count = int(defect.pixels[y-top:b-top, x-left:r-left].sum())
        if count and (not area or count * area > inside * defect.area):
            inside, area = count, defect.area
    coverage = inside / area if area else 0.0
    label = 0 if inside == 0 else (1 if coverage >= minimum else -1)
    return label, coverage, inside, area


def enumerate_tiles(originals, cfg):
    output = []
    for item in originals:
        defects = regions(item, cfg["paths"][item.source])
        for box in windows(item.width, item.height, item.source, cfg["tiling"]):
            label, fraction, inside, area = tile_target(box, defects, cfg["tiling"]["min_defect_coverage"])
            output.append(dict(tile_id=len(output), image_id=item.image_id, split=item.split,
                               x0=box[0], y0=box[1], x1=box[2], y1=box[3], label=label,
                               overlap_fraction=fraction, overlap_px=inside, region_px=area))
    return output
