"""Visual label audit on training originals only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .tiles import read_mask


def annotated(image, row, root):
    overlay = image.copy()
    pen = ImageDraw.Draw(overlay)
    stroke = max(3, image.width // 350)
    for x, y, r, b in json.loads(row.boxes):
        pen.rectangle((x, y, r - 1, b - 1), outline="#e00056", width=stroke)
    if row.source == "VisA" and row.label:
        mask = read_mask(root / row.annotation, image.size)
        alpha = Image.fromarray(mask.astype("uint8") * 165)
        overlay.paste(Image.new("RGB", image.size, "#ff0066"), mask=alpha)
    return overlay


def draw_previews(directory, originals, tiles, cfg):
    directory.mkdir()
    by_id = originals.set_index("image_id")
    joined = tiles[tiles.split == "train"].merge(
        originals[["image_id", "source", "label"]].rename(columns={"label": "image_label"}), on="image_id")
    examples = []
    for source, rows in joined.groupby("source", sort=False):
        choices = []
        for label, name, image_label in [(0, "normal", 0), (1, "defect", 1), (-1, "excluded", 1)]:
            subset = rows[(rows.label == label) & (rows.image_label == image_label)]
            if not subset.empty:
                choices.append((name, subset.iloc[0]))
        if not choices:
            continue
        fig, axes = plt.subplots(len(choices), 3, squeeze=False, figsize=(10, 3 * len(choices)))
        for (name, tile), axes_row in zip(choices, axes):
            rec = by_id.loc[tile.image_id]
            root = Path(cfg["paths"][source])
            with Image.open(root / rec.path) as raw:
                image = raw.convert("RGB")
            overlay = annotated(image, rec, root)
            box = tuple(int(tile[k]) for k in ("x0", "y0", "x1", "y1"))
            crop = overlay.crop(box)
            ImageDraw.Draw(overlay).rectangle((box[0], box[1], box[2]-1, box[3]-1),
                                               outline="#00b6d9", width=max(4, image.width // 200))
            views = ((overlay, f"{source} / {rec.board}  |  tile window"),
                     (image.crop(box), f"Input tile  |  {name}"),
                     (crop, f"GT  |  coverage {tile.overlap_fraction:.3f}"))
            for ax, (view, title) in zip(axes_row, views):
                view.thumbnail((640, 640))
                ax.imshow(view)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            examples.append(dict(source=source, example=name, image_id=tile.image_id,
                                 tile_id=int(tile.tile_id), overlap_px=int(tile.overlap_px),
                                 region_px=int(tile.region_px), overlap_fraction=float(tile.overlap_fraction)))
        fig.tight_layout()
        fig.savefig(directory / f"{source}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
    return examples
