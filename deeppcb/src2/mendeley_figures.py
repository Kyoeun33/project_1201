"""Figures for saved external evaluation scores and annotation context."""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from PIL import Image


def score_figures(figure_dir, image_rows, threshold):
    scores = np.array([row["max_prob_defect"] for row in image_rows])
    detected = np.array([row["prediction"] == "defect" for row in image_rows])
    bins = np.linspace(0, 1, 26)

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.hist(scores, bins=bins, color="#31688e", edgecolor="white",
            label=f"All evaluated images (n={len(scores)})")
    ax.axvline(threshold, c="#242b33", ls="--", label=f"Validation threshold {threshold:.4f}")
    ax.set(xlim=(0, 1), xlabel="Maximum tile defect probability", ylabel="Images",
           title="Mendeley PCB-Defect image score distribution")
    ax.legend(loc="upper left")
    fig.savefig(figure_dir / "score_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, layout="constrained")
    labels = [
        (axes[0], scores[detected], "#257068", "Detected defects (TP)"),
        (axes[1], scores[~detected], "#b44b37", "Missed defects (FN)"),
    ]
    for ax, group, color, title in labels:
        if len(group):
            ax.hist(group, bins=bins, color=color, edgecolor="white")
        else:
            ax.text(.5, .5, "No images in this group", ha="center", va="center", transform=ax.transAxes)
        ax.axvline(threshold, c="#242b33", ls="--")
        ax.set(xlim=(0, 1), xlabel="Maximum tile defect probability", title=f"{title} · n={len(group)}")
    axes[0].set_ylabel("Images")
    fig.suptitle(f"Detection and missed detection · validation threshold {threshold:.4f}")
    fig.savefig(figure_dir / "detected_vs_missed.png", dpi=150)
    plt.close(fig)


def image_overlay(figure_path, image_path, result, tiles, ground_truth):
    width, height = result["width"], result["height"]
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
    image.thumbnail((1500, 1500))

    fig, (original, overlay) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    for ax in (original, overlay):
        ax.imshow(image, extent=(0, width, height, 0))
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis("off")

    rectangles = [
        Rectangle((row["x0"], row["y0"]), row["x1"] - row["x0"], row["y1"] - row["y0"])
        for row in tiles
    ]
    grid = PatchCollection(rectangles, cmap="viridis", norm=Normalize(0, 1),
                           edgecolor="white", linewidth=0.55, alpha=0.50)
    grid.set_array(np.array([row["prob_defect"] for row in tiles]))
    overlay.add_collection(grid)
    for _, points in ground_truth:
        if points is not None:
            overlay.add_patch(Polygon(points, closed=True, fill=False,
                                      edgecolor="#ffb44d", linewidth=1.3, zorder=3))

    best_index = result["best_tile_number"] - 1
    best = tiles[best_index]
    overlay.add_patch(Rectangle(
        (best["x0"], best["y0"]), best["x1"] - best["x0"], best["y1"] - best["y0"],
        fill=False, edgecolor="#00e0f0", linewidth=2, zorder=4,
    ))
    overlay.annotate(
        f"max {best['prob_defect']:.4f}", (best["x0"], best["y0"]), xytext=(4, -4),
        textcoords="offset points", va="top", color="white", fontsize=8,
        bbox={"facecolor": "#202634", "pad": 2, "alpha": .85},
    )
    handles = [Line2D([], [], color="#00e0f0", label="Highest-scoring tile")]
    if any(points is not None for _, points in ground_truth):
        handles.append(Line2D([], [], color="#ffb44d", label="Mendeley ground truth"))
    overlay.legend(handles=handles, loc="lower left", fontsize=8)
    fig.colorbar(grid, ax=overlay, shrink=.57, label="Tile defect probability")
    original.set_title("Original PCB image")
    overlay.set_title(f"Classification tile scores · {len(tiles)} tiles")
    fig.suptitle(f"{image_path.name} · prediction {result['prediction']} · ground truth defect")
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)


def save_figures(output, image_dir, image_rows, tile_rows, boxes, threshold):
    figure_dir = output / "figures"
    figure_dir.mkdir()
    score_figures(figure_dir, image_rows, threshold)
    figures = {
        "score_distribution": "figures/score_distribution.png",
        "detected_vs_missed": "figures/detected_vs_missed.png",
    }
    tiles_by_image = defaultdict(list)
    for row in tile_rows:
        tiles_by_image[row["image_path"]].append(row)

    for key, label in (("detected_example", "defect"), ("missed_example", "normal")):
        candidates = [row for row in image_rows if row["prediction"] == label]
        if not candidates:
            figures[key] = {"path": None, "reason": f"No images predicted {label}"}
            continue
        # Prefer an image with drawable ground truth, then choose the strongest group example.
        annotated = [row for row in candidates if any(
            points is not None for _, points in boxes[row["image_path"]]
        )]
        options = annotated or candidates
        select = max if label == "defect" else min
        example = select(options, key=lambda row: row["max_prob_defect"])
        relative = example["image_path"]
        path = figure_dir / f"{key}.png"
        image_overlay(path, image_dir / relative, example, tiles_by_image[relative], boxes[relative])
        figures[key] = {
            "path": path.relative_to(output).as_posix(),
            "image_path": relative,
            "max_prob_defect": example["max_prob_defect"],
        }
    return figures
