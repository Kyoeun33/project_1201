"""Verify originals, split intact groups, then label native-resolution tiles."""
from __future__ import annotations

import argparse
import random
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .common import (DEFAULT_CONFIG, SOURCES, SPLITS, config, load_prepared, preparation_settings,
                     save_json, sha256, write_rows)
from .preview import draw_previews
from .sources import IMAGE_SUFFIXES, scan
from .tiles import enumerate_tiles


def assign_splits(records, fractions, seed):
    strata = defaultdict(lambda: defaultdict(list))
    for item in records:
        key = (item.source, item.board, item.label) if item.source == "VisA" else (item.source,)
        strata[key][item.group_id].append(item)
    for key, groups in sorted(strata.items()):
        order = sorted(groups)
        random.Random(f"{seed}/{key}").shuffle(order)
        total = sum(map(len, groups.values()))
        counts = [0, 0, 0]
        for i, group in enumerate(order):
            empty = [j for j in range(3) if not counts[j]]
            # Reserve a whole group for each remaining split if there are enough groups.
            choices = empty if len(order) - i <= len(empty) else range(3)
            dest = max(choices, key=lambda j: fractions[j] * total - counts[j])
            for item in groups[group]:
                item.split = SPLITS[dest]
            counts[dest] += len(groups[group])


def audit(originals, tiles):
    tests = {}
    for key in ("image_id", "group_id", "sha256"):
        tests[f"{key}_in_multiple_splits"] = int((originals.groupby(key).split.nunique() > 1).sum())
    tests["same_bytes_conflicting_labels"] = int((originals.groupby("sha256").label.nunique() > 1).sum())
    tests["duplicate_original_id"] = int(originals.image_id.duplicated().sum())
    tests["duplicate_tile_id"] = int(tiles.tile_id.duplicated().sum())
    tests["repeated_tile_in_original"] = int(tiles.duplicated(["image_id", "x0", "y0", "x1", "y1"]).sum())
    parent_split = originals.set_index("image_id").split
    tests["tile_parent_split_mismatch"] = int((tiles.image_id.map(parent_split) != tiles.split).sum())
    tests["originals_without_tiles"] = len(set(originals.image_id) - set(tiles.image_id))
    if any(tests.values()):
        raise ValueError(f"Split or identity audit failed: {tests}")
    return dict(passed=True, violations=tests,
                repeated_image_files_within_split=int(originals.sha256.duplicated().sum()),
                augmentation="on-the-fly, training loader only")


def population(originals, tiles):
    rows = []
    sources = originals.set_index("image_id").source
    local_tiles = tiles.assign(source=tiles.image_id.map(sources))
    for source in SOURCES:
        if source not in set(originals.source):
            continue
        for split in SPLITS:
            items = originals[(originals.source == source) & (originals.split == split)]
            t = local_tiles[(local_tiles.source == source) & (local_tiles.split == split)]
            kept = set(t.loc[t.label >= 0, "image_id"])
            positive = set(t.loc[t.label == 1, "image_id"])
            rows.append(dict(source=source, split=split, groups=items.group_id.nunique(),
                             originals=len(items), original_normal=int((items.label == 0).sum()),
                             original_defect=int(items.label.sum()), all_tiles=len(t),
                             tile_normal=int((t.label == 0).sum()), tile_defect=int((t.label == 1).sum()),
                             tile_excluded=int((t.label == -1).sum()), kept_tiles=int((t.label >= 0).sum()),
                             originals_with_kept_tiles=len(kept),
                             defect_originals_without_positive_tile=int(items.label.sum()) - len(positive)))
    return rows


def omitted_files(cfg, originals):
    skipped = {}
    if "DeepPCB" in cfg["sources"]:
        root = Path(cfg["paths"]["DeepPCB"])
        all_paths = {p.relative_to(root).as_posix() for p in root.glob("group*/*/*.jpg")}
        skipped["DeepPCB_unlisted_images"] = sorted(all_paths - set(originals.loc[
            originals.source == "DeepPCB", "path"]))
    if "PKU" in cfg["sources"]:
        folder = Path(cfg["paths"]["PKU"]) / "rotation"
        skipped["PKU_rotation_images"] = sum(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.rglob("*"))
    return skipped


def prepare(cfg):
    root = Path(cfg["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "prepared"
    if dest.exists():
        _, _, _, report = load_prepared(cfg)
        print(f"Verified existing manifests in {dest}. Use a new output_dir to rescan changed data.")
        return report
    records = scan(cfg)
    assign_splits(records, cfg["split_fractions"], cfg["seed"])
    originals = pd.DataFrame([r.row() for r in records])
    tiles = pd.DataFrame(enumerate_tiles(records, cfg))
    checks = audit(originals, tiles)
    counts = population(originals, tiles)
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        stage = Path(tmp) / "prepared"
        stage.mkdir()
        originals.to_csv(stage / "originals.csv", index=False)
        tiles.to_csv(stage / "tiles.csv", index=False)
        write_rows(stage / "counts.csv", counts)
        examples = draw_previews(stage / "previews", originals, tiles, cfg)
        write_rows(stage / "preview_examples.csv", examples)
        report = dict(settings=preparation_settings(cfg), paths=cfg["paths"], counts=counts,
                      leakage=checks, omitted=omitted_files(cfg, originals),
                      group_counts=originals.groupby("source").group_id.nunique().to_dict(),
                      groups_missing_split=[r for r in counts if not r["groups"]],
                      manifest_sha256={name: sha256(stage / name) for name in ("originals.csv", "tiles.csv")})
        save_json(stage / "summary.json", report)
        stage.rename(dest)
    print(pd.DataFrame(counts).to_string(index=False))
    print(f"Label previews, manifests and leakage audit: {dest}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare(config(parser.parse_args().config))
