"""Exactly the same crop-then-resize path for training, evaluation and deployment."""
from __future__ import annotations

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .common import DEFAULT_CONFIG, config, load_prepared, sha256

COORDS = ["x0", "y0", "x1", "y1"]


def rgb_tile(image, box, size):
    patch = image.crop(box).resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(patch, dtype=np.uint8)


def input_cache(cfg, originals, tiles, report):
    if not cfg["training"]["cache_inputs"]:
        return None
    size = cfg["training"]["image_size"]
    signature = report["manifest_sha256"]
    folder = Path(tempfile.gettempdir()) / "deeppcb-src2-rgb-v1"
    folder.mkdir(exist_ok=True)
    path = folder / f"{signature['originals.csv'][:16]}_{signature['tiles.csv'][:16]}_{size}.npy"
    with FileLock(str(path) + ".lock"):
        shape = (len(tiles), size, size, 3)
        if path.is_file():
            existing = np.load(path, mmap_mode="r")
            if existing.shape != shape or existing.dtype != np.uint8:
                raise ValueError(f"Tile cache has wrong shape/dtype: {path}")
            return path
        print(f"Caching {len(tiles):,} cropped RGB tiles ({np.prod(shape) / 2**30:.1f} GiB) in {folder}",
              flush=True)
        paths = {r.image_id: (Path(cfg["paths"][r.source]) / r.path, r.sha256)
                 for r in originals.itertuples()}
        geometry = tiles[["tile_id", *COORDS]].to_numpy(dtype=np.int64)
        with tempfile.TemporaryDirectory(dir=folder) as temp:
            stage = Path(temp) / "inputs.npy"
            cache = np.lib.format.open_memmap(stage, mode="w+", dtype=np.uint8, shape=shape)

            def fill(group):
                image_id, indices = group
                image_path, expected = paths[image_id]
                if sha256(image_path) != expected:
                    raise ValueError(f"Original changed after preparation: {image_path}")
                with Image.open(image_path) as raw:
                    image = raw.convert("RGB")
                for i, x, y, r, b in geometry[indices]:
                    cache[i] = rgb_tile(image, (x, y, r, b), size)

            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for _ in pool.map(fill, tiles.groupby("image_id", sort=False).indices.items()):
                        pass
                cache.flush()
            finally:
                del cache
            stage.replace(path)
    return path


class TileDataset(Dataset):
    def __init__(self, rows, originals, cfg, cache_path):
        self.indices = rows.tile_id.to_numpy(dtype=np.int64)
        self.labels = rows.label.to_numpy(dtype=np.int64)
        self.cache_path = cache_path
        self.cache = None
        self.size = cfg["training"]["image_size"]
        self.last_path = None
        self.last_image = None
        if cache_path is None:
            paths = {r.image_id: Path(cfg["paths"][r.source]) / r.path for r in originals.itertuples()}
            self.paths = rows.image_id.map(paths).tolist()
            self.boxes = rows[COORDS].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        if self.cache_path is None:
            path = self.paths[i]
            if path != self.last_path:
                with Image.open(path) as raw:
                    self.last_image = raw.convert("RGB")
                self.last_path = path
            image = rgb_tile(self.last_image, tuple(self.boxes[i]), self.size)
        else:
            if self.cache is None:
                self.cache = np.load(self.cache_path, mmap_mode="r")
            image = self.cache[self.indices[i]]
        return image, self.labels[i]


def collate(batch):
    images, labels = zip(*batch)
    return torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2), torch.tensor(labels, dtype=torch.long)


def loader(dataset, cfg, device, *, shuffle=False, sampler=None):
    tr = cfg["training"]
    return DataLoader(dataset, batch_size=tr["batch_size"], shuffle=shuffle, sampler=sampler,
                      num_workers=tr["workers"], pin_memory=device.type == "cuda", collate_fn=collate,
                      persistent_workers=tr["workers"] > 0,
                      generator=torch.Generator().manual_seed(cfg["seed"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the optional shared RGB tile cache.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    cfg = config(parser.parse_args().config)
    _, originals, tiles, report = load_prepared(cfg)
    print(input_cache(cfg, originals, tiles, report))
