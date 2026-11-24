"""DeepPCB manifest, validation, paired image loading and shared geometry."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = {
    1: "open",
    2: "short",
    3: "mousebite",
    4: "spur",
    5: "spurious copper",
    6: "pin hole",
}
IMAGE_SIZE = (640, 640)
CSV_FIELDS = ("sample_id", "split", "temp_path", "test_path", "label_path")


def _inside_root(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"데이터 루트 밖의 경로: {relative}")
    return path


def read_labels(path: Path, size: tuple[int, int] = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    boxes, classes = [], []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        parts = re.split(r"[\s,]+", raw.strip())
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: 5개 값(x1 y1 x2 y2 class_id)이 필요")
        try:
            x1, y1, x2, y2, cls = map(int, parts)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: 정수 라벨이 아님") from exc
        w, h = size
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"{path}:{line_number}: 박스 좌표가 {size} 범위를 벗어남")
        if cls not in CLASS_NAMES:
            raise ValueError(f"{path}:{line_number}: 알 수 없는 클래스 {cls}")
        boxes.append((x1, y1, x2, y2))
        classes.append(cls)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=np.int64)


def read_official_list(root: Path, name: str) -> list[dict[str, str]]:
    listing = root / name
    if not listing.is_file():
        raise FileNotFoundError(f"공식 목록이 없음: {listing}")
    records = []
    for line_number, line in enumerate(listing.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.strip().split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"{listing}:{line_number}: 이미지 기본 경로와 라벨 경로가 필요")
        image_rel, label_rel = fields
        image = Path(image_rel)
        if image.suffix.lower() not in (".jpg", ".jpeg"):
            raise ValueError(f"{listing}:{line_number}: JPG 기본 경로가 아님")
        base = image.stem.removesuffix("_temp").removesuffix("_test")
        temp_rel = (image.parent / f"{base}_temp.jpg").as_posix()
        test_rel = (image.parent / f"{base}_test.jpg").as_posix()
        records.append({"sample_id": base, "temp_path": temp_rel,
                        "test_path": test_rel, "label_path": Path(label_rel).as_posix()})
    return records


def validate_records(root: Path, records: list[dict[str, str]]) -> dict[str, Counter]:
    seen = set()
    counts = {"trainval": Counter(), "test": Counter()}
    for record in records:
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"중복 sample_id: {sample_id}")
        seen.add(sample_id)
        for key in ("temp_path", "test_path", "label_path"):
            path = _inside_root(root, record[key])
            if not path.is_file():
                raise FileNotFoundError(f"{sample_id}: 누락 파일 {path}")
        for key in ("temp_path", "test_path"):
            image_path = _inside_root(root, record[key])
            with Image.open(image_path) as image:
                image.load()
                if image.size != IMAGE_SIZE:
                    raise ValueError(f"{image_path}: {IMAGE_SIZE}가 아닌 {image.size}")
        boxes, classes = read_labels(_inside_root(root, record["label_path"]))
        if len(boxes) != len(classes):
            raise AssertionError("박스와 클래스 수가 맞지 않음")
        record["_classes"] = classes.tolist()  # split 진단에만 사용; CSV에서는 제외
        counts[record["split"]].update(record["_classes"])
    return counts


def split_trainval(records: list[dict[str, str]], seed: int = 42,
                   val_count: int = 200) -> list[dict[str, str]]:
    if not 0 < val_count < len(records):
        raise ValueError("val_count는 학습용 목록의 0과 전체 수 사이여야 함")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    val_ids = set(order[:val_count])
    for i, record in enumerate(records):
        record["split"] = "val" if i in val_ids else "train"
    for split in ("train", "val"):
        classes = {cls for r in records if r["split"] == split for cls in r["_classes"]}
        missing = set(CLASS_NAMES) - classes
        if missing:
            raise ValueError(f"{split} 분할에 클래스 누락: {sorted(missing)}")
    return records


def write_split_csv(records: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({key: record[key] for key in CSV_FIELDS} for record in records)


def read_split_csv(path: Path, split: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(CSV_FIELDS) - set(reader.fieldnames or []):
            raise ValueError(f"분할 CSV 열 누락: {path}")
        return [row for row in reader if row["split"] == split]


def load_pair(root: Path, record: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(_inside_root(root, record["temp_path"])) as image:
        temp = np.asarray(image.convert("L")).copy()
    with Image.open(_inside_root(root, record["test_path"])) as image:
        test = np.asarray(image.convert("L")).copy()
    boxes, classes = read_labels(_inside_root(root, record["label_path"]))
    return temp, test, boxes, classes


def augment_pair(temp: np.ndarray, test: np.ndarray, boxes: np.ndarray,
                 rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same multiples-of-90 rotation and flips to both images and boxes.

    Boxes use half-open [x1,y1,x2,y2] coordinates throughout the project.
    """
    boxes = boxes.copy()
    for _ in range(rng.randrange(4)):
        h, w = temp.shape
        temp, test = np.rot90(temp), np.rot90(test)
        if len(boxes):
            x1, y1, x2, y2 = boxes.T.copy()
            boxes = np.stack((y1, w - x2, y2, w - x1), axis=1)
    if rng.choice((False, True)):
        w = temp.shape[1]
        temp, test = np.fliplr(temp), np.fliplr(test)
        if len(boxes):
            x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
            boxes[:, 0], boxes[:, 2] = w - x2, w - x1
    if rng.choice((False, True)):
        h = temp.shape[0]
        temp, test = np.flipud(temp), np.flipud(test)
        if len(boxes):
            y1, y2 = boxes[:, 1].copy(), boxes[:, 3].copy()
            boxes[:, 1], boxes[:, 3] = h - y2, h - y1
    return np.ascontiguousarray(temp), np.ascontiguousarray(test), boxes


def save_previews(root: Path, records: list[dict[str, str]], out: Path, count: int = 6) -> None:
    out.mkdir(parents=True, exist_ok=True)
    colors = (None, "#e53935", "#f57c00", "#43a047", "#1e88e5", "#8e24aa", "#00897b")
    for record in records[:count]:
        _, test, boxes, classes = load_pair(root, record)
        image = Image.fromarray(test).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box, cls in zip(boxes.astype(int), classes):
            draw.rectangle(tuple(box), outline=colors[int(cls)], width=2)
            draw.text((int(box[0]), max(0, int(box[1]) - 12)),
                      f"{int(cls)} {CLASS_NAMES[int(cls)]}", fill=colors[int(cls)])
        image.save(out / f"{record['sample_id']}.png")
