"""Read-only image audit and COCO/YOLO OBB annotation matching for Mendeley PCB."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

from .common import read_json, sha256


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PUBLISHED_V2 = {"image_files": 230, "annotation_rows": 2506}


def match_images(paths):
    relative = {path.casefold(): path for path in paths}
    basenames = defaultdict(list)
    for path in paths:
        basenames[PurePosixPath(path).name.casefold()].append(path)

    def match(filename):
        key = PurePosixPath(str(filename).replace("\\", "/")).as_posix().casefold()
        if key in relative:
            return relative[key]
        matches = basenames[PurePosixPath(key).name]
        return matches[0] if len(matches) == 1 else None

    return match


def coco_annotations(path, images, issue, result):
    document = read_json(path)
    categories = {entry["id"]: entry["name"] for entry in document["categories"]}
    counts = Counter(annotation["image_id"] for annotation in document["annotations"])
    match = match_images(images)
    links = {}
    mapped = set()
    bad_dimensions = set()
    for entry in document["images"]:
        image_id = entry["id"]
        if image_id in links:
            raise ValueError(f"Duplicate COCO image ID: {image_id}")
        image_path = match(entry["file_name"])
        links[image_id] = image_path
        if image_path is None:
            issue("annotation_without_image", entry["file_name"],
                  f"COCO image ID {image_id} has no unique image match ({counts[image_id]} objects)")
            continue
        if image_path in mapped:
            raise ValueError(f"Multiple COCO image records resolve to {image_path}")
        mapped.add(image_path)
        image = images[image_path]
        expected_size = (entry.get("width"), entry.get("height"))
        if (image["status"] == "ok" and None not in expected_size
                and expected_size != (image["width"], image["height"])):
            bad_dimensions.add(image_path)
            issue("annotation_size_mismatch", image_path,
                  f"COCO dimensions {expected_size} differ from the decoded image. Box overlay omitted.")

    for entry in document["annotations"]:
        object_id = entry.get("id", "")
        image_id = entry["image_id"]
        if image_id not in links:
            issue("annotation_without_image", str(object_id), f"Unknown COCO image ID {image_id}")
            continue
        image_path = links[image_id]
        if image_path is None:
            continue
        try:
            category = categories[entry["category_id"]]
            x, y, width, height = map(float, entry["bbox"])
            if not all(map(math.isfinite, (x, y, width, height))) or min(width, height) <= 0:
                raise ValueError("Expected a finite COCO [x, y, width, height] with positive size")
        except (KeyError, TypeError, ValueError) as error:
            issue("invalid_annotation", image_path, f"COCO annotation {object_id}: {error}")
            continue
        points = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
        result[image_path].append((category, None if image_path in bad_dimensions else points))
    return mapped, len(document["annotations"]), sorted(set(categories.values()))


def obb_annotations(path, images, classes, issue, result):
    if not isinstance(classes, dict) or not classes:
        raise ValueError("Set obb_classes to the class-ID-to-name mapping for these YOLO OBB labels")
    root = path if path.is_dir() else path.parent
    files = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
    by_stem = defaultdict(list)
    for image_path in images:
        by_stem[str(PurePosixPath(image_path).with_suffix("")).casefold()].append(image_path)
    match = match_images(by_stem)
    raw_count = 0
    mapped = set()
    for file in files:
        relative = file.relative_to(root).as_posix()
        lines = [(number, line.split()) for number, line in
                 enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1) if line.strip()]
        raw_count += len(lines)
        key = match(str(PurePosixPath(relative).with_suffix("")))
        candidates = by_stem.get(key, [])
        if len(candidates) != 1:
            issue("annotation_without_image", relative,
                  f"No unique image with this relative stem or basename ({len(lines)} objects)")
            continue
        image_path = candidates[0]
        if image_path in mapped:
            raise ValueError(f"Multiple OBB label files resolve to {image_path}")
        mapped.add(image_path)
        image = images[image_path]
        for number, fields in lines:
            try:
                category = classes[str(int(fields[0]))]
                coords = list(map(float, fields[1:]))
                if len(coords) != 8 or not all(0 <= value <= 1 for value in coords):
                    raise ValueError("Expected four normalized OBB corner pairs in [0, 1]")
                area_twice = sum(
                    coords[i] * coords[(i + 3) % 8] - coords[i + 1] * coords[(i + 2) % 8]
                    for i in range(0, 8, 2)
                )
                if not area_twice:
                    raise ValueError("OBB polygon has zero area")
            except (KeyError, ValueError) as error:
                issue("invalid_annotation", image_path, f"{relative}:{number}: {error}")
                continue
            points = None
            if image["status"] == "ok":
                points = [[x * image["width"], y * image["height"]]
                          for x, y in zip(coords[::2], coords[1::2])]
            result[image_path].append((category, points))
    return mapped, raw_count, sorted(set(classes.values()))


def audit_dataset(image_dir, annotations, annotation_format, classes, output):
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    issues = []

    def issue(kind, path, detail):
        issues.append({"kind": kind, "path": path, "detail": detail})

    images = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(image_dir).as_posix()
        row = {"image_path": relative, "width": None, "height": None, "status": "ok", "sha256": None}
        try:
            with Image.open(path) as image:
                row["width"], row["height"] = image.size
                image.load()
            row["sha256"] = sha256(path)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            row["status"] = "unreadable"
            issue("unreadable_image", relative, str(error))
        images[relative] = row

    boxes = defaultdict(list)
    mapped, raw_count, names = set(), None, []
    annotation_status = "ok"
    try:
        if annotation_format == "coco":
            mapped, raw_count, names = coco_annotations(annotations, images, issue, boxes)
        else:
            if not annotations.is_dir() and not (annotations.is_file() and annotations.suffix.lower() == ".txt"):
                raise ValueError("Use a YOLO OBB label directory or a corresponding .txt label file")
            mapped, raw_count, names = obb_annotations(annotations, images, classes, issue, boxes)
    except (OSError, KeyError, TypeError, ValueError) as error:
        annotation_status = "unavailable"
        boxes.clear()
        issue("annotation_source_error", str(annotations), str(error))

    for path, row in images.items():
        records = boxes[path]
        class_names = sorted({name for name, _ in records})
        row["annotation_linked"] = path in mapped
        row["defect_box_count"] = len(records) if path in mapped else None
        row["defect_class_count"] = len(class_names) if path in mapped else None
        row["original_classes"] = json.dumps(class_names, ensure_ascii=False) if path in mapped else None
        if path not in mapped:
            issue("image_without_annotation", path, "No linked annotation record or label file")
        elif not records:
            issue("image_without_valid_box", path, "Linked annotation contains no valid defect boxes")

    observed = {"image_files": len(images), "annotation_rows": raw_count}
    summary = {
        "dataset": "Mendeley PCB-Defect",
        "image_dir": str(image_dir),
        "annotation_path": str(annotations),
        "annotation_format": annotation_format,
        "annotation_status": annotation_status,
        "published_v2_reference": PUBLISHED_V2,
        "observed": observed,
        "difference_from_published_v2": {
            key: None if value is None else value - PUBLISHED_V2[key] for key, value in observed.items()
        },
        "readable_images": sum(row["status"] == "ok" for row in images.values()),
        "linked_images": len(mapped),
        "valid_boxes_linked": sum(map(len, boxes.values())),
        "annotation_categories": names,
        "classes_with_linked_boxes": sorted({name for records in boxes.values() for name, _ in records}),
        "known_normal_images": 0,
        "image_ground_truth": "defect for every image in the PCB-Defect cohort",
        "issue_counts": dict(Counter(row["kind"] for row in issues)),
    }
    pd.DataFrame(images.values(), columns=[
        "image_path", "width", "height", "status", "sha256", "annotation_linked",
        "defect_box_count", "defect_class_count", "original_classes",
    ]).to_csv(output / "audit_images.csv", index=False)
    pd.DataFrame(issues, columns=["kind", "path", "detail"]).to_csv(output / "audit_issues.csv", index=False)
    return images, boxes, summary
