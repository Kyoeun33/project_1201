"""Read original files and validate image/annotation pairs before assigning splits."""
from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from src.dataset import read_labels, read_official_list
from .common import BOARDS, sha256

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Original:
    image_id: str
    source: str
    board: str
    group_id: str
    path: str
    role: str
    label: int
    width: int
    height: int
    sha256: str
    annotation: str = ""
    boxes: list = field(default_factory=list)
    split: str = ""

    def row(self):
        row = asdict(self)
        row["boxes"] = json.dumps(self.boxes, separators=(",", ":"))
        return row


def images(directory):
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory missing: {directory}")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def original(root, path, source, board, group, role, label, annotation=None):
    rel = path.relative_to(root).as_posix()
    with Image.open(path) as im:
        width, height = im.size
        im.verify()
    return Original(f"{source}/{rel}", source, board, f"{source}/{group}", rel, role, label,
                    width, height, sha256(path),
                    annotation.relative_to(root).as_posix() if annotation else "")


def check_sets(actual, expected, context):
    if actual != expected:
        extra = sorted(str(v) for v in actual - expected)[:3]
        missing = sorted(str(v) for v in expected - actual)[:3]
        raise ValueError(f"{context} mismatch: extra={extra}, missing={missing}")


def deeppcb(root):
    # Reuse the project's reader for the official base-path lists and five-column TXT.
    records = read_official_list(root, "trainval.txt") + read_official_list(root, "test.txt")
    originals = []
    for row in records:
        test, temp, txt = (root / row[k] for k in ("test_path", "temp_path", "label_path"))
        board = Path(row["test_path"]).parts[0]
        if not re.fullmatch(r"group\d+", board):
            raise ValueError(f"Missing DeepPCB board group: {test}")
        item = original(root, test, "DeepPCB", board, board, "tested", 0, txt)
        ref = original(root, temp, "DeepPCB", board, board, "template", 0)
        if (item.width, item.height) != (ref.width, ref.height) or (item.width, item.height) != (640, 640):
            raise ValueError(f"Expected an aligned 640 x 640 pair: {test}, {temp}")
        boxes, _ = read_labels(txt, (item.width, item.height))
        # DeepPCB records corner pixels. Store all rectangles as zero-based half-open boxes.
        for x1, y1, x2, y2 in boxes.astype(int).tolist():
            item.boxes.append([x1, y1, min(x2 + 1, item.width), min(y2 + 1, item.height)])
        item.label = int(bool(item.boxes))
        originals.extend([item, ref])
    return originals


def pku(root):
    if not (root / "Annotations").is_dir():
        raise FileNotFoundError(f"PKU Pascal VOC Annotations missing in {root}. "
                                "Set paths.PKU to PCB_DATASET containing images/ and Annotations/.")
    originals = []
    declared_xml = set()
    for path in images(root / "images"):
        prefix = re.match(r"^(\d+)_", path.stem)
        if not prefix:
            raise ValueError(f"Missing PKU board number: {path}")
        board = f"{int(prefix[1]):02d}"
        xml = root / "Annotations" / path.relative_to(root / "images").with_suffix(".xml")
        tag = ET.parse(xml).getroot()
        item = original(root, path, "PKU", board, board, "tested", 1, xml)
        size = tuple(int(tag.findtext(f"size/{axis}")) for axis in ("width", "height"))
        if size != (item.width, item.height) or Path(tag.findtext("filename", "")).name.lower() != path.name.lower():
            raise ValueError(f"XML/image filename or size mismatch: {xml}")
        for obj in tag.findall("object"):
            x1, y1, x2, y2 = (int(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax"))
            if not (1 <= x1 <= x2 <= item.width and 1 <= y1 <= y2 <= item.height):
                raise ValueError(f"Invalid VOC box in {xml}: {(x1, y1, x2, y2)}")
            item.boxes.append([x1 - 1, y1 - 1, x2, y2])
        if not item.boxes:
            raise ValueError(f"PKU defect image has no annotated object: {xml}")
        declared_xml.add(xml)
        originals.append(item)
    check_sets(set((root / "Annotations").rglob("*.xml")), declared_xml, "PKU annotations/images")
    template_root = root / "PCB_USED"
    if template_root.is_dir():
        for path in images(template_root):
            if not path.stem.isdecimal():
                raise ValueError(f"Missing PKU template board number: {path}")
            board = f"{int(path.stem):02d}"
            originals.append(original(root, path, "PKU", board, board, "template", 0))
    return originals


def visa(root):
    originals = []
    for board in BOARDS:
        base = root / board / "Data"
        refs = {}
        mask_paths = images(base / "Masks" / "Anomaly")
        masks = {p.stem: p for p in mask_paths}
        if len(masks) != len(mask_paths):
            raise ValueError(f"Ambiguous mask stems: {base}")
        anomaly_stems = set()
        for folder, label in (("Normal", 0), ("Anomaly", 1)):
            files = images(base / "Images" / folder)
            if not files:
                raise ValueError(f"Empty VisA image directory: {base / 'Images' / folder}")
            for path in files:
                mask = None
                if label:
                    if path.stem not in masks or path.stem in anomaly_stems:
                        raise ValueError(f"Expected one mask per anomaly: {path}")
                    mask = masks[path.stem]
                    anomaly_stems.add(path.stem)
                group = f"{board}/{folder}/{path.stem}"
                item = original(root, path, "VisA", board, group, folder.lower(), label, mask)
                originals.append(item)
                refs[item.path] = (item.label, item.annotation)
        check_sets(set(masks), anomaly_stems, f"{board} masks/images")
        metadata = root / board / "image_anno.csv"
        if metadata.is_file():
            with metadata.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != len(refs):
                raise ValueError(f"VisA metadata row count mismatch: {metadata}")
            expected = {(r["image"], int(r["label"].lower() != "normal"), r["mask"]) for r in rows}
            check_sets({(p, *v) for p, v in refs.items()}, expected, str(metadata))
    return originals


def scan(cfg):
    records = []
    for source in cfg["sources"]:
        root = Path(cfg["paths"][source])
        if not root.is_dir():
            raise FileNotFoundError(f"Set paths.{source} in the configuration. Missing {root}")
        items = {"DeepPCB": deeppcb, "PKU": pku, "VisA": visa}[source](root)
        if not items:
            raise ValueError(f"No {source} originals found under {root}")
        records.extend(items)
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("Duplicate original image ID")
    return sorted(records, key=lambda item: item.image_id)
