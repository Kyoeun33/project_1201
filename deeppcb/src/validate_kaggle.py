"""Kaggle PCB Defects 원본을 DeepPCB V2 모델로 평가한다.

이 파일을 deeppcb/src/deeppcb_v2.py 옆에 놓고 프로젝트 루트에서 실행한다.

    python src/validate_kaggle.py audit
    python src/validate_kaggle.py evaluate --limit 6
    python src/validate_kaggle.py evaluate

데이터 기본 경로는 out_data/PCB_DATASET, 모델 경로는 runs/v2이다.
audit은 원본 images, PCB_USED, Annotations를 읽고 회로 윤곽을 비교한다.
evaluate는 같은 전처리를 수행하고 DeepPCB val에서 보정한 임계값을 사용한다.
원본 이미지 좌표계에서 채점하며, CSV 박스 좌표는 0부터 시작하는 xyxy 반열림 구간이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from scipy import fft, ndimage

from candidates import box_iou
from dataset import IMAGE_SIZE


ROOT = Path(__file__).resolve().parents[1]
TILE = IMAGE_SIZE[0]
KNOWN = ("Open_circuit", "Short", "Mouse_bite", "Spur", "Spurious_copper")
UNSEEN = "Missing_hole"
CLASSES = (*KNOWN, UNSEEN)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BILINEAR = Image.Resampling.BILINEAR

# 정답 라벨을 사용하지 않는 정렬 기준. 전처리 설정은 결과 JSON에도 저장한다.
REGISTRATION = {"thumbnail_max_side": 800, "identity_ncc": 0.995,
                "minimum_ncc": 0.80, "max_rotation_deg": 3.0,
                "max_shift_fraction": 0.05, "aspect_ratio_tolerance": 0.005}


@dataclass
class Sample:
    sample_id: str
    board: str
    image: Path | None
    template: Path | None
    annotation: Path | None
    problem: str


@dataclass(frozen=True)
class Truth:
    label: str
    box: tuple[int, int, int, int]


class SampleIssue(ValueError):
    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage


def file_index(folder: Path, extensions: set[str]) -> dict[str, list[Path]]:
    index = defaultdict(list)
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.casefold() in extensions:
            key = path.relative_to(folder).with_suffix("").as_posix().casefold()
            index[key].append(path)
    return index


def discover(root: Path) -> list[Sample]:
    images, templates, annotations = (root / name for name in ("images", "PCB_USED", "Annotations"))
    for folder in (images, templates, annotations):
        if not folder.is_dir():
            raise FileNotFoundError(f"필수 폴더가 없습니다: {folder}. --data-root로 PCB_DATASET을 지정하세요.")
    image_index = file_index(images, IMAGE_EXTENSIONS)
    xml_index = file_index(annotations, {".xml"})
    by_board = defaultdict(list)
    for paths in file_index(templates, IMAGE_EXTENSIONS).values():
        for path in paths:
            if path.stem.isascii() and path.stem.isdigit():
                by_board[int(path.stem)].append(path)

    by_folder = defaultdict(list)
    for key in sorted(image_index.keys() | xml_index.keys()):
        pictures, labels = image_index.get(key, []), xml_index.get(key, [])
        source = pictures[0] if pictures else labels[0]
        sample_id = source.relative_to(images if pictures else annotations).with_suffix("").as_posix()
        board = source.stem.split("_", 1)[0]
        refs = by_board.get(int(board), []) if board.isascii() and board.isdigit() else []
        issues = [f"{name} {len(files)}개" for name, files in
                  (("검사 이미지", pictures), ("템플릿", refs), ("XML", labels)) if len(files) != 1]
        by_folder[Path(key).parent.as_posix()].append(
            Sample(sample_id, board, pictures[0] if pictures else None,
                   refs[0] if len(refs) == 1 else None, labels[0] if labels else None, ", ".join(issues))
        )
    if not by_folder:
        raise ValueError(f"검사 이미지나 XML이 없습니다: {root}")
    # 부분 실행에서도 가능한 한 여러 폴더가 포함되도록 번갈아 선택한다.
    return [sample for group in zip_longest(*by_folder.values()) for sample in group if sample is not None]


def label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


LABEL_MAP = {label_key(name): name for name in CLASSES}
LABEL_MAP.update({"open": "Open_circuit", "shortcircuit": "Short"})


def read_truth(sample: Sample, size: tuple[int, int]) -> list[Truth]:
    try:
        root = ET.parse(sample.annotation).getroot()
        if root.tag != "annotation":
            raise ValueError("Pascal VOC annotation 루트가 필요합니다.")
        xml_name = (root.findtext("filename") or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
        if xml_name and xml_name.casefold() != sample.image.name.casefold():
            raise ValueError(f"XML filename={xml_name}, 검사 파일={sample.image.name}")
        declared = tuple(int(root.findtext(f"size/{key}", "")) for key in ("width", "height"))
        if declared != size:
            raise ValueError(f"XML 크기 {declared}, 실제 검사 이미지 크기 {size}")
        truths = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            label = LABEL_MAP.get(label_key(name))
            if label is None:
                raise ValueError(f"지원하지 않는 Kaggle 결함 라벨 {name!r}")
            x1, y1, x2, y2 = (int(obj.findtext(f"bndbox/{key}", ""))
                              for key in ("xmin", "ymin", "xmax", "ymax"))
            if not (1 <= x1 <= x2 <= size[0] and 1 <= y1 <= y2 <= size[1]):
                raise ValueError(f"VOC 좌표가 이미지 범위를 벗어납니다: {(x1, y1, x2, y2)}")
            truths.append(Truth(label, (x1 - 1, y1 - 1, x2, y2)))
        if not truths and Path(sample.sample_id).parent.name.casefold() != "normal":
            raise ValueError("결함 폴더의 XML에 object가 없습니다.")
        return truths
    except (OSError, ET.ParseError, ValueError) as exc:
        raise SampleIssue("annotation_error", str(exc)) from exc


def read_gray(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L")


@dataclass(frozen=True)
class Alignment:
    method: str
    angle_deg: float
    dx: float
    dy: float
    ncc_before: float
    ncc_after: float


def detail(image: Image.Image) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    return values - ndimage.gaussian_filter(values, sigma=2.0)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    # 가장자리 채움의 영향을 줄이고 보드 내부 회로 패턴의 일치도를 계산한다.
    h, w = first.shape
    my, mx = int(h * 0.08), int(w * 0.08)
    a, b = first[my:h-my, mx:w-mx], second[my:h-my, mx:w-mx]
    a, b = a - a.mean(), b - b.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    return float(np.clip(np.sum(a * b) / denom, -1, 1)) if denom > 1e-6 else 0.0


def phase_shift(ref_detail: np.ndarray, test_fft: np.ndarray, window: np.ndarray) -> tuple[float, float]:
    cross = test_fft * fft.rfft2(ref_detail * window).conj()
    cross /= np.maximum(np.abs(cross), 1e-6)
    response = fft.irfft2(cross, s=ref_detail.shape)
    row, col = np.unravel_index(np.argmax(response), response.shape)

    def offset(line, i):
        left, mid, right = (float(line[j % len(line)]) for j in (i - 1, i, i + 1))
        curvature = left - 2 * mid + right
        fraction = np.clip(0.5 * (left - right) / curvature, -0.5, 0.5) if curvature < -1e-9 else 0.0
        return float((i if i <= len(line) // 2 else i - len(line)) + fraction)

    return offset(response[row], col), offset(response[:, col], row)


def estimate_alignment(template: Image.Image, inspected: Image.Image) -> Alignment:
    reference = template.copy()
    reference.thumbnail((REGISTRATION["thumbnail_max_side"],) * 2, BILINEAR)
    target = detail(inspected.resize(reference.size, BILINEAR))
    before = correlation(detail(reference), target)
    identity = Alignment("identity", 0.0, 0.0, 0.0, before, before)
    if before >= REGISTRATION["identity_ncc"]:
        return identity

    w, h = reference.size
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    target_fft = fft.rfft2(target * window)
    best = identity
    visited = set()

    def try_angle(angle: float):
        nonlocal best
        if angle in visited or abs(angle) > REGISTRATION["max_rotation_deg"]:
            return
        visited.add(angle)
        rotated = detail(reference.rotate(angle, resample=BILINEAR))
        dx, dy = phase_shift(rotated, target_fft, window)
        if max(abs(dx) / w, abs(dy) / h) > REGISTRATION["max_shift_fraction"]:
            return
        shifted = ndimage.shift(rotated, (dy, dx), order=1, mode="nearest", prefilter=False)
        score = correlation(shifted, target)
        if score > best.ncc_after:
            best = Alignment("rigid", angle, dx * template.width / w, dy * template.height / h, before, score)

    for angle in (0.0, -1.5, 1.5, -3.0, 3.0):
        try_angle(angle)
    for step in (0.75, 0.375, 0.1875, 0.09375):
        center = best.angle_deg
        try_angle(center - step)
        try_angle(center + step)
    if best.ncc_after < REGISTRATION["minimum_ncc"]:
        raise SampleIssue("alignment_error", f"회로 패턴 NCC 부족 (보정 전 {before:.4f}, 최선 {best.ncc_after:.4f})")
    if before >= REGISTRATION["minimum_ncc"] and best.ncc_after - before < 0.001:
        return identity
    return best


@dataclass
class Pair:
    template: np.ndarray
    image: np.ndarray
    valid: np.ndarray | None
    alignment: Alignment
    original_template: Image.Image
    template_size: tuple[int, int]


def prepare(sample: Sample) -> Pair:
    if sample.problem:
        raise SampleIssue("pair_error", sample.problem)
    try:
        reference, inspected = read_gray(sample.template), read_gray(sample.image)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SampleIssue("image_error", str(exc)) from exc

    source_size = reference.size
    if source_size != inspected.size:
        aspect_change = (reference.width / reference.height) / (inspected.width / inspected.height) - 1
        if abs(aspect_change) > REGISTRATION["aspect_ratio_tolerance"]:
            raise SampleIssue("alignment_error", f"템플릿과 검사 이미지 종횡비가 다릅니다: {source_size}, {inspected.size}")
        reference = reference.resize(inspected.size, BILINEAR)
    alignment = estimate_alignment(reference, inspected)
    image = np.asarray(inspected)
    if alignment.method == "identity":
        return Pair(np.asarray(reference), image, None, alignment, reference, source_size)
    warp = dict(angle=alignment.angle_deg, translate=(alignment.dx, alignment.dy), resample=BILINEAR, fillcolor=0)
    corrected = np.asarray(reference.rotate(**warp))
    footprint = np.asarray(Image.new("L", inspected.size, 255).rotate(**warp)) == 255
    # 시야 밖 영역은 양쪽 입력에 같은 값으로 채우고 예측 중심도 검사한다.
    # 원본 이미지의 모든 GT를 채점 분모에 유지한다.
    return Pair(np.where(footprint, corrected, 0), np.where(footprint, image, 0),
                footprint, alignment, reference, source_size)


def tile_starts(length: int, overlap: int) -> list[int]:
    last = max(0, length - TILE)
    starts = list(range(0, last + 1, TILE - overlap))
    if starts[-1] != last:
        starts.append(last)
    return starts


def tile_at(array: np.ndarray, x: int, y: int) -> np.ndarray:
    tile = array[y:y+TILE, x:x+TILE]
    h, w = tile.shape
    if h < TILE or w < TILE:
        tile = np.pad(tile, ((0, TILE - h), (0, TILE - w)), mode="edge")
    return np.ascontiguousarray(tile)


@dataclass
class Runner:
    kind: str
    model: object
    config: dict
    device: object
    threshold: float
    info: dict


def load_runners(args) -> list[Runner]:
    from deeppcb_v2 import restore_model

    runners = []
    for kind in (("fast", "accurate") if args.model == "both" else (args.model,)):
        options = argparse.Namespace(kind=kind, device=args.device, runs_dir=args.runs_dir)
        model, checkpoint, digest, device = restore_model(options)
        path = args.runs_dir / f"{kind}_v2_calibration.json"
        calibration = json.loads(path.read_text(encoding="utf-8"))
        if (calibration.get("kind") != kind or calibration.get("checkpoint_sha256") != digest or
                calibration.get("split_sha256") != checkpoint["split_sha256"]):
            raise ValueError(f"{kind} 가중치와 임계값 파일이 일치해야 합니다: {path}")
        threshold = float(calibration["threshold"])
        # V2 보정에서 1.000001은 모든 박스를 제외하는 유효한 값이다.
        if not 0.0 <= threshold <= 1.000001:
            raise ValueError(f"보정 파일의 임계값이 범위를 벗어납니다: {threshold}")
        config = checkpoint["config"]
        info = {"checkpoint_sha256": digest, "split_sha256": checkpoint["split_sha256"],
                "epoch": checkpoint["epoch"], "train_pairs": checkpoint.get("train_pairs"),
                "calibration_file": str(path.resolve()), "calibration_pairs": calibration["val_pairs"],
                "partial_calibration": calibration["partial_val"], "threshold": threshold,
                "nms_iou": float(config["model"]["nms_iou"]), "device": str(device)}
        runners.append(Runner(kind, model, config, device, threshold, info))
        print(f"V2-{kind} | {device} | val 임계값 {threshold:.8g} | val {calibration['val_pairs']}쌍")
    if len({r.info["split_sha256"] for r in runners}) > 1:
        raise ValueError("두 모델을 비교하려면 동일한 DeepPCB 분할에서 학습한 체크포인트가 필요합니다.")
    return runners


def original_box(local_box, x, y, width, height, footprint):
    left, top, right, bottom = local_box
    cx, cy = x + (left + right) / 2, y + (top + bottom) / 2
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    if footprint is not None and not footprint[int(cy), int(cx)]:
        return None
    left, top = max(0.0, x + left), max(0.0, y + top)
    right, bottom = min(float(width), x + right), min(float(height), y + bottom)
    return (left, top, right, bottom) if left < right and top < bottom else None


@dataclass
class ModelOutput:
    detections: list
    tiles: int
    tile_boxes: int
    max_tile_score: float | None
    elapsed_ms: float


def detect(runner: Runner, pair: Pair, xs: list[int], ys: list[int]) -> ModelOutput:
    import torch
    from torchvision.ops import nms
    from deeppcb_v2 import Detection, predict_boxes

    synchronize(runner.device)
    start = perf_counter()
    boxes, scores = [], []
    tile_boxes, max_tile_score = 0, None
    height, width = pair.image.shape
    for y in ys:
        for x in xs:
            output = predict_boxes(runner.kind, runner.model, runner.config, tile_at(pair.template, x, y),
                                   tile_at(pair.image, x, y), runner.device)
            tile_boxes += len(output)
            if output:
                top_score = output[0].score  # predict_boxes는 점수 내림차순으로 반환한다.
                max_tile_score = top_score if max_tile_score is None else max(max_tile_score, top_score)
            for item in output:
                if item.score >= runner.threshold:
                    box = original_box(item.box, x, y, width, height, pair.valid)
                    if box is not None:
                        boxes.append(box)
                        scores.append(item.score)
    kept = []
    if boxes:
        keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores, dtype=torch.float32),
                   runner.info["nms_iou"])
        kept = [Detection(boxes[i], scores[i]) for i in keep.tolist()]
    synchronize(runner.device)
    return ModelOutput(kept, len(xs) * len(ys), tile_boxes, max_tile_score, (perf_counter() - start) * 1000)


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def warm_up(runner: Runner, pair: Pair):
    from deeppcb_v2 import predict_boxes
    boxes = predict_boxes(runner.kind, runner.model, runner.config, tile_at(pair.template, 0, 0),
                          tile_at(pair.image, 0, 0), runner.device)
    if runner.kind == "fast" and not boxes:
        import torch
        side = int(runner.config["model"]["patch_size"])
        with torch.inference_mode():
            runner.model(torch.zeros((1, 4, side, side), device=runner.device))
    synchronize(runner.device)


def match(detections, truths: list[Truth], iou_threshold: float):
    # 전역 NMS는 점수 내림차순으로 반환한다. 한 정답을 최대 한 번 매칭한다.
    available = list(range(len(truths)))
    pairs = []
    for detection in detections:
        overlaps = [(i, box_iou(detection.box, truths[i].box)) for i in available]
        index, iou = max(overlaps, key=lambda item: item[1], default=(None, 0.0))
        if index is not None and iou >= iou_threshold:
            available.remove(index)
            pairs.append((index, iou))
        else:
            pairs.append((None, None))
    return pairs


def group_of(labels) -> str:
    if not labels:
        return "normal"
    if UNSEEN not in labels:
        return "known_only"
    return "missing_hole_only" if all(label == UNSEEN for label in labels) else "mixed"


@dataclass
class ImageResult:
    group: str
    labels: list[str]
    matched_labels: list[str]
    predictions: int
    prepare_ms: float
    inference_ms: float


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def location_metrics(tp, fp, fn):
    return {"gt_boxes": tp + fn, "predicted_boxes": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, 2 * tp + fn + fp)}


def time_stats(values):
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def group_metrics(records: list[ImageResult]) -> dict:
    positives = sum(bool(r.labels) for r in records)
    normals = len(records) - positives
    tp = sum(bool(r.labels) and r.predictions > 0 for r in records)
    fp = sum(not r.labels and r.predictions > 0 for r in records)
    fn, tn = positives - tp, normals - fp
    gt = sum(len(r.labels) for r in records)
    matched = sum(len(r.matched_labels) for r in records)
    pred = sum(r.predictions for r in records)
    return {
        "images": len(records),
        "image": {"defective": positives, "normal": normals, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                  "recall": ratio(tp, positives), "fnr": ratio(fn, positives), "fpr": ratio(fp, normals),
                  "precision": ratio(tp, tp + fp) if normals else None,
                  "f1": ratio(2 * tp, 2 * tp + fp + fn) if normals else None},
        "boxes": {**location_metrics(matched, pred - matched, gt - matched),
                  "fp_per_image": ratio(pred - matched, len(records))},
        "latency_ms_per_image": {
            "inference_and_nms": time_stats([r.inference_ms for r in records]),
            "with_io_and_alignment": time_stats([r.prepare_ms + r.inference_ms for r in records])},
    }


def summarize(records: list[ImageResult]) -> dict:
    result = {
        "method_1_known": group_metrics([r for r in records if r.group in ("known_only", "normal")]),
        "method_2_all": group_metrics(records),
        "missing_hole_only_images": group_metrics([r for r in records if r.group == "missing_hole_only"]),
        "mixed_images": sum(r.group == "mixed" for r in records),
    }
    by_class = {}
    for label in CLASSES:
        count = sum(r.labels.count(label) for r in records)
        hits = sum(r.matched_labels.count(label) for r in records)
        by_class[label] = {"images_with_class": sum(label in r.labels for r in records),
                           "gt_boxes": count, "matched": hits, "missed": count - hits,
                           "recall": ratio(hits, count)}
    result["per_class_all_images"] = by_class
    return result


def edge_view(image: Image.Image, size) -> np.ndarray:
    gray = np.asarray(image.resize(size, BILINEAR), dtype=np.float32)
    edges = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    return (255 * np.clip(edges / max(1.0, float(np.percentile(edges, 97))), 0, 1)).astype(np.uint8)


def save_alignment(path: Path, pair: Pair):
    measured = Image.fromarray(pair.image)
    measured.thumbnail((650, 500), BILINEAR)
    target = edge_view(measured, measured.size)
    w, h = measured.size
    canvas = Image.new("RGB", (2 * w, h + 24), "black")
    draw = ImageDraw.Draw(canvas)
    captions = ("BEFORE", f"AFTER   NCC {pair.alignment.ncc_after:.3f}")
    for i, reference in enumerate((pair.original_template, Image.fromarray(pair.template))):
        ref = edge_view(reference, measured.size)
        overlay = Image.fromarray(np.stack((ref, target, target), axis=-1))
        canvas.paste(overlay, (i * w, 24))
        draw.text((i * w + 5, 5), captions[i] + "    red=template   cyan=inspection", fill="white")
    canvas.save(path)


def save_predictions(path, pair, truths, detections):
    preview = Image.fromarray(pair.image).convert("RGB")
    preview.thumbnail((1280, 940), BILINEAR)
    sx, sy = preview.width / pair.image.shape[1], preview.height / pair.image.shape[0]
    draw = ImageDraw.Draw(preview)
    for color, items in (("#ffae32", detections), ("#37edda", truths)):
        for item in items:
            x1, y1, x2, y2 = item.box
            draw.rectangle((x1 * sx, y1 * sy, x2 * sx, y2 * sy), outline=color, width=2)
    draw.rectangle((0, 0, 310, 20), fill="black")
    draw.text((5, 4), "cyan = XML truth     orange = V2 prediction", fill="white")
    preview.save(path)


MANIFEST_FIELDS = (
    "sample_id board image template annotation status reason width height template_width template_height "
    "alignment angle_deg dx dy ncc_before ncc_after valid_fraction tiles group gt_boxes"
).split()
IMAGE_FIELDS = (
    "model sample_id group truth decision gt_classes gt_boxes predicted_boxes matched fp_boxes fn_boxes "
    "threshold tile_boxes_before_threshold max_tile_score max_kept_score tiles prepare_ms inference_nms_ms total_ms"
).split()


def open_table(stack, folder, name, fields):
    handle = stack.enter_context((folder / name).open("w", newline="", encoding="utf-8-sig"))
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return writer


def record_scores(tables, runner, sample, group, truths, output: ModelOutput, pairs, prepare_ms):
    detections = output.detections
    labels = [obj.label for obj in truths]
    by_truth = {}
    for number, (detection, (index, iou)) in enumerate(zip(detections, pairs), 1):
        if index is not None:
            by_truth[index] = (number, iou)
        x1, y1, x2, y2 = detection.box
        tables["predictions"].writerow({
            "model": runner.kind, "sample_id": sample.sample_id, "prediction_id": number,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": detection.score,
            "matched_gt_id": index + 1 if index is not None else None,
            "matched_gt_class": labels[index] if index is not None else None, "iou": iou})
    for index, obj in enumerate(truths):
        number, iou = by_truth.get(index, (None, None))
        x1, y1, x2, y2 = obj.box
        tables["ground_truth"].writerow({
            "model": runner.kind, "sample_id": sample.sample_id, "gt_id": index + 1,
            "gt_class": obj.label, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "matched_prediction_id": number, "iou": iou})
    tables["images"].writerow({
        "model": runner.kind, "sample_id": sample.sample_id, "group": group,
        "truth": "FAIL" if truths else "PASS", "decision": "FAIL" if detections else "PASS",
        "gt_classes": "|".join(sorted(set(labels))), "gt_boxes": len(truths),
        "predicted_boxes": len(detections), "matched": len(by_truth), "fp_boxes": len(detections) - len(by_truth),
        "fn_boxes": len(truths) - len(by_truth), "threshold": runner.threshold,
        "tile_boxes_before_threshold": output.tile_boxes, "max_tile_score": output.max_tile_score,
        "max_kept_score": detections[0].score if detections else None, "tiles": output.tiles,
        "prepare_ms": round(prepare_ms, 3), "inference_nms_ms": round(output.elapsed_ms, 3),
        "total_ms": round(prepare_ms + output.elapsed_ms, 3)})
    return ImageResult(group, labels, [labels[i] for i in by_truth], len(detections), prepare_ms, output.elapsed_ms)


def run(args, samples, discovered_count, runners, folder) -> dict:
    preview_dir = folder / "previews"
    if args.previews:
        preview_dir.mkdir()
    stages, groups, class_counts = Counter(), Counter(), Counter()
    histories = {r.kind: [] for r in runners}
    summary = {
        "command": args.command, "data_root": str(args.data_root.resolve()),
        "discovered": discovered_count, "selected": len(samples), "limit": args.limit,
        "partial_selection": len(samples) < discovered_count,
        "selection_order": "sorted paths, round-robin over image folders",
        "coordinates": "Pascal VOC 1-based inclusive XML to 0-based half-open xyxy in the inspected image",
        "tile_size": TILE, "overlap_px": args.overlap, "registration": REGISTRATION,
        "input": "Pillow grayscale L, native inspection scale, right/bottom edge padding for small images",
        "coverage_policy": "manifest errors excluded from both models; all XML boxes on accepted images counted",
        "common_classes": KNOWN, "unseen_class": UNSEEN,
    }
    if runners:
        import torch
        from deeppcb_v2 import MATCH_IOU
        summary.update(
            match_iou=MATCH_IOU, checkpoint_dir=str(args.runs_dir.resolve()),
            decision_rule="at least one thresholded detection after full-image NMS gives FAIL",
            method_1="known-only defective images and annotated independent normals",
            method_2="all accepted images, including Missing_hole and mixed images",
            one_class_detection="type-specific recall uses the class of the matched XML box",
            tile_boxes_before_threshold="sum of V2 tile outputs before threshold and global NMS; overlaps can repeat a defect",
            metric_population="ratios with empty denominators are null; image precision/F1 require normal examples",
            latency="per original image. Common time includes grayscale I/O and alignment. Model time includes tiling, "
                    "transfers, inference, threshold and global NMS. XML, scoring, previews and warm-up excluded.",
            torch_version=str(torch.__version__),
            device_name=torch.cuda.get_device_name(runners[0].device) if runners[0].device.type == "cuda" else "CPU")

    with ExitStack() as stack:
        manifest = open_table(stack, folder, "manifest.csv", MANIFEST_FIELDS)
        tables = {}
        if runners:
            tables["images"] = open_table(stack, folder, "images.csv", IMAGE_FIELDS)
            tables["predictions"] = open_table(stack, folder, "predictions.csv", (
                "model sample_id prediction_id x1 y1 x2 y2 score matched_gt_id matched_gt_class iou").split())
            tables["ground_truth"] = open_table(stack, folder, "ground_truth.csv", (
                "model sample_id gt_id gt_class x1 y1 x2 y2 matched_prediction_id iou").split())
        for number, sample in enumerate(samples, 1):
            entry = {"sample_id": sample.sample_id, "board": sample.board}
            for field in ("image", "template", "annotation"):
                path = getattr(sample, field)
                entry[field] = path.relative_to(args.data_root).as_posix() if path is not None else ""
            try:
                start = perf_counter()
                pair = prepare(sample)
                prepare_ms = (perf_counter() - start) * 1000
                height, width = pair.image.shape
                transform = pair.alignment
                entry.update(width=width, height=height, template_width=pair.template_size[0],
                             template_height=pair.template_size[1], alignment=transform.method,
                             dx=transform.dx, dy=transform.dy, angle_deg=transform.angle_deg,
                             ncc_before=transform.ncc_before, ncc_after=transform.ncc_after,
                             valid_fraction=float(pair.valid.mean()) if pair.valid is not None else 1.0)
                truths = read_truth(sample, (width, height))
            except SampleIssue as exc:
                stages[exc.stage] += 1
                manifest.writerow({**entry, "status": exc.stage, "reason": str(exc)})
            else:
                xs, ys = tile_starts(width, args.overlap), tile_starts(height, args.overlap)
                num_tiles = len(xs) * len(ys)
                labels = [obj.label for obj in truths]
                group = group_of(labels)
                groups[group] += 1
                class_counts.update(labels)
                stages["ok"] += 1
                manifest.writerow({**entry, "status": "ok", "group": group, "gt_boxes": len(truths),
                                   "tiles": num_tiles})
                preview = stages["ok"] <= args.previews
                basename = f"{number:04d}_{Path(sample.sample_id).stem}"
                if preview and not runners:
                    save_alignment(preview_dir / f"{basename}_alignment.png", pair)
                for runner in runners:
                    if stages["ok"] == 1:
                        warm_up(runner, pair)
                    try:
                        output = detect(runner, pair, xs, ys)
                    except RuntimeError as exc:
                        raise RuntimeError(f"{runner.kind}, {sample.sample_id} 추론 중 오류: {exc}") from exc
                    pairs = match(output.detections, truths, MATCH_IOU)
                    result = record_scores(tables, runner, sample, group, truths, output, pairs, prepare_ms)
                    histories[runner.kind].append(result)
                    if preview:
                        save_predictions(preview_dir / f"{basename}_{runner.kind}.png", pair, truths, output.detections)
            if number == 1 or number % 10 == 0 or number == len(samples):
                print(f"{number}/{len(samples)}장 | 입력 확인 {stages['ok']} | 제외 {number - stages['ok']}", flush=True)

    status_names = ("ok", "pair_error", "image_error", "annotation_error", "alignment_error")
    summary.update(accepted=stages["ok"], excluded=len(samples) - stages["ok"],
                   input_coverage=ratio(stages["ok"], len(samples)),
                   status_counts={name: stages[name] for name in status_names},
                   image_groups=dict(groups), ground_truth_counts={label: class_counts[label] for label in CLASSES})
    if runners:
        summary["models"] = {r.kind: {**r.info, **summarize(histories[r.kind])} for r in runners}
    return summary


def cli():
    parser = argparse.ArgumentParser(description="DeepPCB V2의 Kaggle 외부 이미지 쌍 검증")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("audit", "파일 짝, XML, 정렬 미리보기 확인"),
                            ("evaluate", "V2 추론과 원본 이미지 단위 채점")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--data-root", type=Path, default=ROOT / "out_data" / "PCB_DATASET",
                             help="images, PCB_USED, Annotations가 있는 폴더")
        command.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "v2", help="V2 가중치·보정 폴더")
        command.add_argument("--output-dir", type=Path, help="결과 상위 폴더. 기본값은 runs-dir/external_kaggle")
        command.add_argument("--limit", type=int, default=0, help="폴더별로 번갈아 고르는 총 이미지 수. 0이면 전체")
        command.add_argument("--overlap", type=int, default=128, help="인접 타일의 기본 겹침 픽셀 수")
        command.add_argument("--previews", type=int, default=6, help="미리보기를 저장할 유효 이미지 수")
        if name == "evaluate":
            command.add_argument("--model", choices=("both", "fast", "accurate"), default="both")
            command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.limit < 0 or args.previews < 0:
        parser.error("--limit와 --previews는 0 이상이어야 합니다.")
    if not 0 <= args.overlap < TILE:
        parser.error(f"--overlap은 0부터 {TILE - 1}까지 지정하세요.")
    return args


def main() -> int:
    args = cli()
    args.data_root = args.data_root.resolve()
    args.runs_dir = args.runs_dir.resolve()
    all_samples = discover(args.data_root)
    samples = all_samples[:args.limit] if args.limit else all_samples
    runners = load_runners(args) if args.command == "evaluate" else []
    output_root = (args.output_dir or args.runs_dir / "external_kaggle").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"{args.data_root} | 발견 {len(all_samples)}건, 이번 실행 {len(samples)}건", flush=True)
    # 부분 실행과 전체 실행의 결과를 각각 보존한다. 완성된 파일 묶음을 한 번에 공개한다.
    with TemporaryDirectory(prefix=".work_", dir=output_root) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        summary = run(args, samples, len(all_samples), runners, staging)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                          allow_nan=False), encoding="utf-8")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = output_root / f"{args.command}_{stamp}_{Path(temporary).name[6:]}"
        staging.rename(destination)
    print(f"결과 폴더: {destination}")
    for kind, model in summary.get("models", {}).items():
        for key, name in (("method_1_known", "공통 5종"), ("method_2_all", "전체 6종")):
            metrics = model[key]
            boxes = metrics["boxes"]
            board = metrics["image"]
            print(f"{kind} {name} | 불량 이미지 검출 {board['tp']}/{board['defective']} | "
                  f"결함 박스 검출 {boxes['tp']}/{boxes['gt_boxes']}")
    if not summary["accepted"]:
        print("평가 가능한 쌍이 없습니다. manifest.csv의 status와 reason을 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        sys.exit(f"오류: {exc}")
