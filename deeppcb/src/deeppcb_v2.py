"""DeepPCB V2 이진 불량 검출. 기존 프로젝트의 src/에 이 파일을 넣어 실행한다.

프로젝트 루트에서 다음 명령으로 실행한다.
    python src/deeppcb_v2.py train fast
    python src/deeppcb_v2.py train accurate
    python src/deeppcb_v2.py calibrate fast
    python src/deeppcb_v2.py calibrate accurate
    python src/deeppcb_v2.py predict fast --template normal.jpg --image inspected.jpg

학습과 임계값 보정은 V1의 runs/splits.csv를 사용한다. 이 CSV가 없다면
V1의 src/01_prepare_data.py로 생성한다. 정답 TXT의 1~6번 결함 유형은
그대로 두고, DataLoader에서만 불량(1)으로 통합한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

from candidates import box_iou, find_candidates, patch_tensor
from dataset import IMAGE_SIZE, load_pair, read_split_csv
from evaluation import file_sha256
from models import FastPatchCNN, make_accurate_model
from training import (FastCandidateDataset, PairDataset, collate_fast, collate_pairs,
                      load_config, select_device, set_seed)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs" / "v2"
MATCH_IOU = 0.33  # V1의 박스 평가 기준


class BinaryFastDataset(FastCandidateDataset):
    def __getitem__(self, index: int):
        patches, labels = super().__getitem__(index)
        return patches, labels.gt(0).long()


class BinaryPairDataset(PairDataset):
    def __getitem__(self, index: int):
        pair, target = super().__getitem__(index)
        # 박스 좌표와 정상 샘플의 빈 정답 텐서를 유지한다.
        target["labels"] = target["labels"].gt(0).long()
        return pair, target


def make_v2_model(kind: str, config: dict) -> nn.Module:
    if kind == "fast":
        model = FastPatchCNN()
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        return model
    if kind == "accurate":
        model = make_accurate_model(config)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        return model
    raise ValueError(f"알 수 없는 모델: {kind}")


def training_config(args: argparse.Namespace) -> dict:
    config_path = args.config or PROJECT_ROOT / "configs" / f"{args.kind}.yaml"
    if args.config is not None or config_path.is_file():
        config = load_config(config_path)
    else:
        # src.zip만 풀어 둔 경우에도 기존 DeepPCB.zip의 기본 설정으로 실행한다.
        config = {
            "data_root": PROJECT_ROOT / "data",
            "split_csv": PROJECT_ROOT / "runs" / "splits.csv",
            "seed": 42,
            "device": "auto",
            "model": {"nms_iou": 0.50},
            "training": {"epochs": 10, "augmentation": False},
        }
        if args.kind == "fast":
            config["model"]["patch_size"] = 96
            config["candidate"] = {
                "binary_threshold": 128, "min_area": 16, "output_margin": 8,
                "context_margin": 12, "copper_polarity": "dark",
            }
            config["training"].update(
                batch_images=2, learning_rate=0.001, max_background_per_image=24,
            )
        else:
            config["model"].update(
                fpn_channels=128, anchors=[[8, 16], [16, 32], [32, 64], [64, 128], [128, 256]],
            )
            config["training"].update(batch_images=1, learning_rate=0.0001)

    config["data_root"] = str(Path(args.data_root or config["data_root"]).resolve())
    config["split_csv"] = str(Path(args.split_csv or config["split_csv"]).resolve())
    config["runs_dir"] = str(args.runs_dir.resolve())
    config["device"] = args.device or config["device"]
    settings = config["training"]
    settings["epochs"] = args.epochs or int(settings["epochs"])
    settings["batch_images"] = args.batch_images or int(settings["batch_images"])
    settings["early_stopping_patience"] = int(settings.get("early_stopping_patience", 7))
    if min(settings[option] for option in ("epochs", "batch_images", "early_stopping_patience")) < 1:
        raise ValueError("epochs, batch_images, early_stopping_patience는 1 이상이어야 합니다.")
    if not Path(config["split_csv"]).is_file():
        raise FileNotFoundError(
            f"공유 분할 CSV가 없습니다: {config['split_csv']}. "
            "src/01_prepare_data.py를 실행하거나 --split-csv로 V1에서 사용한 CSV를 지정하세요."
        )
    return config


def run_epoch(model, loader, device, *, criterion=None, optimizer=None) -> float:
    learning = optimizer is not None
    is_fast = criterion is not None
    model.train(learning or not is_fast)
    if not learning and not is_fast:
        # Faster R-CNN은 train 모드에서 손실을 반환한다. val에서 BN 통계는 고정한다.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    total_loss, total_items = 0.0, 0
    with torch.set_grad_enabled(learning):
        for inputs, targets in loader:
            if is_fast and len(targets) == 0:
                continue
            if learning:
                optimizer.zero_grad(set_to_none=True)
            if is_fast:
                loss = criterion(model(inputs.to(device)), targets.to(device))
                count = len(targets)  # V1과 같은 패치 수 기준 집계
            else:
                images = [image.to(device) for image in inputs]
                annotations = [{key: value.to(device) for key, value in target.items()}
                               for target in targets]
                loss = sum(model(images, annotations).values())
                count = 1  # V1과 같은 배치 기준 집계

            value = float(loss.detach())
            if not math.isfinite(value):
                raise ValueError("학습 또는 검증 손실이 유한하지 않습니다.")
            if learning:
                loss.backward()
                optimizer.step()
            total_loss += value * count
            total_items += count

    if total_items == 0:
        split = "train" if learning else "val"
        raise ValueError(f"{split}에 분류할 후보 패치가 없습니다. XOR 후보 설정과 이미지 쌍을 확인하세요.")
    return total_loss / total_items


def write_atomically(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / path.name
        write(temporary_path)
        temporary_path.replace(path)


def train(args: argparse.Namespace) -> None:
    config = training_config(args)
    settings = config["training"]
    split_path = Path(config["split_csv"])
    split_hash = file_sha256(split_path)
    set_seed(int(config["seed"]))
    device = select_device(config["device"])
    is_fast = args.kind == "fast"
    dataset_type, collate = (BinaryFastDataset, collate_fast) if is_fast else (BinaryPairDataset, collate_pairs)
    train_data = dataset_type(config, "train", augment=settings["augmentation"], limit=args.limit_train)
    val_data = dataset_type(config, "val", limit=args.limit_val)
    loader_settings = dict(batch_size=settings["batch_images"], collate_fn=collate, num_workers=0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_settings)
    val_loader = DataLoader(val_data, shuffle=False, **loader_settings)

    model = make_v2_model(args.kind, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0], device=device)) if is_fast else None
    print(f"V2-{args.kind}: train {len(train_data)}쌍, val {len(val_data)}쌍, device={device}", flush=True)

    checkpoint_path = args.runs_dir / f"{args.kind}_v2.pt"
    best_loss, stale_epochs = float("inf"), 0
    for epoch in range(1, settings["epochs"] + 1):
        train_loss = run_epoch(model, train_loader, device, criterion=criterion, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion=criterion)
        print(f"epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss:
            if file_sha256(split_path) != split_hash:
                raise ValueError("학습 도중 분할 CSV가 변경되었습니다.")
            best_loss, stale_epochs = val_loss, 0
            checkpoint = {
                "version": 2, "kind": args.kind, "num_classes": 2, "config": config,
                "model_state": model.state_dict(), "split_sha256": split_hash,
                "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                "train_pairs": len(train_data), "val_pairs": len(val_data),
            }
            write_atomically(checkpoint_path, lambda path: torch.save(checkpoint, path))
        else:
            stale_epochs += 1
            if stale_epochs >= settings["early_stopping_patience"]:
                print(f"조기 종료: {stale_epochs} epoch 동안 검증 손실 개선 없음")
                break
    print(f"최저 val_loss 체크포인트: {checkpoint_path.resolve()}")


def restore_model(args: argparse.Namespace):
    set_seed(42)  # V1과 같은 CPU 스레드 설정
    device = select_device(args.device or "auto")
    path = args.runs_dir / f"{args.kind}_v2.pt"
    # 같은 파일 핸들에서 가중치와 해시를 읽어 임계값을 그 가중치에 연결한다.
    with path.open("rb") as handle:
        checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        handle.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    if (checkpoint.get("version"), checkpoint.get("kind"), checkpoint.get("num_classes")) != (2, args.kind, 2):
        raise ValueError(f"V2-{args.kind} 체크포인트가 아닙니다: {path}")
    model = make_v2_model(args.kind, checkpoint["config"])
    model.load_state_dict(checkpoint.pop("model_state"))
    return model.to(device).eval(), checkpoint, digest.hexdigest(), device


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int = 1


@torch.inference_mode()
def predict_boxes(kind, model, config, temp, test, device) -> list[Detection]:
    expected_shape = (IMAGE_SIZE[1], IMAGE_SIZE[0])
    if temp.shape != expected_shape or test.shape != expected_shape:
        raise ValueError(f"정렬된 {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} 이미지 쌍이 필요합니다.")

    if kind == "fast":
        candidates, added, missing = find_candidates(temp, test, config["candidate"])
        if not candidates:
            return []
        patch_size = int(config["model"]["patch_size"])
        score_chunks = []
        for start in range(0, len(candidates), 128):
            crops = np.stack([patch_tensor(temp, test, added, missing, item.patch_box, patch_size)
                              for item in candidates[start:start + 128]])
            logits = model(torch.from_numpy(crops).to(device))
            score_chunks.append(logits.softmax(dim=1)[:, 1].cpu())
        scores = torch.cat(score_chunks)
        boxes = torch.tensor([item.output_box for item in candidates], dtype=torch.float32)
        keep = nms(boxes, scores, float(config["model"]["nms_iou"]))
        boxes, scores = boxes[keep], scores[keep]
    else:
        pair = torch.from_numpy(np.stack((test, temp))).float().div_(255).to(device)
        # Faster R-CNN 내부에서 단일 불량 클래스의 NMS를 수행한다.
        prediction = model([pair])[0]
        boxes, scores = prediction["boxes"].cpu(), prediction["scores"].cpu()

    return sorted((Detection(tuple(box), score) for box, score in zip(boxes.tolist(), scores.tolist())),
                  key=lambda detection: detection.score, reverse=True)


def match_detections(detections: list[Detection], ground_truth: np.ndarray) -> list[bool]:
    # 점수 내림차순으로 정답 박스 하나당 최대 한 예측을 매칭한다.
    remaining = set(range(len(ground_truth)))
    hits = []
    for detection in detections:
        iou, index = max(((box_iou(detection.box, ground_truth[i]), i) for i in remaining),
                         default=(0.0, -1))
        hits.append(iou >= MATCH_IOU)
        if hits[-1]:
            remaining.remove(index)
    return hits


def select_threshold(validation: list[tuple[list[Detection], np.ndarray]], max_fpr: float,
                     max_fp_per_image: float) -> dict:
    is_defective = np.array([len(boxes) > 0 for _, boxes in validation], dtype=bool)
    defects = int(is_defective.sum())
    normals = len(validation) - defects
    if defects == 0:
        raise ValueError("val에 불량 PCB가 없어 임계값을 보정할 수 없습니다.")

    pair_scores = np.array([items[0].score if items else -np.inf for items, _ in validation])
    loc_scores = np.array([item.score for items, _ in validation for item in items])
    loc_hits = np.array([hit for items, boxes in validation for hit in match_detections(items, boxes)], dtype=bool)
    gt_count = sum(len(boxes) for _, boxes in validation)

    # 1보다 큰 후보는 점수가 1.0인 예측까지 제외할 수 있게 한다.
    best_key = None
    best = {}
    for threshold in [*(i / 100 for i in range(101)), 1.000001]:
        flagged = pair_scores >= threshold
        tp = int((flagged & is_defective).sum())
        fp = int((flagged & ~is_defective).sum())
        fpr = fp / normals if normals else None
        kept = loc_scores >= threshold
        loc_tp = int((kept & loc_hits).sum())
        loc_fp = int(kept.sum()) - loc_tp
        loc_fp_per_image = loc_fp / len(validation)
        if normals:
            if fpr > max_fpr:
                continue
            key = (tp, -fp, loc_tp, -loc_fp, threshold)
        else:
            # 정상 PCB가 없는 검증셋에서는 위치 오탐 상한으로 보정한다.
            if loc_fp_per_image > max_fp_per_image:
                continue
            key = (loc_tp, -loc_fp, threshold)

        if best_key is None or key > best_key:
            fn = defects - tp
            best_key = key
            best = {
                "threshold": threshold,
                "selection": "max_pcb_recall_under_fpr_cap" if normals else "max_box_recall_under_fp_per_image_cap",
                "pcb": {"defective_pairs": defects, "normal_pairs": normals,
                        "tp": tp, "fp": fp, "fn": fn, "tn": normals - fp,
                        "recall": tp / defects, "fnr": fn / defects, "fpr": fpr,
                        "precision": tp / (tp + fp) if normals and tp + fp else None,
                        "f1": 2 * tp / (2 * tp + fp + fn) if normals else None},
                "localization": {"iou": MATCH_IOU, "gt_boxes": gt_count, "tp": loc_tp,
                                 "fp": loc_fp, "recall": loc_tp / gt_count,
                                 "fp_per_image": loc_fp_per_image},
            }
    return best


def calibrate(args: argparse.Namespace) -> None:
    model, checkpoint, checkpoint_hash, device = restore_model(args)
    config = checkpoint["config"]
    split_path = args.split_csv or Path(config["split_csv"])
    if file_sha256(split_path) != checkpoint["split_sha256"]:
        raise ValueError("학습에 사용한 분할 CSV와 보정용 CSV가 다릅니다.")
    data_root = args.data_root or Path(config["data_root"])
    all_records = read_split_csv(split_path, "val")
    records = all_records[:args.limit_val] if args.limit_val else all_records
    validation = []
    for index, record in enumerate(records, 1):
        temp, test, boxes, _ = load_pair(data_root, record)
        validation.append((predict_boxes(args.kind, model, config, temp, test, device), boxes))
        if index % 50 == 0:
            print(f"val 추론 {index}/{len(records)}", flush=True)

    result = select_threshold(validation, args.max_fpr, args.max_fp_per_image)
    if file_sha256(split_path) != checkpoint["split_sha256"]:
        raise ValueError("임계값 보정 도중 분할 CSV가 변경되었습니다.")
    report = {**result, "kind": args.kind, "checkpoint_sha256": checkpoint_hash,
              "split_sha256": checkpoint["split_sha256"], "val_pairs": len(records),
              "partial_val": len(records) != len(all_records), "max_fpr": args.max_fpr,
              "max_fp_per_image": args.max_fp_per_image}
    path = args.runs_dir / f"{args.kind}_v2_calibration.json"
    write_atomically(path, lambda file: file.write_text(json.dumps(report, indent=2), encoding="utf-8"))
    print(f"val 임계값 {result['threshold']:.8g}, PCB Recall {result['pcb']['recall']:.2%}")
    print(f"보정 결과 저장 경로: {path.resolve()}")
    if report["pcb"]["normal_pairs"] == 0:
        print("val에 별도 정상 샘플이 없어 박스 오탐 수/이미지를 기준으로 보정했습니다.")


def read_gray_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def predict(args: argparse.Namespace) -> None:
    model, checkpoint, checkpoint_hash, device = restore_model(args)
    if args.threshold is None:
        calibration_path = args.runs_dir / f"{args.kind}_v2_calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError("먼저 calibrate를 실행하세요. 연결 점검에는 --threshold를 지정할 수 있습니다.")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("체크포인트가 갱신되었습니다. calibrate를 다시 실행하세요.")
        threshold = float(calibration["threshold"])
        threshold_source = "partial_val" if calibration["partial_val"] else "val"
    else:
        threshold, threshold_source = args.threshold, "manual"
    detections = predict_boxes(args.kind, model, checkpoint["config"], read_gray_image(args.template),
                               read_gray_image(args.image), device)
    kept = [item for item in detections if item.score >= threshold]
    result = {
        "model": f"V2-{args.kind}", "decision": "FAIL" if kept else "PASS",
        "defect_score": detections[0].score if detections else None,
        "threshold": threshold, "threshold_source": threshold_source,
        "detections": [asdict(item) for item in kept],
    }
    print(json.dumps(result, indent=2))


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1 이상의 정수를 입력하세요.")
    return number


def sample_limit(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수를 입력하세요.")
    return number


def nonnegative_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0 이상의 유한한 수를 입력하세요.")
    return number


def probability(value: str) -> float:
    number = nonnegative_number(value)
    if number > 1:
        raise argparse.ArgumentTypeError("0~1 사이의 수를 입력하세요.")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepPCB V2: 정상/불량 학습, val 보정, 단일 이미지 쌍 검사")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, action in (("train", train), ("calibrate", calibrate), ("predict", predict)):
        command = commands.add_parser(name)
        command.set_defaults(action=action)
        command.add_argument("kind", choices=("fast", "accurate"))
        command.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR, help="V2 결과 저장 폴더")
        command.add_argument("--device", choices=("auto", "cpu", "cuda"))
        if name != "predict":
            command.add_argument("--data-root", type=Path)
            command.add_argument("--split-csv", type=Path)
            command.add_argument("--limit-val", type=sample_limit, default=0, help="점검용 val 쌍 수. 0이면 전체")
        if name == "train":
            command.add_argument("--config", type=Path, help="해당 모델의 V1 YAML 설정")
            command.add_argument("--epochs", type=positive_integer)
            command.add_argument("--batch-images", type=positive_integer)
            command.add_argument("--limit-train", type=sample_limit, default=0, help="점검용 train 쌍 수. 0이면 전체")
        elif name == "calibrate":
            command.add_argument("--max-fpr", type=probability, default=0.05, help="정상 PCB 오탐률 상한")
            command.add_argument("--max-fp-per-image", type=nonnegative_number, default=1.0,
                                 help="정상 PCB가 없을 때 사용하는 이미지당 오탐 박스 수 상한")
        else:
            command.add_argument("--template", type=Path, required=True, help="정상 템플릿 이미지")
            command.add_argument("--image", type=Path, required=True, help="검사 이미지")
            command.add_argument("--threshold", type=probability, help="점검용 수동 임계값")
    args = parser.parse_args()
    try:
        args.action(args)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
