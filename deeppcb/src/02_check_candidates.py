"""Check XOR patch coverage and output-box localization on validation images."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from candidates import box_iou, find_candidates
from dataset import CLASS_NAMES, load_pair, read_split_csv


def _fraction(covered: int, count: int) -> float | None:
    return covered / count if count else None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=project_root / "configs" / "fast.yaml")
    parser.add_argument("--limit", type=int, default=0, help="진단용 이미지 수, 0이면 val 전체")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    # Config paths are project-relative, regardless of the shell's cwd.
    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    split_path = project_path(cfg["split_csv"])
    records = read_split_csv(split_path, "val")
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError("val 이미지가 없음; 먼저 01을 실행하세요")
    buckets = {cls: {"count": 0, "patch_enclosed": 0, "output_iou_gt_0_33": 0}
               for cls in (0, *CLASS_NAMES)}
    candidate_counts = []
    preview_dir = project_path(cfg["runs_dir"]) / "candidate_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        temp, test, boxes, classes = load_pair(project_path(cfg["data_root"]), record)
        candidates, _, _ = find_candidates(temp, test, cfg["candidate"])
        candidate_counts.append(len(candidates))
        for gt, cls in zip(boxes, classes):
            for key in (0, int(cls)):
                buckets[key]["count"] += 1
                buckets[key]["patch_enclosed"] += int(any(
                    c.patch_box[0] <= gt[0] and c.patch_box[1] <= gt[1]
                    and c.patch_box[2] >= gt[2] and c.patch_box[3] >= gt[3]
                    for c in candidates))
                buckets[key]["output_iou_gt_0_33"] += int(any(
                    box_iou(c.output_box, gt) > 0.33 for c in candidates))
        if index < 3:
            image = Image.fromarray(test).convert("RGB")
            draw = ImageDraw.Draw(image)
            for candidate in candidates:
                draw.rectangle(candidate.output_box, outline="#00e5ff", width=1)
            for gt in boxes:
                draw.rectangle(tuple(map(int, gt)), outline="#f44336", width=2)
            image.save(preview_dir / f"{record['sample_id']}.png")
    coverage = {}
    for cls, values in buckets.items():
        coverage["overall" if cls == 0 else CLASS_NAMES[cls]] = {
            "count": values["count"],
            "patch_enclosed_rate": _fraction(values["patch_enclosed"], values["count"]),
            "output_iou_gt_0_33_rate": _fraction(values["output_iou_gt_0_33"], values["count"]),
        }
    report = {
        "evaluated_images": len(records), "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "candidate_config": cfg["candidate"], "coverage": coverage,
        "candidates_per_image": {"mean": float(np.mean(candidate_counts)),
                                 "p95": float(np.percentile(candidate_counts, 95))},
    }
    out = project_path(cfg["runs_dir"]) / "candidate_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"val 후보 실험 {len(records)}장 완료: {out}")
    print(json.dumps({"coverage": coverage["overall"], "candidates_per_image": report["candidates_per_image"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
