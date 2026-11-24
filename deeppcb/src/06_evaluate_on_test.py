"""Evaluate held-out test with fixed val thresholds and benchmark identical pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from dataset import read_split_csv
from evaluation import (benchmark, file_sha256, filter_predictions, infer_records,
                        mean_average_precision, operating_metrics, write_predictions,
                        evaluate_loss)
from training import load_checkpoint, load_config, select_device, set_seed


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-config", type=Path, default=project_root / "configs" / "fast.yaml")
    parser.add_argument("--accurate-config", type=Path, default=project_root / "configs" / "accurate.yaml")
    parser.add_argument("--limit-test", type=int, default=0, help="연결 점검용; 전체 평가는 0")
    parser.add_argument("--benchmark-pairs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args()

    configs = {"fast": load_config(args.fast_config), "accurate": load_config(args.accurate_config)}
    fast_cfg, accurate_cfg = configs.values()
    split_path = Path(fast_cfg["split_csv"])
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if (hashlib.sha256(Path(accurate_cfg["split_csv"]).read_bytes()).hexdigest() != split_hash
            or Path(fast_cfg["data_root"]).resolve() != Path(accurate_cfg["data_root"]).resolve()):
        raise ValueError("두 모델의 데이터 루트와 분할 CSV가 동일해야 함")
    runs = Path(fast_cfg["runs_dir"])
    calibration = json.loads((runs / "calibration.json").read_text(encoding="utf-8"))
    if calibration["split_sha256"] != split_hash:
        raise ValueError("val 보정에 사용한 분할 CSV가 변경됨; 05를 다시 실행하세요")
    if calibration["partial_val"] and not args.limit_test:
        raise ValueError("일부 val로 선택한 임계값으로 전체 test 평가 불가; 05를 전체 val에서 실행하세요")
    all_records = read_split_csv(split_path, "test")
    records = all_records[:args.limit_test] if args.limit_test else all_records
    if not records:
        raise ValueError("test 분할이 비어 있음")
    set_seed(int(fast_cfg["seed"]))
    results = {}
    devices = set()
    for kind, source_cfg in configs.items():
        checkpoint_path = Path(source_cfg["runs_dir"]) / f"{kind}.pt"
        if calibration["models"][kind]["checkpoint_sha256"] != file_sha256(checkpoint_path):
            raise ValueError(f"{kind} 모델 체크포인트가 val 보정 후 변경됨; 05를 다시 실행하세요")
        device = select_device(args.device or source_cfg["device"])
        devices.add(str(device))
        if len(devices) > 1:
            raise ValueError("두 모델의 속도 비교는 같은 CPU 또는 CUDA 장치로 실행해야 함")
        model, saved_cfg = load_checkpoint(checkpoint_path, kind, device)
        checkpoint_info = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        raw_predictions, truths = infer_records(kind, model, saved_cfg, device, records)
        threshold = float(calibration["models"][kind]["threshold"])
        final_predictions = filter_predictions(raw_predictions, threshold)
        write_predictions(runs / f"test_raw_{kind}.csv", raw_predictions)
        write_predictions(runs / f"test_predictions_{kind}.csv", final_predictions)
        operating = operating_metrics(raw_predictions, truths, threshold=threshold)
        test_loss = evaluate_loss(kind, model, saved_cfg, device, records)
        ap = mean_average_precision(raw_predictions, truths)
        timing = benchmark(kind, model, saved_cfg, device, records,
                           warmup=args.warmup, sample_count=args.benchmark_pairs)
        results[kind] = {"operating": operating, "loss": test_loss,
                         "overfitting": checkpoint_info.get("overfitting", {
                             "detected": None, "reason": "training history unavailable; retrain to generate this result"}),
                         "training_history": checkpoint_info.get("training_history"),
                         "ranking": ap, "latency": timing,
                         "raw_predictions": len(raw_predictions),
                         "final_predictions": len(final_predictions)}
        print(f"{kind}: accuracy={operating['accuracy']:.3f}, precision={operating['precision']:.3f}, recall={operating['recall']:.3f}, loss={test_loss:.4f}, overfitting={results[kind]['overfitting']['detected']}, F1={operating['f1']:.3f}, mAP@0.33={ap['mAP_0_33']:.3f}, median={timing['median_ms']:.1f}ms")
    report = {"test_images": len(records), "partial_test": len(records) != len(all_records),
              "val_images_used_for_threshold": calibration["val_images"],
              "partial_val_calibration": calibration["partial_val"],
              "val_max_fp_per_image": calibration["max_fp_per_image"],
              "shared_split_sha256": split_hash, "models": results}
    out = runs / "test_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test 비교 보고서: {out}")


if __name__ == "__main__":
    main()
