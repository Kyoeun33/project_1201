"""Choose operating thresholds under the SAME val FP/image cap for both models."""

import argparse
import hashlib
import json
from pathlib import Path

from dataset import read_split_csv
from evaluation import calibrate_threshold, file_sha256, infer_records, write_predictions
from training import load_checkpoint, load_config, select_device, set_seed


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-config", type=Path, default=project_root / "configs" / "fast.yaml")
    parser.add_argument("--accurate-config", type=Path, default=project_root / "configs" / "accurate.yaml")
    parser.add_argument("--max-fp-per-image", type=float, default=1.0)
    parser.add_argument("--limit-val", type=int, default=0, help="연결 점검용; 전체 보정은 0")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    args = parser.parse_args()

    configs = {"fast": load_config(args.fast_config), "accurate": load_config(args.accurate_config)}
    fast_cfg, accurate_cfg = configs.values()
    split_path = Path(fast_cfg["split_csv"])
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    if (hashlib.sha256(Path(accurate_cfg["split_csv"]).read_bytes()).hexdigest() != split_hash
            or Path(fast_cfg["data_root"]).resolve() != Path(accurate_cfg["data_root"]).resolve()):
        raise ValueError("두 모델의 데이터 루트와 분할 CSV가 동일해야 함")
    all_records = read_split_csv(split_path, "val")
    records = all_records[:args.limit_val] if args.limit_val else all_records
    if not records:
        raise ValueError("val 분할이 비어 있음")
    set_seed(int(fast_cfg["seed"]))
    runs = Path(fast_cfg["runs_dir"])
    results = {}
    for kind, source_cfg in configs.items():
        checkpoint_path = Path(source_cfg["runs_dir"]) / f"{kind}.pt"
        device = select_device(args.device or source_cfg["device"])
        model, saved_cfg = load_checkpoint(checkpoint_path, kind, device)
        predictions, truths = infer_records(kind, model, saved_cfg, device, records)
        metrics = calibrate_threshold(predictions, truths, args.max_fp_per_image)
        write_predictions(runs / f"val_raw_{kind}.csv", predictions)
        results[kind] = {"checkpoint_sha256": file_sha256(checkpoint_path),
                         "threshold": metrics["score_threshold"], "val_metrics": metrics,
                         "raw_predictions": len(predictions)}
        print(f"{kind}: threshold={metrics['score_threshold']:.2f}, val FP/image={metrics['false_positives_per_image']:.3f}, F1={metrics['f1']:.3f}")
    report = {"max_fp_per_image": args.max_fp_per_image, "val_images": len(records),
              "partial_val": len(records) != len(all_records), "split_sha256": split_hash,
              "selection": "0.00..1.00 in steps of 0.01 plus 1.000001; max F1 subject to FP/image cap",
              "models": results}
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / "calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"val 보정 결과: {out}")


if __name__ == "__main__":
    main()
