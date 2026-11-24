"""Validate the attached DeepPCB layout and create the shared 800/200/500 split."""

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset import CLASS_NAMES, read_official_list, save_previews, split_trainval, validate_records, write_split_csv


def main() -> None:
    # Resolve defaults from the project root so the script works when launched
    # either from deeppcb/ or directly from deeppcb/src/.
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--split-csv", type=Path, default=project_root / "runs/splits.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-count", type=int, default=200)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--expected-trainval", type=int, default=1000)
    parser.add_argument("--expected-test", type=int, default=500)
    args = parser.parse_args()

    root = args.data_root.resolve()
    trainval = read_official_list(root, "trainval.txt")
    test = read_official_list(root, "test.txt")
    if len(trainval) != args.expected_trainval or len(test) != args.expected_test:
        raise ValueError(f"공식 분할 크기 불일치: trainval={len(trainval)}, test={len(test)}")
    for record in trainval:
        record["split"] = "trainval"
    for record in test:
        record["split"] = "test"
    validate_records(root, trainval + test)  # also checks duplicate IDs across both lists
    split_trainval(trainval, seed=args.seed, val_count=args.val_count)
    all_records = trainval + test
    counts = {split: Counter(cls for rec in all_records if rec["split"] == split for cls in rec["_classes"])
              for split in ("train", "val", "test")}
    write_split_csv(all_records, args.split_csv)
    save_previews(root, [rec for rec in all_records if rec["split"] == "val"],
                  args.split_csv.parent / "previews", args.preview_count)
    report = {
        "data_root": str(root), "seed": args.seed,
        "image_count": {s: sum(r["split"] == s for r in all_records) for s in counts},
        "object_count": {s: {str(c): counts[s][c] for c in CLASS_NAMES} for s in counts},
        "classes": CLASS_NAMES,
    }
    report_path = args.split_csv.parent / "data_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"공식 분할 검증 완료: {report['image_count']}; CSV: {args.split_csv}; 미리보기: {args.split_csv.parent / 'previews'}")


if __name__ == "__main__":
    main()
