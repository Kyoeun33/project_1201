"""VisA PCB1~PCB4로 EfficientNet-B0를 학습하고 원본 단위 시험 결과를 저장한다."""
import sys
from importlib import import_module
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src3.settings import load_config


def main():
    cfg = load_config()
    cfg["sources"] = ["VisA"]
    cfg["paths"] = {"VisA": cfg["paths"]["VisA"]}
    cfg["output_dir"] = str(Path(__file__).resolve().parent / "artifacts_visa")

    import_module("src2.01_prepare").prepare(cfg)
    trainer = import_module("src2.02_train")
    # 공통 학습기는 실험 이름으로 학습 출처를 조회하고 같은 이름을 체크포인트에 기록한다.
    standard_experiments = trainer.EXPERIMENTS
    try:
        trainer.EXPERIMENTS = {**standard_experiments, "VisA": ("VisA",)}
        trainer.train(cfg, ["VisA"])
    finally:
        trainer.EXPERIMENTS = standard_experiments

    import_module("src2.03_evaluate").evaluate(cfg, ["VisA"])


if __name__ == "__main__":
    main()
