"""VisA PCB1~PCB4로 EfficientNet-B0를 학습하고 원본 단위 시험 결과를 저장한다."""
import sys
from importlib import import_module
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src3.settings import load_config


def fit_with_early_stopping(trainer, patience):
    best_loss = float("inf")
    stale_epochs = 0

    def epoch_range(start, stop):
        for epoch in range(start, stop):
            if stale_epochs >= patience:
                print(f"VisA early stopping at epoch {epoch - 1}: validation image loss "
                      f"did not improve for {patience} epochs", flush=True)
                break
            yield epoch

    def score_validation_images(originals, tiles, probabilities):
        nonlocal best_loss, stale_epochs
        predictions = trainer.image_scores(originals, tiles, probabilities)
        loss = trainer.log_loss(predictions.label, predictions.prob_defect)
        if loss < best_loss:
            best_loss, stale_epochs = loss, 0
        else:
            stale_epochs += 1
        return predictions

    # Use VisA's epoch iterator and validation scoring inside the shared fit function.
    # Exhausting the iterator lets fit save its best checkpoint and validation results.
    fit = trainer.fit
    fit_globals = dict(fit.__globals__, range=epoch_range, image_scores=score_validation_images)
    return FunctionType(fit.__code__, fit_globals, fit.__name__)


def main():
    cfg = load_config()
    cfg["sources"] = ["VisA"]
    cfg["paths"] = {"VisA": cfg["paths"]["VisA"]}
    cfg["training"].update(epochs=25, early_stopping_patience=7)
    cfg["output_dir"] = str(Path(__file__).resolve().parent / "artifacts_visa_25epochs")

    import_module("src2.01_prepare").prepare(cfg)
    trainer = import_module("src2.02_train")
    experiments = {**trainer.EXPERIMENTS, "VisA": ("VisA",)}
    with patch.object(trainer, "EXPERIMENTS", experiments):
        fit = fit_with_early_stopping(trainer, cfg["training"]["early_stopping_patience"])
        with patch.object(trainer, "fit", fit):
            trainer.train(cfg, ["VisA"])

    import_module("src2.03_evaluate").evaluate(cfg, ["VisA"])


if __name__ == "__main__":
    main()
