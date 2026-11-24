"""Evaluate fixed EfficientNet-B0 checkpoints and compare to saved ResNet18 results."""
import sys
from importlib import import_module
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src3.comparison import save_comparison
from src3.settings import arguments, load_config


if __name__ == "__main__":
    args = arguments(__doc__, experiments=True)
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    missing = [exp for exp in args.experiments if not (root / "runs" / exp / "run.json").is_file()]
    if missing:
        raise SystemExit(f'First run python -m scr3.02_train --config "{args.config}" '
                         f'--experiments {" ".join(missing)}')
    import_module("src2.03_evaluate").evaluate(cfg, args.experiments)
    save_comparison(cfg)
