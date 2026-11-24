"""Train EfficientNet-B0 on A, B and C, and select each checkpoint on validation."""
import sys
from importlib import import_module
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src3.settings import arguments, load_config


if __name__ == "__main__":
    args = arguments(__doc__, experiments=True)
    cfg = load_config(args.config)
    if not (Path(cfg["output_dir"]) / "prepared" / "summary.json").is_file():
        raise SystemExit(f'First run python -m scr3.01_prepare --config "{args.config}"')
    import_module("src2.02_train").train(cfg, args.experiments)
