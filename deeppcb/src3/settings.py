"""Read src3 paths relative to its config, including the ResNet18 comparison run."""
import argparse
from pathlib import Path

from src2.common import EXPERIMENTS, config

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def load_config(path=DEFAULT_CONFIG):
    cfg = config(path)
    reference = (Path(path).resolve().parent / cfg["resnet_output_dir"]).resolve()
    if cfg["training"]["model"] != "efficientnet_b0":
        raise ValueError("src3 expects training.model = efficientnet_b0")
    if reference == Path(cfg["output_dir"]):
        raise ValueError("Choose distinct output directories for EfficientNet-B0 and ResNet18.")
    cfg["resnet_output_dir"] = str(reference)
    return cfg


def arguments(description, *, experiments=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if experiments:
        parser.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    return parser.parse_args()
