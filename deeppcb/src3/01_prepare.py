"""Prepare the original-image splits and labeled tiles for EfficientNet-B0."""
import sys
from importlib import import_module
from pathlib import Path

if not __package__:  # Also support running this file from an IDE.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src3.settings import arguments, load_config
from src2.common import read_json


if __name__ == "__main__":
    cfg = load_config(arguments(__doc__).config)
    report = import_module("src2.01_prepare").prepare(cfg)
    reference = Path(cfg["resnet_output_dir"]) / "prepared" / "summary.json"
    if reference.is_file():
        baseline = read_json(reference)
        if baseline["manifest_sha256"] == report["manifest_sha256"]:
            print("ResNet18 and EfficientNet-B0 have identical original and tile manifests.")
        else:
            print("Different manifests from ResNet18. A matched comparison requires the same split and tiles.")
