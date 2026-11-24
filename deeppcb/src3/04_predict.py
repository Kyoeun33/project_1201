"""Classify one PCB image with a saved EfficientNet-B0 checkpoint."""
import argparse
import json
import sys
from importlib import import_module
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src2.common import SOURCES


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--source", choices=["auto", *SOURCES], default="auto")
    args = parser.parse_args()
    predict = import_module("src2.04_predict").predict
    print(json.dumps(predict(args.checkpoint, args.image, args.device, args.source), indent=2))
