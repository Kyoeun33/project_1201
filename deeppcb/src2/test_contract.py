"""Regression checks for partial defects, final-image pooling and split isolation."""
import unittest
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .metrics import choose_threshold, image_scores, metrics
prepare = import_module(".01_prepare", __package__)
assign_splits, audit = prepare.assign_splits, prepare.audit
from .tiles import Region, tile_target, windows


class PCBContract(unittest.TestCase):
    def test_window_coverage_and_short_edge(self):
        cfg = dict(size=640, overlap=128)
        rects = list(windows(1410, 410, "PKU", cfg))
        self.assertEqual(rects, [(0, 0, 640, 410), (512, 0, 1152, 410), (770, 0, 1410, 410)])
        self.assertEqual(list(windows(640, 640, "DeepPCB", cfg)), [(0, 0, 640, 640)])

    def test_threshold_equality_and_sparse_component(self):
        region = Region((2, 2, 6, 4), 4, np.array([[1, 0, 0, 1], [1, 0, 0, 1]], bool))
        self.assertEqual(tile_target((3, 2, 5, 4), [region], 0.5)[0], 0)
        self.assertEqual(tile_target((1, 1, 3, 3), [region], 0.5), (-1, 0.25, 1, 4))
        self.assertEqual(tile_target((1, 1, 3, 4), [region], 0.5), (1, 0.5, 2, 4))
        small_box = Region((1, 1, 2, 2), 1)
        self.assertEqual(tile_target((1, 1, 3, 3), [region, small_box], 0.5), (1, 1, 1, 1))

    def test_group_split_and_parent_audit(self):
        records = [SimpleNamespace(source="PKU", group_id=f"PKU/{i}", board=str(i), label=label)
                   for i in range(7) for label in (0, 1)]
        assign_splits(records, [0.7, 0.15, 0.15], 42)
        result = [x.split for x in records]
        assign_splits(list(reversed(records)), [0.7, 0.15, 0.15], 42)
        self.assertEqual(result, [x.split for x in records])
        self.assertEqual(set(result), {"train", "validation", "test"})
        self.assertEqual(result[::2], result[1::2])
        originals = pd.DataFrame([dict(image_id="a", group_id="g", split="train", sha256="h", label=1)])
        tiles = pd.DataFrame([dict(tile_id=0, image_id="a", split="test", x0=0, y0=0, x1=1, y1=1)])
        with self.assertRaisesRegex(ValueError, "tile_parent_split_mismatch"):
            audit(originals, tiles)

    def test_image_score_includes_partial_and_no_annotation_input(self):
        originals = pd.DataFrame([dict(image_id="a", source="PKU", board="01", group_id="PKU/01",
                                       path="a.jpg", label=1)])
        tiles = pd.DataFrame(dict(image_id=["a", "a"], tile_id=[0, 1], label=[0, -1]))
        pred = image_scores(originals, tiles, [0.02, 0.91]).iloc[0]
        self.assertEqual((pred.prob_defect, pred.n_tiles, pred.max_tile_id), (0.91, 2, 1))

    def test_auc_thresholds_and_missing_denominators(self):
        single = metrics([1], [0.1], 0.5)
        for key in ("precision", "specificity", "fpr", "pr_auc", "roc_auc"):
            self.assertIsNone(single[key])
        self.assertEqual(single["fnr"], 1)
        tied = metrics([0, 1], [0.5, 0.5], 0.5)
        self.assertEqual((tied["roc_auc"], tied["pr_auc"]), (0.5, 0.75))
        val = pd.DataFrame(dict(label=[0, 1, 1], prob_defect=[0.4, 0.35, 0.8]))
        choice, _ = choose_threshold(val, dict(max_fpr=0, min_recall=1))
        self.assertFalse(choice["targets_met"])
        self.assertEqual(choice["validation"]["recall"], 0.5)
        self.assertEqual(choice["validation"]["fpr"], 0)
        missing, _ = choose_threshold(val.iloc[:1], dict(max_fpr=0, min_recall=1))
        self.assertIsNone(missing["targets_met"])
        self.assertEqual(missing["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
