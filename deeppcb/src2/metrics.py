"""Metrics with explicit denominators, image pooling and validation-only calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import BOARDS, SOURCES


def ratio(a, b):
    return float(a / b) if b else None


def arrays(labels, scores):
    y = np.asarray(labels)
    p = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected matching arrays of binary targets and scores")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Nonfinite or out-of-range probability")
    return y.astype(np.int64), p


def threshold_counts(y, p, thresholds):
    order = np.argsort(p, kind="stable")
    positive = np.r_[0, np.cumsum(y[order])]
    cuts = np.searchsorted(p[order], thresholds, side="left")
    tp = positive[-1] - positive[cuts]
    fp = len(y) - cuts - tp
    return tp, fp


def area(x, y):
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def metrics(labels, scores, threshold):
    y, p = arrays(labels, scores)
    decision = p >= threshold
    tp = int(np.sum(decision & (y == 1)))
    fp = int(np.sum(decision & (y == 0)))
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    fn, tn = positives - tp, negatives - fp
    roc_auc = pr_auc = None
    if positives and negatives:
        true_pos, false_pos = threshold_counts(y, p, np.unique(p)[::-1])
        recall = np.r_[0, true_pos / positives]
        roc_auc = area(np.r_[0, false_pos / negatives], recall)
        pr_auc = area(recall, np.r_[1, true_pos / (true_pos + false_pos)])
    measures = dict(accuracy=ratio(tp + tn, len(y)), precision=ratio(tp, tp + fp),
                    recall=ratio(tp, positives), specificity=ratio(tn, negatives),
                    f1=ratio(2 * tp, 2 * tp + fp + fn), roc_auc=roc_auc, pr_auc=pr_auc,
                    fpr=ratio(fp, negatives), fnr=ratio(fn, positives))
    undefined = [k for k, v in measures.items() if v is None]
    return dict(n=len(y), n_normal=negatives, n_defect=positives, tn=tn, fp=fp, fn=fn, tp=tp,
                **measures, undefined=", ".join(undefined))


def log_loss(labels, scores):
    y, p = arrays(labels, scores)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def image_scores(originals, tiles, probabilities):
    # The input is the complete grid, including partial-overlap tiles excluded from supervision.
    if set(tiles.image_id) != set(originals.image_id) or len(probabilities) != len(tiles):
        raise ValueError("Inference must cover every original and its complete tile grid")
    scored = tiles[["image_id", "tile_id"]].assign(prob_defect=np.asarray(probabilities, dtype=np.float64))
    grouped = scored.groupby("image_id", sort=False)
    maxima = scored.loc[grouped.prob_defect.idxmax()].rename(columns={"tile_id": "max_tile_id"})
    result = originals[["image_id", "source", "board", "group_id", "path", "label"]].merge(
        maxima, on="image_id", validate="one_to_one")
    result["n_tiles"] = result.image_id.map(grouped.size())
    result["prob_normal"] = 1.0 - result.prob_defect
    return result


def choose_threshold(validation, policy):
    y, p = arrays(validation.label, validation.prob_defect)
    thresholds = np.unique(np.r_[p, 0.0, 0.5, 1.0])
    tp, fp = threshold_counts(y, p, thresholds)
    positives, negatives = y.sum(), len(y) - y.sum()
    recall = tp / positives if positives else np.full(len(tp), np.nan)
    fpr = fp / negatives if negatives else np.full(len(fp), np.nan)
    curve = pd.DataFrame(dict(threshold=thresholds, recall=recall, fpr=fpr, tp=tp, fp=fp))
    if not positives or not negatives:
        selected = 0.5
        reason = "Validation lacks one or both classes; the default threshold is retained."
        feasible = None
    else:
        cap = fpr <= policy["max_fpr"]
        valid = cap & (recall >= policy["min_recall"])
        feasible = bool(valid.any())
        if feasible:
            candidates = valid
            reason = "Both targets met. Maximize recall, then minimize FPR, then choose closest to 0.5."
        elif cap.any():
            candidates = cap
            reason = "Recall target infeasible under the FPR cap. Maximize recall within the cap."
        else:
            candidates = fpr == fpr.min()
            reason = "FPR cap infeasible. Minimize FPR, then maximize recall."
        options = curve[candidates].assign(distance=lambda frame: abs(frame.threshold - 0.5))
        best = options.sort_values(["recall", "fpr", "distance", "threshold"],
                                   ascending=[False, True, True, True]).iloc[0]
        selected = float(best.threshold)
    result = dict(threshold=selected, selected_on="validation originals", policy=policy,
                  targets_met=feasible, reason=reason, validation=metrics(y, p, selected))
    return result, curve


def evaluations(predictions, operating):
    scopes = [("all", predictions)]
    scopes.extend((source, predictions[predictions.source == source]) for source in SOURCES)
    scopes.extend((f"VisA/{board}", predictions[(predictions.source == "VisA") &
                                                (predictions.board == board)]) for board in BOARDS)
    rows = []
    for mode, threshold in (("default", 0.5), ("operating", operating)):
        for scope, frame in scopes:
            rows.append(dict(scope=scope, mode=mode, threshold=threshold,
                             **metrics(frame.label, frame.prob_defect, threshold)))
    return pd.DataFrame(rows)


def save_predictions(directory, predictions, operating):
    predictions = predictions.assign(pred_default=(predictions.prob_defect >= 0.5).astype(int),
                                     pred_operating=(predictions.prob_defect >= operating).astype(int))
    predictions.to_csv(directory / "predictions.csv", index=False)
    evaluations(predictions, operating).to_csv(directory / "metrics.csv", index=False)
