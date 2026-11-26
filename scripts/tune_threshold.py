'''
Picks the decision threshold from a cost ratio: one missed defect (FN) costs as much as `ratio` false alarms (FP).
Tuned on the validation split only; test numbers are shown for information, never used to choose.
Run: python -m scripts.tune_threshold   (after scripts.evaluate)
'''
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.evaluate import predict_split
from src.evaluation.metrics import compute_metrics
from src.inspection.reporting import save_json
from src.utils.visualization import plot_cost_curves

RATIOS = [1, 2, 5, 10, 20, 50]
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)

def costs(probs, labels, ratio):
    '''Total cost (ratio * FN + FP) at every threshold in THRESHOLDS'''
    pred = np.asarray(probs)[:, None] >= THRESHOLDS
    labels = np.asarray(labels)[:, None]
    return ratio * ((labels == 1) & ~pred).sum(0) + ((labels == 0) & pred).sum(0)

def summary(name, t, val_probs, val_y, test):
    val = compute_metrics(val_y, [int(p >= t) for p in val_probs])
    tst = compute_metrics(test.label, (test.defect_probability >= t).astype(int))
    (_, vfp), (vfn, _) = val["confusion_matrix"]
    (_, tfp), (tfn, _) = tst["confusion_matrix"]
    return {"ratio": name, "threshold": t, "val_fn": vfn, "val_fp": vfp, "val_recall": val["recall"],
            "test_fn": tfn, "test_fp": tfp, "test_recall": tst["recall"], "test_precision": tst["precision"]}

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    val_probs, val_y = predict_split(cfg, Path(cfg["data"]["splits_dir"]) / "val.csv")
    test = pd.read_csv("outputs/predictions/test_predictions.csv", keep_default_na=False)

    rows = [summary("current", cfg["model"]["threshold"], val_probs, val_y, test)]
    curves, chosen = {}, {}
    for ratio in RATIOS:
        curves[ratio] = costs(val_probs, val_y, ratio)
        best = THRESHOLDS[curves[ratio] == curves[ratio].min()]
        t = float(best[len(best) // 2])  # middle of the cheapest range, so it isn't sitting on an edge
        chosen[ratio] = t
        rows.append(summary(ratio, t, val_probs, val_y, test))

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format="%.3f"))
    save_json(table.to_dict("records"), "outputs/reports/threshold_tuning.json")
    plot_cost_curves(THRESHOLDS, curves, chosen, "outputs/reports/cost_curves.png")
    print("\nPick the ratio that matches the real cost of a missed defect, then set model.threshold in config.yaml")

if __name__ == "__main__":
    main()
