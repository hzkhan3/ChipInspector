'''
Breaks the test results down by dataset and lists the missed defects
Run: python -m scripts.error_analysis   (after scripts.evaluate)
'''
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_auc_score

from src.data.dataset import read_label_boxes
from src.evaluation.metrics import compute_metrics
from src.inspection.reporting import save_json
from src.utils.visualization import plot_pr_curves

def main():
    threshold = yaml.safe_load(open("config/config.yaml"))["model"]["threshold"]
    df = pd.read_csv("outputs/predictions/test_predictions.csv", keep_default_na=False)

    report, curves, points = {}, {}, {}
    for name, part in [("all", df), ("A", df[df.source == "A"]), ("B", df[df.source == "B"])]:
        m = compute_metrics(part.label, part.pred)
        m["roc_auc"] = roc_auc_score(part.label, part.defect_probability)
        m["n"] = len(part)
        report[name] = m
        p, r, _ = precision_recall_curve(part.label, part.defect_probability)
        curves[f"Dataset {name}"], points[f"Dataset {name}"] = (p, r), (m["precision"], m["recall"])
        print(f"{name:>3}: n={m['n']:3d}  acc={m['accuracy']:.3f}  recall={m['recall']:.3f}  "
              f"precision={m['precision']:.3f}  auc={m['roc_auc']:.3f}")

    # Missed defects: what was labelled, and how small the smallest defect box is
    missed = []
    for row in df[(df.label == 1) & (df.pred == 0)].itertuples():
        boxes = read_label_boxes(row.label_path)
        missed.append({
            "image": row.image_path.split("/")[-1],
            "source": row.source,
            "defect_probability": round(row.defect_probability, 3),
            "defects": [name for name, _ in boxes],
            "smallest_box_px": min(min(x1 - x0, y1 - y0) for _, (x0, y0, x1, y1) in boxes),
        })
    report["missed_defects"] = missed
    report["threshold"] = threshold

    print(f"\nMissed {len(missed)} defects ({sum(m['source'] == 'B' for m in missed)} from Dataset B):")
    for m in sorted(missed, key=lambda m: m["smallest_box_px"]):
        print(f"  {m['source']}  p={m['defect_probability']:.3f}  smallest box {m['smallest_box_px']:3d}px  "
              f"{', '.join(m['defects'])}")

    save_json(report, "outputs/reports/error_analysis.json")
    plot_pr_curves(curves, points, "outputs/reports/pr_curve.png")
    print("\nSaved outputs/reports/error_analysis.json and pr_curve.png")

if __name__ == "__main__":
    main()
