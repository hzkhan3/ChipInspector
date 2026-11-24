import json
from pathlib import Path

def save_json(data, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def format_metrics(metrics):
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    return (
        f"Accuracy : {metrics['accuracy']:.3f}\n"
        f"Precision: {metrics['precision']:.3f}\n"
        f"Recall   : {metrics['recall']:.3f}\n"
        f"F1       : {metrics['f1']:.3f}\n"
        f"TN={tn} FP={fp} FN={fn} TP={tp}"
    )
