'''
Evaluates the saved model on the held-out test split
Run: python -m scripts.evaluate
'''
from pathlib import Path
import pandas as pd
import torch
import yaml

from src.data.dataset import dataset_from_csv
from src.data.preprocessing import get_eval_transforms
from src.evaluation.metrics import compute_metrics
from src.inspection.reporting import save_json, format_metrics
from src.models.classifier import load_model
from src.models.inference import get_device
from src.utils.visualization import plot_confusion_matrix

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    test_csv = Path(cfg["data"]["splits_dir"]) / "test.csv"
    threshold = cfg["model"]["threshold"]

    test_ds = dataset_from_csv(test_csv, get_eval_transforms(cfg["train"]["image_size"]))
    loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])

    device = get_device()
    model = load_model(cfg["model"]["path"], device)

    defect_probs, y_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            defect_probs.extend(probs.cpu().tolist())
            y_true.extend(labels.tolist())

    y_pred = [int(p >= threshold) for p in defect_probs]
    metrics = compute_metrics(y_true, y_pred)
    print(format_metrics(metrics))

    # Save per-image predictions, metrics and confusion matrix
    df = pd.read_csv(test_csv, keep_default_na=False)
    df["pred"] = y_pred
    df["defect_probability"] = defect_probs
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    df.to_csv("outputs/predictions/test_predictions.csv", index=False)

    save_json(metrics, "outputs/reports/test_metrics.json")
    plot_confusion_matrix(metrics["confusion_matrix"], "outputs/reports/confusion_matrix.png")
    print("Saved predictions and reports to outputs/")

if __name__ == "__main__":
    main()
