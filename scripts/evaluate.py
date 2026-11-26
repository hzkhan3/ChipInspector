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

def predict_split(cfg, csv_path):
    '''Runs the saved model over a split CSV. Returns (P(defective) per image, true labels)'''
    ds = dataset_from_csv(csv_path, get_eval_transforms(cfg["train"]["image_size"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg["train"]["batch_size"])
    device = get_device()
    model = load_model(cfg["model"]["path"], device)

    defect_probs, y_true = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1]
            defect_probs.extend(probs.cpu().tolist())
            y_true.extend(labels.tolist())
    return defect_probs, y_true

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    test_csv = Path(cfg["data"]["splits_dir"]) / "test.csv"
    threshold = cfg["model"]["threshold"]
    defect_probs, y_true = predict_split(cfg, test_csv)

    y_pred = [int(p >= threshold) for p in defect_probs]
    metrics = compute_metrics(y_true, y_pred)
    print(format_metrics(metrics))

    # Save per-image predictions, metrics and confusion matrix
    df = pd.read_csv(test_csv, keep_default_na=False)
    df["pred"] = y_pred
    df["defect_probability"] = defect_probs
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)
    df.to_csv("outputs/predictions/test_predictions.csv", index=False)

    save_json(metrics, "assets/test_metrics.json")  # tracked, so the app shows it in Docker/Spaces too
    plot_confusion_matrix(metrics["confusion_matrix"], "outputs/reports/confusion_matrix.png")
    print("Saved predictions to outputs/ and metrics to assets/test_metrics.json")

if __name__ == "__main__":
    main()
