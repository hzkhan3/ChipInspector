import json
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from src.utils.visualization import plot_training_curves

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

def git_commit():
    '''Short commit hash, with "-dirty" if tracked files have uncommitted changes'''
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):  # no git, or not a repo
        return "unknown"

def save_run(history, cfg, model_sha256, out_dir):
    '''Saves what's needed to compare or reproduce a training run: config, commit, per-epoch history, curves'''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best = min(history, key=lambda h: h["val_loss"])
    save_json({"git_commit": git_commit(), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
               "best_val_acc": best["val_acc"], "model_sha256": model_sha256}, out_dir / "run.json")
    plot_training_curves(history, best["epoch"], out_dir / "curves.png")
