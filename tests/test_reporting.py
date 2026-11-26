import json

from src.inspection.reporting import save_run

def test_save_run_writes_log(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 0.5, "train_acc": 0.7, "val_loss": 0.40, "val_acc": 0.80},
        {"epoch": 2, "train_loss": 0.3, "train_acc": 0.9, "val_loss": 0.35, "val_acc": 0.85},
        {"epoch": 3, "train_loss": 0.2, "train_acc": 0.9, "val_loss": 0.45, "val_acc": 0.82},
    ]
    save_run(history, {"train": {"lr": 0.0001}}, "abc123", tmp_path)

    run = json.loads((tmp_path / "run.json").read_text())
    assert run["best_epoch"] == 2  # lowest val loss, like the checkpoint train.py keeps
    assert run["model_sha256"] == "abc123"
    assert run["git_commit"]  # a hash, or "unknown" outside git
    for name in ["history.csv", "config.yaml", "curves.png"]:
        assert (tmp_path / name).exists()
