'''
Fine-tunes a pretrained ResNet18 on the train split and keeps the best model (lowest val loss)
Run: python -m scripts.train
'''
from datetime import datetime
from functools import partial
from pathlib import Path
import torch
import yaml

from src.data.dataset import dataset_from_csv
from src.data.preprocessing import get_train_transforms, get_eval_transforms
from src.models.classifier import ChipClassifier, focal_loss, train_one_epoch, validate_one_epoch, save_model
from src.models.inference import get_device
from src.inspection.reporting import save_run
from scripts.download_model import sha256

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    train_cfg = cfg["train"]
    splits_dir = Path(cfg["data"]["splits_dir"])
    torch.manual_seed(train_cfg["seed"])

    train_ds = dataset_from_csv(splits_dir / "train.csv", get_train_transforms(train_cfg["image_size"]))
    val_ds = dataset_from_csv(splits_dir / "val.csv", get_eval_transforms(train_cfg["image_size"]))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=train_cfg["batch_size"])

    device = get_device()
    model = ChipClassifier(pretrained=True)
    criterion = torch.nn.CrossEntropyLoss()
    train_criterion = criterion
    if train_cfg["loss"] == "focal":
        train_criterion = partial(focal_loss, gamma=train_cfg["focal_gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = None
    if train_cfg["lr_schedule"] == "cosine":  # big steps early, small steps late, so val loss settles
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    print(f"Training on {device} | train={len(train_ds)} val={len(val_ds)}")
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)  # always cross-entropy, so runs compare
        history.append({"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, cfg["model"]["path"], epoch, val_loss, val_acc)
            print(f"  -> saved best model to {cfg['model']['path']}")

    run_dir = Path("outputs/runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    save_run(history, cfg, sha256(cfg["model"]["path"]), run_dir)
    print(f"Saved run log to {run_dir}")

if __name__ == "__main__":
    main()
