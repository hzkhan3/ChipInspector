'''
Splits the real images into train / val / test CSVs (stratified by source + class)
Run: python -m scripts.split_dataset
'''
from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.data.dataset import get_image_paths_and_labels

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))["data"]

    images, labels = get_image_paths_and_labels()
    df = pd.DataFrame({
        "image_path": [str(p) for p in images],
        "label_path": [str(p) if p else "" for p in labels],
        "label": [0 if p is None else 1 for p in labels],
        "source": ["A" if "DatasetA" in str(p) else "B" for p in images],
    })

    # Stratify on source + class so every split has the same mix
    strata = df["source"] + "_" + df["label"].astype(str)

    train_df, temp_df = train_test_split(
        df, test_size=cfg["val_size"] + cfg["test_size"], stratify=strata, random_state=cfg["seed"]
    )
    temp_strata = temp_df["source"] + "_" + temp_df["label"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, test_size=cfg["test_size"] / (cfg["val_size"] + cfg["test_size"]),
        stratify=temp_strata, random_state=cfg["seed"]
    )

    out_dir = Path(cfg["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(split)} images ({split['label'].sum()} defective)")

if __name__ == "__main__":
    main()
