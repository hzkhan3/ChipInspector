from PIL import Image
import xml.etree.ElementTree as ET
import pandas as pd
import torchvision
import torch
from pathlib import Path

class ChipDataset(torch.utils.data.Dataset):
    '''
    Dataset class that is inputted into the DataLoader
    1. Stores image paths, label paths, and transformation pipeline
    2. Stores length of dataset
    3. Allows you to get item that returns a transformed image and the corresponding transformed label
    '''
    def __init__(self, image_paths, labels, transform=None):
        assert len(image_paths) == len(labels), "Mismatched lists"
        
        self.image_paths = image_paths
        self.labels = labels

        if transform is None:
            self.transform = torchvision.transforms.ToTensor()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label_path = self.labels[index]

        if label_path is None:
            target = 0
        else:
            target = 1

        image = Image.open(img_path).convert('RGB')
        image_transformed = self.transform(image)

        target = torch.tensor(target, dtype=torch.long)

        return (image_transformed, target)

def get_image_paths_and_labels(raw_dir="data/raw/real"):
    '''
    Walks the real DatasetA/DatasetB folders under raw_dir and returns the image paths list and the label paths list
    '''
    raw_dir = Path(raw_dir).expanduser()
    roots = [raw_dir / "DatasetA-Real", raw_dir / "DatasetB-Real"]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        raise FileNotFoundError(f"Dataset not found at {missing[0]}. See 'Get the data' in README.md")

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    all_images = []
    all_labels = []

    for root in roots:
        # Defective images + matching XML labels
        img_dir_defect = root / "defective" / "image"
        label_dir_defect = root / "defective" / "label"

        label_map = {p.stem: p for p in label_dir_defect.glob("*.xml")}
        img_files_defect = [p for p in img_dir_defect.iterdir() if p.suffix.lower() in img_exts]
        label_files_defect = [label_map[p.stem] for p in img_files_defect]

        # Nondefective images + None labels
        img_dir_nondefect = root / "nondefective"
        img_files_nondefect = [p for p in img_dir_nondefect.iterdir() if p.suffix.lower() in img_exts]
        label_files_nondefect = [None] * len(img_files_nondefect)

        # Combine in matching order
        all_images.extend(img_files_defect + img_files_nondefect)
        all_labels.extend(label_files_defect + label_files_nondefect)

    return all_images, all_labels

def read_label_boxes(label_path):
    '''
    Reads the hand-labelled defect boxes from a Pascal VOC XML file
    Returns a list of (defect_name, (xmin, ymin, xmax, ymax))
    '''
    boxes = []
    for obj in ET.parse(label_path).getroot().iter("object"):
        b = obj.find("bndbox")
        coords = tuple(int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((obj.find("name").text, coords))
    return boxes

def dataset_from_csv(csv_path, transform=None):
    '''
    Builds a ChipDataset from a split CSV made by scripts/split_dataset.py
    (empty label_path means nondefective)
    '''
    df = pd.read_csv(csv_path, keep_default_na=False)
    labels = [p if p else None for p in df["label_path"]]
    return ChipDataset(df["image_path"].tolist(), labels, transform)

