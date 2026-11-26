import pandas as pd
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2

from src.data.dataset import read_label_boxes

class DetectionDataset(torch.utils.data.Dataset):
    '''
    Image + hand-labelled defect boxes for the detector, built from a split CSV.
    - Every box is one class, 1 = defect (torchvision detectors reserve 0 for background)
    - Good images get no boxes, so the detector also learns what "nothing" looks like
    - No die crop and no resize/normalize: boxes stay in image coordinates, and the model resizes + normalizes itself
    '''
    def __init__(self, csv_path, train=False):
        df = pd.read_csv(csv_path, keep_default_na=False)
        self.image_paths = df["image_path"].tolist()
        self.label_paths = df["label_path"].tolist()
        flips = [v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()] if train else []  # flips move the boxes too
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)] + flips)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        label_path = self.label_paths[index]
        boxes = [b for _, b in read_label_boxes(label_path)] if label_path else []
        boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]  # one train label has zero height

        boxes = tv_tensors.BoundingBoxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                         format="XYXY", canvas_size=(image.height, image.width))
        image, boxes = self.transform(image, boxes)
        return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}

def collate(batch):
    '''Detectors take lists (images can differ in size), so don't stack the batch into one tensor'''
    return tuple(zip(*batch))
