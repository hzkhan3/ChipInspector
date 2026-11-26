import torch
from PIL import Image

from src.data.detection import DetectionDataset, collate
from src.models.detector import build_detector, image_scores

XML = """<annotation><object><name>DIE_INK</name>
<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>17</xmax><ymax>30</ymax></bndbox></object>
<object><name>DIE_CRACK</name>
<bndbox><xmin>40</xmin><ymin>5</ymin><xmax>50</xmax><ymax>5</ymax></bndbox></object></annotation>"""

def make_csv(tmp_path):
    Image.new("L", (64, 48)).save(tmp_path / "bad.jpg")  # grayscale like the real data
    Image.new("L", (64, 48)).save(tmp_path / "good.jpg")
    (tmp_path / "bad.xml").write_text(XML)
    csv = tmp_path / "split.csv"
    csv.write_text(f"image_path,label_path\n{tmp_path / 'bad.jpg'},{tmp_path / 'bad.xml'}\n{tmp_path / 'good.jpg'},\n")
    return csv

def test_detection_dataset(tmp_path):
    dataset = DetectionDataset(make_csv(tmp_path))
    image, target = dataset[0]
    assert image.shape == (3, 48, 64)  # RGB, original size (no crop, no resize)
    assert target["boxes"].tolist() == [[10, 20, 17, 30]]  # zero-height box dropped
    assert target["labels"].tolist() == [1]

    _, good = dataset[1]
    assert good["boxes"].shape == (0, 4) and len(good["labels"]) == 0  # good image: no boxes

def test_train_flips_keep_box_size(tmp_path):
    torch.manual_seed(0)
    dataset = DetectionDataset(make_csv(tmp_path), train=True)
    for _ in range(5):
        x0, y0, x1, y1 = dataset[0][1]["boxes"][0].tolist()
        assert (x1 - x0, y1 - y0) == (7, 10)  # flipped boxes move but keep their size

def test_detector_trains_and_scores(tmp_path):
    images, targets = collate([DetectionDataset(make_csv(tmp_path))[i] for i in range(2)])
    model = build_detector(min_size=64, anchor_sizes=[16, 32, 64, 128, 256], pretrained=False)

    model.train()
    losses = model(list(images), list(targets))  # includes the good image with no boxes
    assert all(torch.isfinite(v) for v in losses.values())

    scores = image_scores(model, list(images))
    assert len(scores) == 2 and all(0 <= s <= 1 for s in scores)
