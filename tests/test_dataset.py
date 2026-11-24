from PIL import Image

from src.data.dataset import ChipDataset

def test_label_path_gives_defective_target(tmp_path):
    img_path = tmp_path / "chip.jpg"
    Image.new("L", (50, 40)).save(img_path)  # grayscale like the real data

    dataset = ChipDataset([img_path, img_path], ["some_label.xml", None])

    image, target = dataset[0]
    assert image.shape[0] == 3   # converted to RGB
    assert target.item() == 1    # has a label file -> defective
    assert dataset[1][1].item() == 0  # no label file -> nondefective
    assert len(dataset) == 2
