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

def test_image_listing_is_sorted(tmp_path):
    from src.data.dataset import get_image_paths_and_labels
    for ds in ["DatasetA-Real", "DatasetB-Real"]:
        (tmp_path / ds / "defective" / "image").mkdir(parents=True)
        (tmp_path / ds / "defective" / "label").mkdir()
        (tmp_path / ds / "nondefective").mkdir()
        for name in ["c", "a", "b"]:  # created out of order
            Image.new("L", (8, 8)).save(tmp_path / ds / "nondefective" / f"{name}.jpg")

    images, labels = get_image_paths_and_labels(tmp_path)
    names = [p.name for p in images]
    assert names == ["a.jpg", "b.jpg", "c.jpg"] * 2  # same order on every OS, so the split is reproducible
    assert labels == [None] * 6
