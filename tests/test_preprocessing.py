from PIL import Image, ImageDraw

from src.data.preprocessing import get_train_transforms, get_eval_transforms, find_die_box

def test_transforms_output_shape():
    image = Image.new("RGB", (378, 379), "gray")
    for transform in [get_train_transforms(224), get_eval_transforms(224)]:
        assert transform(image).shape == (3, 224, 224)

def test_find_die_box_finds_bright_square():
    image = Image.new("L", (300, 300), 60)
    ImageDraw.Draw(image).rectangle([100, 100, 199, 199], fill=240)  # the "die"

    x0, y0, x1, y1 = find_die_box(image)
    assert x0 < 100 and y0 < 100 and x1 > 199 and y1 > 199  # die plus padding
    assert (x1 - x0) < 300                                   # but still cropped

def test_find_die_box_falls_back_to_full_image():
    image = Image.new("L", (200, 150), 60)  # no die at all
    assert find_die_box(image) == (0, 0, 200, 150)
