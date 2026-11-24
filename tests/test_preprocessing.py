from PIL import Image

from src.data.preprocessing import get_train_transforms, get_eval_transforms

def test_transforms_output_shape():
    image = Image.new("RGB", (378, 379), "gray")
    for transform in [get_train_transforms(224), get_eval_transforms(224)]:
        assert transform(image).shape == (3, 224, 224)
