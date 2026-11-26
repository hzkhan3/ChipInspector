from PIL import Image, ImageDraw

from src.models.classifier import ChipClassifier, save_model
from src.models.inference import Predictor
from src.utils.visualization import overlay_heatmap

def test_heatmap_shape_and_range(tmp_path):
    save_model(ChipClassifier(pretrained=False), tmp_path / "model.pt")
    predictor = Predictor(tmp_path / "model.pt", device="cpu")

    image = Image.new("L", (300, 200), 60)
    ImageDraw.Draw(image).rectangle([100, 50, 199, 149], fill=240)  # a "die"
    heat = predictor.heatmap(image)

    assert heat.shape == (224, 224)  # the model's input size, not the image size
    assert heat.min() >= 0 and heat.max() <= 1

    overlay = overlay_heatmap(image, heat, (80, 30, 220, 170))
    assert overlay.shape == (200, 300, 3)  # drawn back onto the full image
