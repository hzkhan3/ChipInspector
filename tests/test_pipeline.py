from PIL import Image

from src.inspection.pipeline import inspect_image
from src.models.classifier import ChipClassifier, save_model
from src.models.inference import Predictor

def test_inspect_image_end_to_end(tmp_path):
    # Untrained model is enough to check the plumbing (no download needed)
    model_path = tmp_path / "model.pt"
    save_model(ChipClassifier(pretrained=False), model_path)

    img_path = tmp_path / "chip.jpg"
    Image.new("RGB", (100, 100), "white").save(img_path)

    predictor = Predictor(model_path, device="cpu")
    result = inspect_image(predictor, img_path, output_dir=tmp_path / "outputs")

    assert result["label"] in ("defective", "nondefective")
    assert 0.5 <= result["confidence"] <= 1.0
    assert (tmp_path / "outputs" / "annotated" / "chip.jpg").exists()
    assert (tmp_path / "outputs" / "reports" / "chip.json").exists()
