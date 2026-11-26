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
    (tmp_path / "chip.xml").write_text(  # label file next to the image
        "<annotation><object><name>DIE_CRACK</name>"
        "<bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox>"
        "</object></annotation>"
    )

    predictor = Predictor(model_path, device="cpu")
    result = inspect_image(predictor, img_path, output_dir=tmp_path / "outputs")

    assert result["label"] in ("defective", "nondefective")
    assert 0.5 <= result["confidence"] <= 1.0
    assert result["labelled_defects"] == ["DIE_CRACK"]
    assert (tmp_path / "outputs" / "annotated" / "chip.jpg").exists()
    assert (tmp_path / "outputs" / "annotated" / "chip_gradcam.jpg").exists()
    assert (tmp_path / "outputs" / "reports" / "chip.json").exists()

def test_missing_model_points_to_download_script(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError, match="download_model"):
        Predictor(tmp_path / "nope.pt", device="cpu")
