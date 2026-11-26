import io

import pytest
from PIL import Image

import app as demo
from src.models.classifier import ChipClassifier, save_model
from src.models.inference import Predictor

@pytest.fixture
def client(tmp_path, monkeypatch):
    # Untrained model so the tests don't need the real weights
    save_model(ChipClassifier(pretrained=False), tmp_path / "model.pt")
    monkeypatch.setattr(demo, "predictor", Predictor(tmp_path / "model.pt", device="cpu"))
    return demo.app.test_client()

def upload(client, data, filename):
    return client.post("/", data={"file": (io.BytesIO(data), filename)}, content_type="multipart/form-data")

def test_upload_image_gives_verdict(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "JPEG")
    r = upload(client, buf.getvalue(), "chip.jpg")
    assert r.status_code == 200
    assert b"DEFECTIVE" in r.data or b"PASS" in r.data

def test_non_image_upload_is_rejected(client):
    r = upload(client, b"not an image", "<b>notes</b>.txt")
    assert r.status_code == 400
    assert b"is not an image" in r.data
    assert b"<b>notes</b>" not in r.data  # filename is escaped, not rendered as HTML

def test_oversized_upload_is_rejected(client, monkeypatch):
    monkeypatch.setitem(demo.app.config, "MAX_CONTENT_LENGTH", 1000)
    r = upload(client, b"x" * 2000, "big.jpg")
    assert r.status_code == 413
    assert b"too large" in r.data

def test_sample_shows_labelled_defect(client):
    r = client.get("/?sample=defective_A_crack.jpg")
    assert r.status_code == 200
    assert b"DIE_CRACK" in r.data

def test_unknown_sample_is_404(client):
    assert client.get("/?sample=../config/config.yaml").status_code == 404

def test_samples_route_serves_only_jpgs(client):
    assert client.get("/samples/good_B_1.jpg").status_code == 200
    assert client.get("/samples/defective_A_crack.xml").status_code == 404
