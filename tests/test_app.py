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

def api_upload(client, data, filename):
    return client.post("/api/predict", data={"file": (io.BytesIO(data), filename)}, content_type="multipart/form-data")

def test_api_predict_returns_json(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "gray").save(buf, "PNG")
    r = api_upload(client, buf.getvalue(), "chip.png")
    assert r.status_code == 200
    body = r.get_json()
    assert body["label"] in ("defective", "nondefective")
    assert 0 <= body["defect_probability"] <= 1
    assert body["die_box"] == [0, 0, 64, 64]  # no die in a flat image -> full-image fallback

def test_api_rejects_missing_and_bad_files(client):
    assert client.post("/api/predict").status_code == 400
    r = api_upload(client, b"not an image", "notes.txt")
    assert r.status_code == 400
    assert "not an image" in r.get_json()["error"]

def test_api_oversized_upload_is_json_413(client, monkeypatch):
    monkeypatch.setitem(demo.app.config, "MAX_CONTENT_LENGTH", 1000)
    r = api_upload(client, b"x" * 2000, "big.jpg")
    assert r.status_code == 413
    assert "too large" in r.get_json()["error"]

def test_healthz(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}

def test_page_shows_test_metrics(client):
    assert b"Performance on held-out test set" in client.get("/").data
