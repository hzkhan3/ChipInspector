'''
Minimal local web UI for the demo - runs fully offline
Run: python app.py   then open http://127.0.0.1:8000
'''
import base64
import json
import time
from pathlib import Path

import cv2
import yaml
from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image

from src.inspection.pipeline import analyze
from src.models.inference import Predictor

cfg = yaml.safe_load(open("config/config.yaml"))
predictor = Predictor(cfg["model"]["path"], cfg["train"]["image_size"], cfg["model"]["threshold"])

SAMPLES_DIR = Path("demo_samples")
METRICS_PATH = Path("outputs/reports/test_metrics.json")
metrics = json.load(open(METRICS_PATH)) if METRICS_PATH.exists() else None

app = Flask(__name__)

def to_data_url(image_bgr):
    '''Encode an OpenCV image so it can be shown directly in the page'''
    ok, buf = cv2.imencode(".jpg", image_bgr)
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    image, name, label_path = None, None, None

    if request.method == "POST" and request.files.get("file"):
        file = request.files["file"]
        image, name = Image.open(file.stream).convert("RGB"), file.filename
    elif request.args.get("sample"):
        name = Path(request.args["sample"]).name  # .name stops paths like ../../secret
        image = Image.open(SAMPLES_DIR / name).convert("RGB")
        xml = (SAMPLES_DIR / name).with_suffix(".xml")  # demo defects ship with their label file
        label_path = xml if xml.exists() else None

    result, annotated, ms = None, None, None
    if image:
        start = time.perf_counter()
        result, annotated = analyze(predictor, image, label_path)
        ms = (time.perf_counter() - start) * 1000

    samples = sorted(p.name for p in SAMPLES_DIR.glob("*.jpg"))
    return render_template_string(
        PAGE, result=result, ms=ms, name=name, samples=samples, metrics=metrics,
        image=to_data_url(annotated) if image else None, threshold=predictor.threshold,
    )

@app.route("/samples/<name>")
def sample(name):
    return send_from_directory(SAMPLES_DIR, name)

PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>ChipInspector</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; background: #f4f5f7; color: #1d2330; margin: 0; }
  main { max-width: 960px; margin: 0 auto; padding: 32px 20px; }
  h1 { margin: 0; font-size: 26px; }
  .sub { color: #667085; margin: 4px 0 24px; }
  .card { background: white; border-radius: 12px; padding: 24px; box-shadow: 0 1px 3px #0001; margin-bottom: 20px; }
  .result { display: flex; gap: 32px; align-items: center; flex-wrap: wrap; }
  .result img { width: 380px; max-width: 100%; border-radius: 8px; }
  .defective { color: #d92d20; }
  .nondefective { color: #079455; }
  .key { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 4px; }
  .verdict { font-size: 34px; font-weight: 700; letter-spacing: 1px; }
  .bar { height: 10px; background: #eaecf0; border-radius: 5px; width: 280px; overflow: hidden; margin: 6px 0 16px; }
  .bar div { height: 100%; background: currentColor; }
  .muted { color: #667085; font-size: 14px; }
  .upload { display: inline-block; background: #1d2330; color: white; padding: 10px 18px; border-radius: 8px; cursor: pointer; }
  .upload input { display: none; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; margin-top: 12px; }
  .grid a { text-decoration: none; color: inherit; font-size: 12px; overflow-wrap: anywhere; }
  .grid img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 6px; border: 2px solid transparent; }
  .grid a:hover img { border-color: #1d2330; }
  .stats { display: flex; gap: 28px; }
  .stats b { display: block; font-size: 22px; }
</style>
</head>
<body>
<main>
  <h1>ChipInspector</h1>
  <p class="sub">Classifies chip die images as defective or non-defective &middot; ResNet18 fine-tuned with PyTorch</p>

  {% if result %}
  <div class="card result {{ result.label }}">
    <img src="{{ image }}" alt="inspected chip">
    <div>
      <div class="muted">{{ name }}</div>
      <div class="verdict">{{ "DEFECTIVE" if result.label == "defective" else "PASS" }}</div>
      <div class="muted">Confidence {{ "%.1f"|format(result.confidence * 100) }}%</div>
      <div class="bar"><div style="width: {{ result.confidence * 100 }}%"></div></div>
      <div class="muted">Defect probability: {{ "%.3f"|format(result.defect_probability) }} (threshold {{ threshold }})</div>
      <div class="muted">Processing time: {{ "%.0f"|format(ms) }} ms</div>
      <p class="muted">
        <span class="key" style="background:#00a0e6"></span>Die region the model sees (OpenCV crop)<br>
        {% if result.labelled_defects %}
        <span class="key" style="background:#ffd700"></span>Hand-labelled defect: {{ result.labelled_defects|join(", ") }}
        {% endif %}
      </p>
    </div>
  </div>
  {% endif %}

  <div class="card">
    <form method="post" enctype="multipart/form-data">
      <label class="upload">Upload a chip image
        <input type="file" name="file" accept="image/*" onchange="this.form.submit()">
      </label>
    </form>
    <p class="muted">Or click a sample (held-out test images the model never saw in training):</p>
    <div class="grid">
      {% for s in samples %}
      <a href="/?sample={{ s }}"><img src="/samples/{{ s }}" alt="{{ s }}">{{ s }}</a>
      {% endfor %}
    </div>
  </div>

  {% if metrics %}
  <div class="card">
    <div class="muted" style="margin-bottom: 10px">Performance on held-out test set</div>
    <div class="stats">
      <div><b>{{ "%.1f"|format(metrics.accuracy * 100) }}%</b><span class="muted">Accuracy</span></div>
      <div><b>{{ "%.1f"|format(metrics.recall * 100) }}%</b><span class="muted">Recall (defects caught)</span></div>
      <div><b>{{ "%.1f"|format(metrics.precision * 100) }}%</b><span class="muted">Precision</span></div>
      <div><b>{{ "%.3f"|format(metrics.f1) }}</b><span class="muted">F1</span></div>
    </div>
  </div>
  {% endif %}
</main>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(port=8000, debug=False)  # not 5000: macOS AirPlay uses that port
