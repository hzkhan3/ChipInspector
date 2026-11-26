'''
"Pointing game" for Grad-CAM: on each defective test image, does the heatmap's peak land on a labelled defect?
Run: python -m scripts.gradcam_check   (after scripts.evaluate)
'''
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.dataset import read_label_boxes
from src.data.preprocessing import find_die_box
from src.inspection.reporting import save_json
from src.models.inference import Predictor

def peak_in_boxes(heat, die_box, boxes, tol=0):
    '''Maps the heatmap peak (224x224 crop coords) back to the image and checks it against the label boxes'''
    x0, y0, x1, y1 = die_box
    py, px = np.unravel_index(heat.argmax(), heat.shape)
    x, y = x0 + (px + 0.5) * (x1 - x0) / heat.shape[1], y0 + (py + 0.5) * (y1 - y0) / heat.shape[0]
    return any(bx0 - tol <= x <= bx1 + tol and by0 - tol <= y <= by1 + tol for bx0, by0, bx1, by1 in boxes)

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    predictor = Predictor(cfg["model"]["path"], cfg["train"]["image_size"], cfg["model"]["threshold"])
    df = pd.read_csv("outputs/predictions/test_predictions.csv", keep_default_na=False)

    rows = []
    for row in df[df.label == 1].itertuples():
        image = Image.open(row.image_path).convert("RGB")
        die_box = find_die_box(image)
        boxes = [box for _, box in read_label_boxes(row.label_path)]
        heat = predictor.heatmap(image)
        tol = (die_box[2] - die_box[0]) / 14  # half a cell: Grad-CAM is only 7x7 on ResNet18
        rows.append({"source": row.source, "caught": row.pred == 1,
                     "hit": peak_in_boxes(heat, die_box, boxes), "hit_tol": peak_in_boxes(heat, die_box, boxes, tol)})

    res = pd.DataFrame(rows)
    groups = {"all": res, "A": res[res.source == "A"], "B": res[res.source == "B"],
              "caught": res[res.caught], "missed": res[~res.caught]}
    report = {name: {"n": len(g), "hit": g.hit.mean(), "hit_within_half_cell": g.hit_tol.mean()} for name, g in groups.items()}

    for name, r in report.items():
        print(f"{name:>6}: n={r['n']:3d}  peak in box {r['hit']:.1%}  within half a cell {r['hit_within_half_cell']:.1%}")
    save_json(report, "outputs/reports/gradcam_check.json")

if __name__ == "__main__":
    main()
