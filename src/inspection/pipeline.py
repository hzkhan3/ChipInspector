from pathlib import Path
import cv2
from PIL import Image

from src.data.dataset import read_label_boxes
from src.data.preprocessing import find_die_box
from src.inspection.reporting import save_json
from src.utils.visualization import annotate_image, overlay_heatmap

def find_label_file(image_path):
    '''
    Looks for an XML label next to the image, or in the dataset layout (defective/image -> defective/label)
    '''
    for p in [image_path.with_suffix(".xml"), image_path.parent.parent / "label" / f"{image_path.stem}.xml"]:
        if p.exists():
            return p
    return None

def analyze(predictor, image, label_path=None):
    '''
    Predicts one image and draws the result.
    Returns (result dict, annotated BGR image, Grad-CAM overlay BGR image)
    '''
    result = predictor.predict(image)
    die_box = find_die_box(image)
    label_boxes = read_label_boxes(label_path) if label_path else []

    result["die_box"] = list(die_box)
    result["labelled_defects"] = [name for name, _ in label_boxes]
    heat = predictor.heatmap(image) * result["defect_probability"]  # a confident pass shows no hotspot
    gradcam = overlay_heatmap(image, heat, die_box, label_boxes)
    return result, annotate_image(image, result, die_box, label_boxes), gradcam

def inspect_image(predictor, image_path, output_dir="outputs"):
    '''
    Full inspection of one image file: predict -> save annotated + Grad-CAM images -> save JSON report
    '''
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    result, annotated, gradcam = analyze(predictor, image, find_label_file(image_path))
    result["image"] = str(image_path)

    annotated_path = Path(output_dir) / "annotated" / image_path.name
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(annotated_path.with_name(f"{image_path.stem}_gradcam.jpg")), gradcam)

    save_json(result, Path(output_dir) / "reports" / f"{image_path.stem}.json")
    return result
