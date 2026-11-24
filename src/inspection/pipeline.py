from pathlib import Path
from PIL import Image

from src.inspection.reporting import save_json
from src.utils.visualization import annotate_image

def inspect_image(predictor, image_path, output_dir="outputs"):
    '''
    Full inspection of one image: predict -> save annotated image -> save JSON report
    '''
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")

    result = predictor.predict(image)
    result["image"] = str(image_path)

    annotated_path = Path(output_dir) / "annotated" / image_path.name
    annotated_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_image(image, result).save(annotated_path)

    save_json(result, Path(output_dir) / "reports" / f"{image_path.stem}.json")
    return result
