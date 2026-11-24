'''
Classifies a single image from the command line
Run: python -m scripts.inspect_image path/to/image.jpg
'''
import sys
import yaml

from src.inspection.pipeline import inspect_image
from src.models.inference import Predictor

def main():
    cfg = yaml.safe_load(open("config/config.yaml"))
    predictor = Predictor(cfg["model"]["path"], cfg["train"]["image_size"], cfg["model"]["threshold"])

    for path in sys.argv[1:]:
        result = inspect_image(predictor, path)
        print(f"{path}: {result['label']} ({result['confidence']:.1%} confident)")

if __name__ == "__main__":
    main()
