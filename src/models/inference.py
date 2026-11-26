import torch

from src.data.preprocessing import get_eval_transforms
from src.models.classifier import load_model
from src.models.gradcam import grad_cam

CLASS_NAMES = ["nondefective", "defective"]

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"

class Predictor:
    '''
    Loads a trained model once and classifies single PIL images
    '''
    def __init__(self, model_path, image_size=224, threshold=0.5, device=None):
        self.device = device or get_device()
        self.model = load_model(model_path, self.device)
        self.transform = get_eval_transforms(image_size)
        self.threshold = threshold

    def predict(self, image):
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)  # add batch dim

        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]

        defect_prob = probs[1].item()
        label = "defective" if defect_prob >= self.threshold else "nondefective"
        confidence = defect_prob if label == "defective" else 1 - defect_prob

        return {"label": label, "confidence": confidence, "defect_probability": defect_prob}

    def heatmap(self, image):
        '''Grad-CAM for "defective" over the 224x224 die crop the model sees'''
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return grad_cam(self.model, x)
