import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_detector(min_size, anchor_sizes, pretrained=True):
    '''
    Faster R-CNN (ResNet50-FPN) with one class, "defect".
    - pretrained: start from COCO weights, then swap the 91-class head for defect vs background
    - min_size: images are upscaled so their short side is this, which makes tiny defects bigger
    - anchor_sizes: one per FPN level. Smaller than COCO's 32-512 because Dataset B defects are ~7 px
    '''
    anchors = AnchorGenerator(sizes=tuple((s,) for s in anchor_sizes), aspect_ratios=((0.5, 1.0, 2.0),) * len(anchor_sizes))
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights, weights_backbone=None, min_size=min_size,
                                    rpn_anchor_generator=anchors,
                                    box_score_thresh=0.0)  # keep low scores, so every image gets a score to rank by

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

def image_scores(model, images):
    '''
    Turns detections into one score per image: the most confident defect box (0 if there are none).
    This is the detector's P(defective), so it can be thresholded and compared with the classifier.
    '''
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    return [o["scores"].max().item() if len(o["scores"]) else 0.0 for o in outputs]
