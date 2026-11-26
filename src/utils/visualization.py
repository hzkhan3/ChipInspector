import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt

# OpenCV colours are BGR, not RGB
RED, GREEN, YELLOW, BLUE, WHITE = (40, 40, 220), (70, 170, 40), (0, 215, 255), (230, 160, 0), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def annotate_image(image, result, die_box=None, label_boxes=None):
    '''
    Draws the prediction with OpenCV and returns a BGR numpy array (OpenCV's image format)
    - die_box: region the model actually sees after the die crop (blue)
    - label_boxes: hand-labelled defects from the XML file (yellow)
    '''
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    if die_box:
        cv2.rectangle(img, die_box[:2], die_box[2:], BLUE, 1)

    for name, (x0, y0, x1, y1) in label_boxes or []:
        cv2.rectangle(img, (x0, y0), (x1, y1), YELLOW, 2)
        cv2.putText(img, name.replace("DIE_", ""), (x0, max(34, y0 - 4)), FONT, 0.4, YELLOW, 1, cv2.LINE_AA)

    # Coloured border + banner with the verdict
    color = RED if result["label"] == "defective" else GREEN
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, max(3, w // 60))
    cv2.rectangle(img, (0, 0), (w, 22), color, -1)  # -1 = filled
    text = f"{result['label'].upper()} {result['confidence']:.0%}"
    cv2.putText(img, text, (6, 16), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    return img

def overlay_heatmap(image, heatmap, die_box, label_boxes=None):
    '''
    Blends a Grad-CAM heatmap (computed on the die crop) back onto the die box, then outlines the
    labelled defects so you can check whether the model looked at them. Returns BGR.
    '''
    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = die_box
    heat = cv2.resize(heatmap, (x1 - x0, y1 - y0))  # undo the 224x224 squash of the crop
    color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    img[y0:y1, x0:x1] = cv2.addWeighted(img[y0:y1, x0:x1], 0.55, color, 0.45, 0)

    for _, (bx0, by0, bx1, by1) in label_boxes or []:
        cv2.rectangle(img, (bx0, by0), (bx1, by1), YELLOW, 2)
    return img

def plot_cost_curves(thresholds, curves, chosen, filepath):
    '''curves: {cost ratio: total cost at each threshold} on the validation set, chosen: {cost ratio: threshold}'''
    fig, ax = plt.subplots(figsize=(5, 4))
    for ratio, cost in curves.items():
        line, = ax.plot(thresholds, cost, label=f"miss = {ratio} false alarms")
        ax.plot(chosen[ratio], cost[list(thresholds).index(chosen[ratio])], "o", color=line.get_color())
    ax.set_yscale("log")
    ax.set_xlabel("Threshold on P(defective)")
    ax.set_ylabel("Validation cost (ratio × FN + FP)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)

def plot_confusion_matrix(cm, filepath):
    labels = ["nondefective", "defective"]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)

def plot_pr_curves(curves, threshold_points, filepath):
    '''
    curves: {name: (precision, recall)}, threshold_points: {name: (precision, recall)} at the chosen threshold
    '''
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, (p, r) in curves.items():
        line, = ax.plot(r, p, label=name)
        ax.plot(*threshold_points[name][::-1], "o", color=line.get_color())
    ax.set_xlabel("Recall (defects caught)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on test set (dot = threshold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
