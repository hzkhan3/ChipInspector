import matplotlib
matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt
from PIL import ImageDraw

def annotate_image(image, result):
    '''
    Returns a copy of the image with a coloured border and the prediction written on it
    '''
    image = image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    color = (220, 40, 40) if result["label"] == "defective" else (40, 170, 70)

    w, h = image.size
    draw.rectangle([0, 0, w - 1, h - 1], outline=color, width=max(3, w // 60))
    draw.rectangle([0, 0, w, 18], fill=color)
    draw.text((5, 3), f"{result['label'].upper()}  {result['confidence']:.0%}", fill="white")
    return image

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
