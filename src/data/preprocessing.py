import cv2
import numpy as np
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def find_die_box(image, pad=0.15, min_area=0.08):
    '''
    Finds the die (the large bright square) with OpenCV and returns a padded box (x0, y0, x1, y1).
    Falls back to the full image if no die is found (e.g. NO_DIE images).
    '''
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Threshold: Otsu splits dark vs bright, then move the cutoff halfway to the
    #    brightest pixels so only the die survives (not the bright background)
    otsu, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = (otsu + np.percentile(blur, 99)) / 2
    _, mask = cv2.threshold(blur, cutoff, 255, cv2.THRESH_BINARY)

    # 2. Opening (erode then dilate) removes thin bright bits touching the die, like solder bumps
    k = max(3, min(H, W) // 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    # 3. The largest blob is the die
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, W, H)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < min_area * W * H:
        return (0, 0, W, H)

    # 4. Pad so defects on the die edge stay in view
    px, py = int(w * pad), int(h * pad)
    return (max(0, x - px), max(0, y - py), min(W, x + w + px), min(H, y + h + py))

def crop_to_die(image):
    return image.crop(find_die_box(image))

def get_train_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Lambda(crop_to_die),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=5, fill=(128, 128, 128)),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0,
            hue=0
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return train_transform

def get_eval_transforms(image_size):
    eval_transform = transforms.Compose([
        transforms.Lambda(crop_to_die),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return eval_transform
