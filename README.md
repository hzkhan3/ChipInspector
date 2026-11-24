# ChipInspector

Binary classifier for chip die images (defective vs. non-defective) using a fine-tuned ResNet18.
Each image is first cropped to the die with OpenCV (threshold + morphology + contours).

## Run the demo (offline)

```bash
source .venv/bin/activate
python app.py          # open http://127.0.0.1:8000
```

Click a sample in `demo_samples/` or upload any chip image. Blue shows the die crop the model sees; yellow shows hand-labelled defect boxes (demo defects only).

## Full pipeline

```bash
python -m scripts.explore_dataset            # dataset counts
python -m scripts.split_dataset              # 70/15/15 stratified split -> data/splits/
python -m scripts.train                      # fine-tune ResNet18 -> models/best_model.pt
python -m scripts.evaluate                   # test metrics -> outputs/reports/
python -m scripts.inspect_image img.jpg      # classify one image -> outputs/annotated/
python -m pytest                             # tests
```

Settings live in `config/config.yaml`.

## Results (held-out test set, 527 images)

| Accuracy | Recall | Precision | F1 |
|---|---|---|---|
| 92.8% | 90.3% | 89.4% | 0.898 |

The decision threshold (0.2) was tuned on the validation set to favour recall, since a missed defect costs more than a false alarm.
