# ChipInspector

Binary classifier for chip die images (defective vs. non-defective) using a fine-tuned ResNet18.

## Run the demo (offline)

```bash
source .venv/bin/activate
python app.py          # open http://127.0.0.1:5000
```

Click a sample in `demo_samples/` or upload any chip image.

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
| 93.0% | 88.7% | 91.2% | 0.899 |

The decision threshold (0.3) was tuned on the validation set to favour recall, since a missed defect costs more than a false alarm.
