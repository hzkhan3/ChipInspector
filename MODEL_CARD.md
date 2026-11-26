# Model card: ChipInspector v1.0

Binary classifier that flags semiconductor die images as **defective** or **non-defective**.

## Model

| | |
|---|---|
| Architecture | ResNet18 (ImageNet-pretrained), final layer replaced by `Linear(512, 2)` |
| Input | Grayscale/RGB image → OpenCV die crop → 224×224 → ImageNet normalization |
| Output | P(defective), the softmax of class 1. Label = defective if P ≥ **0.2** |
| Training | Adam, lr 1e-4, batch 32, 10 epochs, cross-entropy. Augmentation: rotation ±5°, brightness/contrast ±15% |
| Checkpoint | Lowest validation loss: epoch 4 (val loss 0.122, val accuracy 95.4% at argmax) |
| Weights | [v1.0 release](https://github.com/hzkhan3/ChipInspector/releases/tag/v1.0), 43 MB, sha256 `0c215726…5e499432` (full hash in `config/config.yaml`) |

## Intended use

- A prototype and portfolio demo of a visual inspection pipeline: crop the die, classify it, show the evidence.
- At most a **screening aid** that sends likely defects to a human. It is **not** a production pass/fail gate.
- Out of scope: other chip types, cameras or lighting setups not in the dataset, defect localization and defect type.

## Data

[Chip-surface-defect-dataset](https://github.com/HiHiAllen/Chip-surface-defect-dataset), **real images only** (the synthetic set was not used):

| | Defective | Non-defective | Total |
|---|---|---|---|
| Dataset A | 780 | 1,361 | 2,141 |
| Dataset B | 461 | 909 | 1,370 |

Split 70/15/15, stratified by dataset and class (seed 42): train 2,457, val 527, test 527 (186 defective each in val and test).
An image counts as defective if it has a Pascal VOC label file. The boxes (`DIE_BROKEN`, `DIE_CRACK`, `DIE_INK`, `NO_DIE`) are used only for drawing and for checking the crop.
The dataset has no license; images remain the property of their authors.

## Performance (held-out test set, threshold 0.2)

| | Images | Accuracy | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| **All** | 527 | 92.8% | 90.3% | 89.4% | 0.973 |
| Dataset A | 322 | 97.8% | 98.3% | 95.8% | 0.998 |
| Dataset B | 205 | 84.9% | 76.8% | 77.9% | 0.866 |

The threshold minimises `ratio × FN + FP` on validation (`scripts/tune_threshold.py`): 0.2 is optimal when a missed defect costs 1–5 false alarms. If misses cost more, use a lower threshold from that script (e.g. 0.14 for 10×: test recall 91.9%, precision 79.2%). Validation recall was 95.7%.
Accuracy's 95% confidence interval is about ±2.2%.
Reproduce: `python -m scripts.evaluate && python -m scripts.error_analysis`.

## Limitations

- **Small defects are missed.** 16 of the 18 missed defects are in Dataset B, and 14 of those are 5–9 px wide (edge slivers, small ink marks). A larger input and the die crop didn't help, so fixing it needs localization (an object detector).
- **Narrow data:** two sources and one imaging setup each. Expect worse results on new lines, cameras or lighting. Watch for data drift.
- **Confidence is not calibrated.** The demo often shows ~100%; read it as a ranking, not a probability.
- **No location or defect type.** The yellow boxes in the demo are hand labels, not predictions.
- **Grad-CAM is coarse** (7×7). Its peak lands within half a cell of the labelled defect on 91.5% of Dataset A defects but only 53.6% of Dataset B's, and on 11.1% of missed defects (`scripts/gradcam_check.py`). Treat it as a rough explanation, not a localizer.
- **Crop fallback:** if no die is found (e.g. `NO_DIE` images), the full image is classified.
- **Split order:** the file listing isn't sorted, so re-running the split on another OS may produce a slightly different split.
