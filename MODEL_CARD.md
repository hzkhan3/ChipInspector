# Model card: ChipInspector v1.1

Binary classifier that flags semiconductor die images as **defective** or **non-defective**.

## Model

| | |
|---|---|
| Architecture | ResNet18 (ImageNet-pretrained), final layer replaced by `Linear(512, 2)` |
| Input | Grayscale/RGB image → OpenCV die crop → 224×224 → ImageNet normalization |
| Output | P(defective), the softmax of class 1. Label = defective if P ≥ **0.32** |
| Training | Adam, lr 1e-4, batch 32, 10 epochs, cross-entropy. Augmentation: rotation ±5°, brightness/contrast ±15% |
| Checkpoint | Lowest validation loss: epoch 7 of 10 (val loss 0.195, val accuracy 91.5% at argmax). [Training curves](assets/training_curves.png) |
| Weights | [v1.1 release](https://github.com/hzkhan3/ChipInspector/releases/tag/v1.1), 43 MB, sha256 `c05d3a1c…8bf7f08` (full hash in `config/config.yaml`) |

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

Split 70/15/15, stratified by dataset and class (seed 42, file lists sorted so the split is the same on every OS): train 2,457, val 527, test 527 (186 defective each in val and test).
An image counts as defective if it has a Pascal VOC label file. The boxes (`DIE_BROKEN`, `DIE_CRACK`, `DIE_INK`, `NO_DIE`) are used only for drawing and for checking the crop.
The dataset has no license; images remain the property of their authors.

## Performance (held-out test set, threshold 0.32)

| | Images | Accuracy | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|
| **All** | 527 | 91.7% | 95.2% | 83.5% | 0.973 |
| Dataset A | 322 | 96.9% | 97.4% | 94.2% | 0.987 |
| Dataset B | 205 | 83.4% | 91.3% | 69.2% | 0.918 |

The threshold minimises `ratio × FN + FP` on validation (`scripts/tune_threshold.py`): 0.32 is optimal when a missed defect costs 2–5 false alarms. If misses cost more, use a lower threshold from that script (e.g. 0.09 for 10–20×: test recall 97.3%, precision 60.9%). Validation recall was 91.4%.
Accuracy's 95% confidence interval is about ±2.4%.
Reproduce: `python -m scripts.evaluate && python -m scripts.error_analysis && python -m scripts.tune_threshold`.

v1.0 (same architecture, OS-dependent split, threshold 0.2) scored 92.8% accuracy and 90.3% recall on a different test set, with the same 0.973 ROC-AUC.

## Limitations

- **Small defects are missed.** 6 of the 9 missed defects are in Dataset B; 4 of those are 5–8 px wide (edge slivers, small ink marks). Dataset B also produces 28 of the 35 false alarms. In v1.0 experiments, a larger input and the die crop didn't help, so fixing it needs localization (an object detector).
- **Some misses are confident.** Dataset A's 3 misses have P ≤ 0.01, including a 41 px crack.
- **Narrow data:** two sources and one imaging setup each. Expect worse results on new lines, cameras or lighting. Watch for data drift.
- **Confidence is not calibrated.** The demo often shows ~100%; read it as a ranking, not a probability.
- **No location or defect type.** The yellow boxes in the demo are hand labels, not predictions.
- **Grad-CAM is coarse** (7×7). Its peak lands within half a cell of the labelled defect on 87.2% of Dataset A defects and 79.7% of Dataset B's, but on only 11.1% of missed defects (`scripts/gradcam_check.py`). Treat it as a rough explanation, not a localizer.
- **Crop fallback:** if no die is found (e.g. `NO_DIE` images), the full image is classified.
- **Reproducibility:** the split and seed are fixed. On the same Mac (MPS), a re-run reproduced epoch 1 to four decimal places; on other machines or GPUs, results may still differ slightly.
