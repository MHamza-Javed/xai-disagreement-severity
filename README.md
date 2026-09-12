# Does Explanation Disagreement Track Defect Severity?

### A Ground-Truth Study and a Lightweight Conflict Detector for XAI in Industrial Visual Defect Inspection

**Author:** M Hamza Javed (Reg No: 2023-MC-27)
**Target Venue:** INMIC 2026 — 28th International Multi-Topic Conference, UET Lahore

---

## Overview

This repository investigates whether **inter-method disagreement between XAI (explainable AI) techniques correlates with physical defect severity** in industrial visual inspection, and whether that disagreement can be turned into a practical trust signal.

We apply four post-hoc explanation methods — **Grad-CAM, Integrated Gradients, Occlusion Sensitivity, and GradientSHAP** — to two CNN/Transformer architectures (**ResNet-18** and a **Vision Transformer**) fine-tuned on two categories from the **MVTec AD** industrial defect dataset (**hazelnut** and **screw**), and measure how much these methods agree or disagree with each other and with ground-truth defect masks.

---

## Research Questions

1. **Severity Correlation (RQ1):** Does inter-method XAI disagreement scale with defect subtlety?
2. **Architecture Dependence (RQ2):** Is this relationship consistent across CNN vs. ViT?
3. **Practical Resolution (RQ3):** Can disagreement be used to build a lightweight Conflict Detector that flags untrustworthy explanations and produces a better consensus explanation?

---

## Key Findings

This study reports its results honestly, including where the original hypothesis was **not** supported — validated throughout with robustness checks and analytical chance baselines rather than relying on raw correlation coefficients alone.

| RQ | Finding |
|----|---------|
| **RQ1** | **Not supported at the pooled level.** Spearman correlations between disagreement and severity were weak and non-significant across all four (category × architecture) conditions. An outlier-sensitivity check showed the one notable trend (hazelnut/ViT) is fragile — it hovers at the edge of significance (p = 0.047–0.104) depending on which high-severity points are included, and should not be reported as a confirmed effect. |
| **RQ2** | **Architecture matters, but unevenly.** No clean universal pattern emerged from the severity correlations. The strongest architecture-dependence evidence instead comes from RQ3: on screw defects, ResNet-18's best-method localization sits **below random chance** (0.56×) while ViT's sits **well above chance** (4.15×) — a 7× gap on the same images, same defects. |
| **RQ3** | **A real, validated positive result.** An accuracy-weighted consensus heatmap significantly outperforms the average single-method baseline in all four conditions (Wilcoxon p < 0.05), and localizes ground-truth defects meaningfully above random chance in **three of four** conditions (3.4×–5.2× chance). The exception (screw/ResNet-18) is not uniform failure — a per-defect-subtype breakdown shows two subtypes localize above chance and two below, indicating a defect-specific rather than purely architectural failure mode. |

**A methodological note on absolute IoU:** XAI methods highlight discriminative regions for a classifier's decision, not full defect extent — they are not segmentation models. Pixel-level IoU against a ground-truth mask is therefore used here as a **relative** comparison (against other methods and against a random-chance baseline), not as an absolute claim about localization/segmentation accuracy.

---

## Repository Structure

```
Code_Start/
├── data/
│   ├── mvtec_ad/                 # Raw MVTec AD dataset (hazelnut, screw)
│   ├── preprocessed/             # Resized (224x224) train/val/test splits
│   └── ground_truth_masks/       # Aligned defect masks
├── models/
│   ├── resnet18/{category}/      # Fine-tuned ResNet-18 checkpoints
│   └── vit/{category}/           # Fine-tuned ViT checkpoints (frozen backbone)
├── xai_outputs/
│   ├── gradcam/
│   ├── integrated_gradients/
│   ├── occlusion_sensitivity/
│   └── gradientsshap/            # Saved .npy heatmaps per image/category/architecture
├── results/
│   ├── metrics/                  # All CSV outputs (evaluation, disagreement, correlation, Phase 4)
│   └── plots/                    # Correlation scatter plots
├── src/
│   ├── data_loader.py            # MVTec AD preprocessing & PyTorch Dataset/DataLoader
│   ├── model_utils.py            # ResNet-18 / ViT training, two-stage fine-tuning
│   ├── xai_methods.py            # All 4 XAI methods (custom Grad-CAM adaptation for ViT)
│   ├── metrics.py                # IoU, Spearman's rho, disagreement aggregation
│   └── conflict_detector.py      # Consensus heatmap + disagreement-threshold flagging
├── config.py                     # Central configuration (paths, hyperparameters, locked decisions)
├── train.py                      # Phase 1: train both architectures on both categories
├── evaluate_models.py            # Phase 1: test-set evaluation, filter to correctly-classified images
├── generate_xai_heatmaps.py      # Phase 2: batch XAI heatmap generation (resumable)
├── analyze_severity_correlation.py  # Phase 3: disagreement vs. severity correlation analysis
├── check_outlier_robustness.py   # Phase 3: outlier-sensitivity robustness check
├── phase4_conflict_detector.py   # Phase 4: consensus map + flag validation
├── check_random_chance_baseline.py  # Phase 4: analytical chance-level IoU baseline
├── check_screw_resnet18_by_defect_type.py  # Phase 4: per-defect-subtype breakdown
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `captum` (XAI), `timm` (ViT), `scipy`, `scikit-learn`, `opencv-python`, `matplotlib`.

Download **MVTec AD** (hazelnut and screw categories) from the [official source](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place under `data/mvtec_ad/{hazelnut,screw}/`.

---

## Reproducing the Pipeline

Run scripts in order from the project root:

```bash
# Phase 1 — Preprocessing, training, evaluation
python src/data_loader.py            # builds 30/20/50 stratified splits, resizes images
python train.py                      # fine-tunes ResNet-18 and ViT on both categories
python evaluate_models.py            # test-set evaluation, filters to correctly-classified images

# Phase 2 — XAI heatmap generation
python generate_xai_heatmaps.py      # generates all 4 XAI heatmaps per correctly-classified image
                                      # (resumable — safe to re-run after an interruption)

# Phase 3 — Severity correlation analysis
python analyze_severity_correlation.py
python check_outlier_robustness.py   # sensitivity check on the severity correlations

# Phase 4 — Conflict Detector
python phase4_conflict_detector.py
python check_random_chance_baseline.py
python check_screw_resnet18_by_defect_type.py
```

---

## Methodology Notes

- **Train/val/test split:** MVTec AD's defective images (native to the `test/` folder only) were re-split 30/20/50 per defect subtype to allow binary classification while maximizing the held-out set used for XAI analysis.
- **Heatmap binarization:** top-20% intensity thresholding (kept constant across all methods to give each method the same "attention budget," rather than architecture/method-dependent thresholds like Otsu's).
- **Disagreement aggregation:** mean across all 6 pairwise method comparisons for the severity-correlation analysis (RQ1/RQ2); max pairwise disagreement is used separately for the Conflict Detector's flagging threshold (RQ3).
- **ViT Grad-CAM adaptation:** standard Grad-CAM assumes convolutional feature maps; for ViT we hook the second-to-last transformer block (the last block's patch-token outputs are a dead end in the gradient graph — only its CLS token feeds the classification head) and pool gradients across the token axis (the spatial analog) rather than the feature axis (which is subject to an exact-zero identity from the following LayerNorm).
- **ViT training:** the ViT backbone is kept fully frozen throughout training (linear-probing the pretrained features); an initial attempt at staged unfreezing destabilized training due to optimizer-state resets combined with a head/backbone feature mismatch.

---

## Limitations

- Sample sizes per condition (35–56 correctly-classified defective test images) are underpowered to detect a moderate correlation effect (ρ ≈ 0.3) at conventional significance.
- Screw's defect severity range is naturally very narrow (~0.1–0.7% of image area), limiting the severity-correlation analysis's ability to detect any relationship for that category regardless of whether one exists.
- Scope was narrowed from the originally planned four MVTec AD categories to two (hazelnut, screw) to fit the project timeline; both architectures and all four XAI methods were retained.

---

## License

[Specify your license here, e.g., MIT]

## Acknowledgments

MVTec AD dataset: Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection," CVPR 2019.
