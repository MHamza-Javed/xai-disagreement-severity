"""
Phase 2 Sanity Check — Single Image XAI Test
Runs all 4 XAI methods on ONE correctly-classified hazelnut defective image,
for both ResNet-18 and ViT, and visualizes the heatmaps side-by-side.

Run this BEFORE the full batch generation to catch bugs early.

Usage:
    python test_xai_single_image.py
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
import config
from src.model_utils import ResNet18Classifier, ViTClassifier, load_checkpoint
from src.xai_methods import XAIAnalyzer
from torchvision import transforms


def load_image_tensor(image_path):
    """Load and preprocess a single image the same way MVTecADDataset does."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    return tensor, img  # tensor for model, raw img (0-255) for display


def get_first_defective_image(category, arch_name):
    """Pull the first correctly-classified defective image from Phase 1's filtered manifest."""
    manifest_path = config.RESULTS_PATH / "metrics" / "phase1_evaluation" / f"{category}_{arch_name}_correctly_classified.csv"
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["true_label"]) == 1:  # defective
                return row
    raise ValueError(f"No correctly-classified defective image found in {manifest_path}")


def plot_heatmaps(raw_img, heatmaps, title):
    """Plot original image + 4 heatmap overlays."""
    plt.close('all')  # force-close any stale windows from previous iterations

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(raw_img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    method_names = ['gradcam', 'ig', 'occlusion', 'gradientsshap']
    display_names = ['Grad-CAM', 'Integrated Gradients', 'Occlusion', 'GradientSHAP']

    for i, (method, display_name) in enumerate(zip(method_names, display_names)):
        hm_sum = heatmaps[method].sum()  # stamped into title for cross-checking vs. console
        axes[i + 1].imshow(raw_img)
        axes[i + 1].imshow(heatmaps[method], cmap='jet', alpha=0.5)
        axes[i + 1].set_title(f"{display_name}\n(sum={hm_sum:.2f})", fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    device = "cpu"

    for category in config.MVTEC_CATEGORIES:
        for arch_name, ModelClass in [("resnet18", ResNet18Classifier), ("vit", ViTClassifier)]:
            print(f"\n{'='*60}")
            print(f"Testing XAI methods: {category} / {arch_name}")
            print(f"{'='*60}")

            # Load model
            ckpt_path = config.MODELS_PATH / arch_name / category / f"{arch_name}_best_val_acc.pth"
            model = ModelClass(pretrained=False, num_classes=2)
            model, _, _ = load_checkpoint(model, None, ckpt_path, device)

            # Get one correctly-classified defective image
            row = get_first_defective_image(category, arch_name)
            print(f"  Using image: {row['image_path']}")

            image_tensor, raw_img = load_image_tensor(row["image_path"])
            target_class = 1  # defective

            # Run all 4 XAI methods
            analyzer = XAIAnalyzer(model, architecture=arch_name, device=device)

            print("  Computing Grad-CAM...")
            gradcam_hm = analyzer.compute_gradcam(image_tensor, target_class)
            print("  Computing Integrated Gradients...")
            ig_hm = analyzer.compute_integrated_gradients(image_tensor, target_class)
            print("  Computing Occlusion Sensitivity...")
            occ_hm = analyzer.compute_occlusion_sensitivity(image_tensor, target_class)
            print("  Computing GradientSHAP...")
            gs_hm = analyzer.compute_gradient_shap(image_tensor, target_class)

            heatmaps = {
                'gradcam': gradcam_hm,
                'ig': ig_hm,
                'occlusion': occ_hm,
                'gradientsshap': gs_hm,
            }

            print(f"  ✅ All 4 heatmaps computed successfully for {category}/{arch_name}")

            # Diagnostic: raw heatmap statistics (min/max/mean/std) to catch flat/degenerate maps
            print(f"  📈 Heatmap statistics (post-normalization, so min=0, max=1 expected if not flat):")
            for method, hm in heatmaps.items():
                print(f"     {method:15s}: min={hm.min():.4f} max={hm.max():.4f} "
                      f"mean={hm.mean():.4f} std={hm.std():.6f}")

            plot_heatmaps(raw_img, heatmaps, title=f"{category} / {arch_name} — {Path(row['image_path']).stem}")


if __name__ == "__main__":
    main()