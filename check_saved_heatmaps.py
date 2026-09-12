"""
Phase 2 Verification — Visualize Saved Heatmaps
Loads a few rows from the Phase 2 manifest and displays the saved .npy heatmaps
directly from disk, alongside the original image and ground-truth mask.

This checks that what was SAVED matches what we expect — no recomputation.

Usage:
    python check_saved_heatmaps.py [category] [architecture] [n_images]

Examples:
    python check_saved_heatmaps.py                      # defaults: hazelnut resnet18 3
    python check_saved_heatmaps.py screw vit 5
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
import config


def load_manifest_rows(category, architecture):
    manifest_path = config.RESULTS_PATH / "metrics" / "phase2_heatmap_manifest.csv"
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == category and row["architecture"] == architecture:
                rows.append(row)
    return rows


def plot_saved_heatmaps(row, idx):
    """Load image, mask, and all 4 saved heatmaps for one manifest row; plot them."""
    img = cv2.imread(row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = None
    if row["mask_path"] and Path(row["mask_path"]).exists():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

    heatmaps = {
        'Grad-CAM': np.load(row["gradcam_path"]),
        'Integrated Gradients': np.load(row["ig_path"]),
        'Occlusion': np.load(row["occlusion_path"]),
        'GradientSHAP': np.load(row["gradientsshap_path"]),
    }

    n_panels = 2 + len(heatmaps)  # original + mask + 4 heatmaps
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    if mask is not None:
        axes[1].imshow(mask, cmap="gray")
        defect_pct = 100 * (mask > 127).sum() / mask.size
        axes[1].set_title(f"Mask ({defect_pct:.2f}%)")
    else:
        axes[1].text(0.5, 0.5, "No mask", ha="center", va="center")
    axes[1].axis("off")

    for i, (name, hm) in enumerate(heatmaps.items()):
        axes[i + 2].imshow(img)
        axes[i + 2].imshow(hm, cmap="jet", alpha=0.5)
        axes[i + 2].set_title(f"{name}\n(min={hm.min():.2f}, max={hm.max():.2f})", fontsize=9)
        axes[i + 2].axis("off")

    plt.suptitle(f"{row['category']} / {row['architecture']} — {Path(row['image_path']).stem} "
                 f"(defect: {row['defect_type']})")
    plt.tight_layout()
    plt.show()


def main():
    category = sys.argv[1] if len(sys.argv) > 1 else "hazelnut"
    architecture = sys.argv[2] if len(sys.argv) > 2 else "resnet18"
    n_images = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    rows = load_manifest_rows(category, architecture)
    print(f"Found {len(rows)} entries for {category}/{architecture} in the manifest.")

    if not rows:
        print("No matching rows found. Check category/architecture spelling.")
        return

    for i, row in enumerate(rows[:n_images]):
        print(f"\nDisplaying image {i+1}/{min(n_images, len(rows))}: {Path(row['image_path']).name}")
        plot_saved_heatmaps(row, i)


if __name__ == "__main__":
    main()