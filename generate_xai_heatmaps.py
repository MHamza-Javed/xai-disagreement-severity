"""
Phase 2 — Full Batch XAI Heatmap Generation

For every correctly-classified test image (from Phase 1's filtered manifests),
generates all 4 XAI heatmaps (Grad-CAM, Integrated Gradients, Occlusion, GradientSHAP)
using the appropriate trained model, and saves each as a .npy file under xai_outputs/.

Also writes a combined manifest (results/metrics/phase2_heatmap_manifest.csv) linking
every image to its heatmap paths, mask path, defect type, and label — this is what
Phase 3's metrics computation will load directly.

Usage:
    python generate_xai_heatmaps.py
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config
from src.model_utils import ResNet18Classifier, ViTClassifier, load_checkpoint
from src.xai_methods import XAIAnalyzer

METHOD_NAMES = ['gradcam', 'ig', 'occlusion', 'gradientsshap']

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image_tensor(image_path):
    """Load and preprocess a single image (same pipeline as training)."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(img).unsqueeze(0)  # [1, C, H, W]
    return tensor


def get_output_path(method, category, architecture, image_stem):
    """
    Build the .npy output path following the project's folder convention:
    xai_outputs/{method}/{category}/{architecture}/{image_stem}_{method}_{architecture}.npy
    """
    method_dir_map = {
        'gradcam': 'gradcam',
        'ig': 'integrated_gradients',
        'occlusion': 'occlusion_sensitivity',
        'gradientsshap': 'gradientsshap',
    }
    out_dir = config.XAI_OUTPUTS_PATH / method_dir_map[method] / category / architecture
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{image_stem}_{method}_{architecture}.npy"


def process_combination(category, arch_name, ModelClass, device, manifest_rows):
    """Generate all 4 heatmaps for every image in this category+architecture combo."""
    ckpt_path = config.MODELS_PATH / arch_name / category / f"{arch_name}_best_val_acc.pth"
    model = ModelClass(pretrained=False, num_classes=2)
    model, _, _ = load_checkpoint(model, None, ckpt_path, device)
    analyzer = XAIAnalyzer(model, architecture=arch_name, device=device)

    filtered_manifest_path = (
        config.RESULTS_PATH / "metrics" / "phase1_evaluation"
        / f"{category}_{arch_name}_correctly_classified.csv"
    )

    with open(filtered_manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    defective_rows = [r for r in rows if int(r["true_label"]) == 1]

    print(f"\n{'='*60}")
    print(f"Generating heatmaps: {category} / {arch_name} "
          f"({len(defective_rows)} correctly-classified defective images)")
    print(f"{'='*60}")

    n_errors = 0

    for row in tqdm(defective_rows, desc=f"  {category}/{arch_name}", unit="img"):
        image_path = Path(row["image_path"])
        image_stem = image_path.stem
        target_class = 1  # defective

        try:
            image_tensor = load_image_tensor(image_path)
            heatmap_paths = {}

            for method in METHOD_NAMES:
                if method == 'gradcam':
                    hm = analyzer.compute_gradcam(image_tensor, target_class)
                elif method == 'ig':
                    hm = analyzer.compute_integrated_gradients(image_tensor, target_class)
                elif method == 'occlusion':
                    hm = analyzer.compute_occlusion_sensitivity(image_tensor, target_class)
                elif method == 'gradientsshap':
                    hm = analyzer.compute_gradient_shap(image_tensor, target_class)

                out_path = get_output_path(method, category, arch_name, image_stem)
                np.save(out_path, hm.astype(np.float32))
                heatmap_paths[method] = str(out_path)

            manifest_rows.append({
                "image_path": str(image_path),
                "mask_path": row["mask_path"],
                "category": category,
                "architecture": arch_name,
                "defect_type": row["defect_type"],
                "gradcam_path": heatmap_paths['gradcam'],
                "ig_path": heatmap_paths['ig'],
                "occlusion_path": heatmap_paths['occlusion'],
                "gradientsshap_path": heatmap_paths['gradientsshap'],
            })

        except Exception as e:
            n_errors += 1
            print(f"\n  ⚠️  Error processing {image_path.name}: {e}")

    if n_errors > 0:
        print(f"  ⚠️  {n_errors} image(s) failed — check errors above.")
    else:
        print(f"  ✅ All {len(defective_rows)} images processed successfully.")


def main():
    device = "cpu"
    print(f"Using device: {device}")

    manifest_rows = []
    start_time = time.time()

    combinations = [
        (category, arch_name, ModelClass)
        for category in config.MVTEC_CATEGORIES
        for arch_name, ModelClass in [("resnet18", ResNet18Classifier), ("vit", ViTClassifier)]
    ]

    for category, arch_name, ModelClass in combinations:
        process_combination(category, arch_name, ModelClass, device, manifest_rows)

    # Save combined manifest for Phase 3
    output_dir = config.RESULTS_PATH / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "phase2_heatmap_manifest.csv"

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✨ Phase 2 heatmap generation complete!")
    print(f"   Total images processed: {len(manifest_rows)}")
    print(f"   Total time: {elapsed/60:.1f} minutes")
    print(f"   Manifest saved: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()