"""
Quick check: is screw/ResNet-18's below-chance IoU uniform across defect subtypes,
or concentrated in specific ones?

Usage:
    python check_screw_resnet18_by_defect_type.py
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

sys.path.append(str(Path(__file__).parent))
import config
from src.conflict_detector import compute_mask_iou
from src.metrics import TOP_PERCENT_THRESHOLD

METHOD_NAMES = ['gradcam', 'ig', 'occlusion', 'gradientsshap']


def expected_random_iou(p1, p2):
    denom = p1 + p2 - p1 * p2
    return (p1 * p2) / denom if denom > 0 else 0.0


def main():
    manifest_path = config.RESULTS_PATH / "metrics" / "phase2_heatmap_manifest.csv"
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r["category"] == "screw" and r["architecture"] == "resnet18"]

    by_defect_type = defaultdict(list)

    for row in rows:
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        severity = (mask > 127).sum() / mask.size

        heatmaps = {m: np.load(row[f"{m}_path"]) for m in METHOD_NAMES}
        ious = [compute_mask_iou(heatmaps[m], mask) for m in METHOD_NAMES]
        best_iou = max(ious)

        chance = expected_random_iou(TOP_PERCENT_THRESHOLD, severity)
        ratio = best_iou / chance if chance > 0 else float('inf')

        by_defect_type[row["defect_type"]].append(ratio)

    print(f"{'='*60}")
    print("screw / resnet18 - Best-Method IoU vs. Chance, by Defect Type")
    print(f"{'='*60}")
    for defect_type, ratios in sorted(by_defect_type.items()):
        print(f"  {defect_type:20s}: mean ratio={np.mean(ratios):.2f}x chance  "
              f"(N={len(ratios)}, range {min(ratios):.2f}-{max(ratios):.2f}x)")


if __name__ == "__main__":
    main()