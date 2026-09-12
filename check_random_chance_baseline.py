"""
Phase 4 Addendum — Random-Chance IoU Baseline

For each category x architecture group, computes the analytical expected IoU
between a random top-20%-of-pixels region and the actual ground-truth mask,
using each image's REAL severity (not an approximation). This tells us whether
the measured consensus/individual IoU values are meaningfully above chance,
or statistically indistinguishable from noise.

Formula (independent random overlap of two binary regions in N total pixels):
    p1 = fraction of pixels flagged by heatmap (locked at 0.20, top-20% binarization)
    p2 = defect severity (fraction of pixels that are the true defect)
    E[IoU] = (p1 * p2) / (p1 + p2 - p1 * p2)

Usage:
    python check_random_chance_baseline.py
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))
import config
from src.metrics import TOP_PERCENT_THRESHOLD


def expected_random_iou(p1, p2):
    """Analytical expected IoU between two independent random binary regions."""
    denom = p1 + p2 - p1 * p2
    if denom <= 0:
        return 0.0
    return (p1 * p2) / denom


def load_per_image_scores():
    path = config.RESULTS_PATH / "metrics" / "phase3_per_image_scores.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["severity"] = float(r["severity"]) if r["severity"] else None
    return rows


def load_phase4_summary():
    path = config.RESULTS_PATH / "metrics" / "phase4_conflict_detector_summary.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    per_image_rows = load_per_image_scores()
    phase4_summary = load_phase4_summary()

    print(f"{'='*90}")
    print("RANDOM-CHANCE IoU BASELINE vs. MEASURED CONSENSUS IoU")
    print(f"{'='*90}")

    results = []

    for p4_row in phase4_summary:
        category = p4_row["category"]
        architecture = p4_row["architecture"]

        group_severities = [
            r["severity"] for r in per_image_rows
            if r["category"] == category and r["architecture"] == architecture
            and r["severity"] is not None
        ]

        # Expected random IoU per image, then averaged (matches how consensus IoU was averaged)
        random_ious = [expected_random_iou(TOP_PERCENT_THRESHOLD, sev) for sev in group_severities]
        mean_random_iou = np.mean(random_ious)

        consensus_iou = float(p4_row["mean_consensus_iou"])
        mean_individual_iou = float(p4_row["mean_of_individual_ious"])

        ratio_consensus = consensus_iou / mean_random_iou if mean_random_iou > 0 else float('inf')
        ratio_individual = mean_individual_iou / mean_random_iou if mean_random_iou > 0 else float('inf')

        status = "ABOVE chance" if ratio_consensus > 1.2 else (
            "AT chance" if ratio_consensus > 0.8 else "BELOW chance"
        )

        print(f"\n{category} / {architecture}")
        print(f"  Mean defect severity:              {np.mean(group_severities):.5f}")
        print(f"  Expected random-chance IoU:         {mean_random_iou:.5f}")
        print(f"  Mean individual-method IoU:         {mean_individual_iou:.5f}  ({ratio_individual:.2f}x chance)")
        print(f"  Mean consensus IoU:                 {consensus_iou:.5f}  ({ratio_consensus:.2f}x chance)")
        print(f"  Verdict: consensus is {status}")

        results.append({
            "category": category,
            "architecture": architecture,
            "mean_severity": np.mean(group_severities),
            "random_chance_iou": mean_random_iou,
            "mean_individual_iou": mean_individual_iou,
            "individual_vs_chance_ratio": ratio_individual,
            "consensus_iou": consensus_iou,
            "consensus_vs_chance_ratio": ratio_consensus,
            "verdict": status,
        })

    output_path = config.RESULTS_PATH / "metrics" / "phase4_random_chance_baseline.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'='*90}")
    print(f"Saved: {output_path}")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()