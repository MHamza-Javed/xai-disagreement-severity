"""
Phase 3 Robustness Check — Outlier Sensitivity
Re-runs the hazelnut correlation analysis after removing the top N highest-severity
points, to check whether the weak correlation is driven by a handful of outliers
or reflects a broader pattern across the full sample.

Usage:
    python check_outlier_robustness.py
"""

import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config
from src.metrics import correlation_analysis


def load_per_image_scores():
    path = config.RESULTS_PATH / "metrics" / "phase3_per_image_scores.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["severity"] = float(r["severity"]) if r["severity"] else None
        r["mean_iou_disagreement"] = float(r["mean_iou_disagreement"])
        r["mean_spearman_disagreement"] = float(r["mean_spearman_disagreement"])
    return rows


def run_check(rows, category, architecture, n_remove_list=(0, 1, 2, 3)):
    group = [r for r in rows if r["category"] == category and r["architecture"] == architecture
             and r["severity"] is not None]
    group_sorted = sorted(group, key=lambda r: r["severity"], reverse=True)

    print(f"\n{'='*70}")
    print(f"{category} / {architecture} — Outlier Sensitivity Check")
    print(f"{'='*70}")
    print(f"Full sample severities (highest 5): "
          f"{[round(r['severity'], 4) for r in group_sorted[:5]]}")

    for n_remove in n_remove_list:
        trimmed = group_sorted[n_remove:]  # drop the n_remove highest-severity points
        severities = [r["severity"] for r in trimmed]

        for metric_name, key in [
            ("iou_disagreement", "mean_iou_disagreement"),
            ("spearman_disagreement", "mean_spearman_disagreement"),
        ]:
            disagreements = [r[key] for r in trimmed]
            rho, p_value = correlation_analysis(disagreements, severities)
            sig = "significant" if p_value < 0.05 else "n.s."
            print(f"  Remove top {n_remove} outlier(s) (N={len(trimmed)}) | {metric_name:22s} | "
                  f"rho={rho:+.4f}  p={p_value:.4f}  [{sig}]")


def main():
    rows = load_per_image_scores()

    for category in config.MVTEC_CATEGORIES:
        for architecture in ["resnet18", "vit"]:
            run_check(rows, category, architecture)


if __name__ == "__main__":
    main()