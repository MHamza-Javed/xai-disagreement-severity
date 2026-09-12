"""
Phase 3 — Severity-Stratified Analysis

For every image in the Phase 2 manifest:
  1. Load its 4 heatmaps
  2. Compute IoU and Spearman's rho disagreement for all 6 method-pairs
  3. Aggregate via mean (primary) and max (for Phase 4) into per-image scores
  4. Compute defect severity from its ground-truth mask

Then, per category x architecture group:
  5. Run Spearman correlation: disagreement vs. severity (answers RQ1, RQ2)
  6. Save per-image scores and the correlation summary
  7. Generate scatter plots

Usage:
    python analyze_severity_correlation.py
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config
from src.metrics import (
    compute_all_pairwise_scores, aggregate_disagreement,
    compute_defect_severity, correlation_analysis,
)


def load_manifest():
    manifest_path = config.RESULTS_PATH / "metrics" / "phase2_heatmap_manifest.csv"
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_per_image_scores(rows):
    """Compute disagreement + severity for every image in the manifest."""
    results = []

    for row in tqdm(rows, desc="Computing scores", unit="img"):
        heatmaps = {
            'gradcam': np.load(row["gradcam_path"]),
            'ig': np.load(row["ig_path"]),
            'occlusion': np.load(row["occlusion_path"]),
            'gradientsshap': np.load(row["gradientsshap_path"]),
        }

        pairwise = compute_all_pairwise_scores(heatmaps)

        mean_iou_disagreement = aggregate_disagreement(pairwise['iou_disagreements'], agg='mean')
        max_iou_disagreement = aggregate_disagreement(pairwise['iou_disagreements'], agg='max')
        mean_spearman_disagreement = aggregate_disagreement(pairwise['spearman_disagreements'], agg='mean')
        max_spearman_disagreement = aggregate_disagreement(pairwise['spearman_disagreements'], agg='max')

        severity = None
        if row["mask_path"] and Path(row["mask_path"]).exists():
            mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
            severity = compute_defect_severity(mask)

        results.append({
            "image_path": row["image_path"],
            "category": row["category"],
            "architecture": row["architecture"],
            "defect_type": row["defect_type"],
            "severity": severity,
            "mean_iou_disagreement": mean_iou_disagreement,
            "max_iou_disagreement": max_iou_disagreement,
            "mean_spearman_disagreement": mean_spearman_disagreement,
            "max_spearman_disagreement": max_spearman_disagreement,
        })

    return results


def save_per_image_scores(results, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📄 Per-image scores saved: {output_path}")


def run_correlation_analysis(results):
    """
    Group by category x architecture, run Spearman correlation between
    mean disagreement (both IoU-based and pixel-based) and severity.
    """
    groups = {}
    for r in results:
        if r["severity"] is None:
            continue
        key = (r["category"], r["architecture"])
        groups.setdefault(key, []).append(r)

    summary_rows = []

    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS: Disagreement vs. Defect Severity")
    print(f"{'='*70}")

    for (category, architecture), rows in sorted(groups.items()):
        severities = [r["severity"] for r in rows]
        n = len(rows)

        for metric_name, key in [
            ("iou_disagreement", "mean_iou_disagreement"),
            ("spearman_disagreement", "mean_spearman_disagreement"),
        ]:
            disagreements = [r[key] for r in rows]
            rho, p_value = correlation_analysis(disagreements, severities)

            significance = "significant (p<0.05)" if p_value < 0.05 else "not significant"
            direction = "inverse (disagreement down as severity up)" if rho < 0 else "positive (disagreement up as severity up)"

            print(f"\n{category} / {architecture} — {metric_name}")
            print(f"  N = {n}")
            print(f"  Spearman's rho = {rho:.4f}  |  p-value = {p_value:.4f}  |  {significance}")
            print(f"  Direction: {direction}")

            summary_rows.append({
                "category": category,
                "architecture": architecture,
                "metric": metric_name,
                "spearman_rho": rho,
                "p_value": p_value,
                "n_samples": n,
                "significant": p_value < 0.05,
            })

    return summary_rows, groups


def save_correlation_summary(summary_rows, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n📄 Correlation summary saved: {output_path}")


def generate_scatter_plots(groups, output_dir):
    """One scatter plot per category x architecture, disagreement vs. severity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for (category, architecture), rows in sorted(groups.items()):
        severities = [r["severity"] for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, key, title in [
            (axes[0], "mean_iou_disagreement", "IoU-based Disagreement"),
            (axes[1], "mean_spearman_disagreement", "Spearman-based Disagreement"),
        ]:
            disagreements = [r[key] for r in rows]
            rho, p_value = correlation_analysis(disagreements, severities)

            ax.scatter(severities, disagreements, alpha=0.6, edgecolors='k', linewidths=0.5)
            ax.set_xlabel("Defect Severity (normalized mask area)")
            ax.set_ylabel("Mean Disagreement (1 - metric, averaged over 6 pairs)")
            ax.set_title(f"{title}\nrho={rho:.3f}, p={p_value:.3f}, N={len(rows)}")
            ax.grid(alpha=0.3)

        plt.suptitle(f"{category} / {architecture} — Disagreement vs. Severity")
        plt.tight_layout()

        out_path = output_dir / f"{category}_{architecture}_disagreement_vs_severity.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {out_path}")


def main():
    print("Loading Phase 2 manifest...")
    rows = load_manifest()
    print(f"Found {len(rows)} images across all category/architecture combinations.\n")

    results = compute_per_image_scores(rows)

    metrics_dir = config.RESULTS_PATH / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_per_image_scores(results, metrics_dir / "phase3_per_image_scores.csv")

    summary_rows, groups = run_correlation_analysis(results)
    save_correlation_summary(summary_rows, metrics_dir / "phase3_correlation_summary.csv")

    print(f"\n{'='*70}")
    print("Generating scatter plots...")
    print(f"{'='*70}")
    plots_dir = config.RESULTS_PATH / "plots" / "correlation_plots"
    generate_scatter_plots(groups, plots_dir)

    print(f"\n{'='*70}")
    print("Phase 3 severity-correlation analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()