"""
Phase 4 — Conflict Detector Engineering & Validation

For each category x architecture group:
  1. Compute each method's mask-IoU (localization accuracy) across all images
  2. Derive consensus weights from those accuracies
  3. Learn a disagreement threshold (75th percentile of max-pairwise disagreement)
  4. For every image: generate the consensus heatmap, compute its mask-IoU,
     and compare against the BEST individual method's mask-IoU on that image
  5. Also check: do FLAGGED (high-disagreement) images have worse mask-IoU on
     average than unflagged ones? (validates that disagreement is a meaningful
     trust signal, independent of the severity-correlation result from Phase 3)

Usage:
    python phase4_conflict_detector.py
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
import config
from src.conflict_detector import (
    compute_mask_iou, compute_method_weights, ConflictDetector,
)

METHOD_NAMES = ['gradcam', 'ig', 'occlusion', 'gradientsshap']


def load_manifest():
    manifest_path = config.RESULTS_PATH / "metrics" / "phase2_heatmap_manifest.csv"
    with open(manifest_path, newline="") as f:
        return list(csv.DictReader(f))


def load_per_image_scores():
    """Load Phase 3's per-image disagreement scores (need max_iou_disagreement)."""
    path = config.RESULTS_PATH / "metrics" / "phase3_per_image_scores.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    scores_by_path = {}
    for r in rows:
        scores_by_path[r["image_path"]] = {
            "max_iou_disagreement": float(r["max_iou_disagreement"]),
            "max_spearman_disagreement": float(r["max_spearman_disagreement"]),
        }
    return scores_by_path


def load_heatmaps(row):
    return {method: np.load(row[f"{method}_path"]) for method in METHOD_NAMES}


def process_group(category, architecture, rows, disagreement_scores):
    """Run the full Phase 4 pipeline for one category x architecture group."""
    group_rows = [r for r in rows if r["category"] == category and r["architecture"] == architecture]

    print(f"\n{'='*70}")
    print(f"Phase 4: {category} / {architecture}  (N={len(group_rows)})")
    print(f"{'='*70}")

    # --- Step 1: compute each method's mask-IoU across all images ---
    mask_ious_by_method = {m: [] for m in METHOD_NAMES}
    per_image_data = []

    for row in tqdm(group_rows, desc="  Computing mask IoUs", unit="img"):
        if not row["mask_path"] or not Path(row["mask_path"]).exists():
            continue
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        heatmaps = load_heatmaps(row)

        method_ious = {}
        for method in METHOD_NAMES:
            iou = compute_mask_iou(heatmaps[method], mask)
            method_ious[method] = iou
            mask_ious_by_method[method].append(iou)

        per_image_data.append({
            "image_path": row["image_path"],
            "mask": mask,
            "heatmaps": heatmaps,
            "method_ious": method_ious,
        })

    # --- Step 2: derive consensus weights from historical accuracy ---
    weights = compute_method_weights(mask_ious_by_method)
    print(f"\n  Consensus weights (by mean mask-IoU accuracy):")
    for method, w in sorted(weights.items(), key=lambda x: -x[1]):
        mean_iou = np.mean(mask_ious_by_method[method])
        print(f"    {method:15s}: weight={w:.3f}  (mean mask-IoU={mean_iou:.4f})")

    # --- Step 3: learn disagreement threshold ---
    max_iou_disagreements = [
        disagreement_scores[d["image_path"]]["max_iou_disagreement"]
        for d in per_image_data
    ]
    detector = ConflictDetector.from_percentile(max_iou_disagreements, weights, percentile=75)
    print(f"\n  Disagreement threshold (75th percentile): {detector.disagreement_threshold:.4f}")

    # --- Step 4: consensus vs. best-individual-method comparison ---
    best_individual_ious = []
    mean_individual_ious = []
    consensus_ious = []
    flagged_flags = []

    for d in per_image_data:
        method_iou_values = list(d["method_ious"].values())
        best_iou = max(method_iou_values)
        mean_iou = np.mean(method_iou_values)  # fair, deployable baseline (no oracle knowledge)
        consensus_hm = detector.get_consensus(d["heatmaps"])
        consensus_iou = compute_mask_iou(consensus_hm, d["mask"])

        best_individual_ious.append(best_iou)
        mean_individual_ious.append(mean_iou)
        consensus_ious.append(consensus_iou)

        disagreement = disagreement_scores[d["image_path"]]["max_iou_disagreement"]
        flagged_flags.append(detector.detect_conflict(disagreement))

    best_individual_ious = np.array(best_individual_ious)
    mean_individual_ious = np.array(mean_individual_ious)
    consensus_ious = np.array(consensus_ious)

    mean_best = best_individual_ious.mean()
    mean_of_means = mean_individual_ious.mean()
    mean_consensus = consensus_ious.mean()
    n_consensus_beats_best = (consensus_ious > best_individual_ious).sum()
    n_consensus_beats_mean = (consensus_ious > mean_individual_ious).sum()
    n_total = len(consensus_ious)

    print(f"\n  --- Consensus vs. Best-Individual-Method (ORACLE baseline — unfair, for reference only) ---")
    print(f"  Mean best-individual-method IoU (oracle, picks best method per image): {mean_best:.4f}")
    print(f"  Mean consensus IoU:                                                     {mean_consensus:.4f}")
    print(f"  Consensus beats oracle-best on {n_consensus_beats_best}/{n_total} images "
          f"({100*n_consensus_beats_best/n_total:.1f}%)")
    if len(best_individual_ious) >= 5:
        stat, p_value = wilcoxon(consensus_ious, best_individual_ious)
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"  Wilcoxon signed-rank test: p={p_value:.4f} [{sig}]")

    print(f"\n  --- Consensus vs. Mean-Individual-Method (FAIR, deployable baseline) ---")
    print(f"  Mean of individual methods' IoU (no oracle knowledge): {mean_of_means:.4f}")
    print(f"  Mean consensus IoU:                                    {mean_consensus:.4f}")
    print(f"  Consensus beats the mean-method baseline on {n_consensus_beats_mean}/{n_total} images "
          f"({100*n_consensus_beats_mean/n_total:.1f}%)")
    if len(mean_individual_ious) >= 5:
        stat, p_value = wilcoxon(consensus_ious, mean_individual_ious)
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"  Wilcoxon signed-rank test: p={p_value:.4f} [{sig}]")

    # --- Step 5: do flagged (high-disagreement) images have worse localization? ---
    flagged_flags = np.array(flagged_flags)
    stats = detector.get_statistics()
    print(f"\n  --- Conflict Flag Validation ---")
    print(f"  Flagged {stats['flagged']}/{stats['total']} images ({100*stats['flag_rate']:.1f}%) as low-trust")

    if flagged_flags.sum() >= 3 and (~flagged_flags).sum() >= 3:
        flagged_ious = best_individual_ious[flagged_flags]
        unflagged_ious = best_individual_ious[~flagged_flags]
        u_stat, p_value = mannwhitneyu(flagged_ious, unflagged_ious, alternative='less')
        sig = "significant" if p_value < 0.05 else "not significant"
        print(f"  Mean best-method IoU (flagged, N={flagged_flags.sum()}):   {flagged_ious.mean():.4f}")
        print(f"  Mean best-method IoU (unflagged, N={(~flagged_flags).sum()}): {unflagged_ious.mean():.4f}")
        print(f"  Mann-Whitney U test (flagged < unflagged): p={p_value:.4f} [{sig}]")
    else:
        print("  Not enough flagged/unflagged images for a meaningful comparison.")

    return {
        "category": category,
        "architecture": architecture,
        "n_images": n_total,
        "weights": weights,
        "threshold": detector.disagreement_threshold,
        "mean_best_individual_iou": mean_best,
        "mean_of_individual_ious": mean_of_means,
        "mean_consensus_iou": mean_consensus,
        "consensus_beats_oracle_rate": n_consensus_beats_best / n_total,
        "consensus_beats_mean_rate": n_consensus_beats_mean / n_total,
        "flag_rate": stats['flag_rate'],
    }


def main():
    rows = load_manifest()
    disagreement_scores = load_per_image_scores()

    all_results = []
    for category in config.MVTEC_CATEGORIES:
        for architecture in ["resnet18", "vit"]:
            result = process_group(category, architecture, rows, disagreement_scores)
            all_results.append(result)

    # Save summary
    output_path = config.RESULTS_PATH / "metrics" / "phase4_conflict_detector_summary.csv"
    with open(output_path, "w", newline="") as f:
        fieldnames = ["category", "architecture", "n_images", "threshold",
                      "mean_best_individual_iou", "mean_of_individual_ious", "mean_consensus_iou",
                      "consensus_beats_oracle_rate", "consensus_beats_mean_rate", "flag_rate"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})

    print(f"\n{'='*70}")
    print(f"Phase 4 complete! Summary saved: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()