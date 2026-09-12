"""
Disagreement Metrics & Correlation Analysis
- Spatial IoU between heatmap pairs (top-20% intensity binarization)
- Pixel-level rank correlation (Spearman's rho)
- Pairwise aggregation: mean (primary, RQ1/RQ2) and max (reserved for Phase 4 Conflict Detector)
- Defect severity quantification (normalized mask area)
"""

import numpy as np
from scipy.stats import spearmanr
import cv2

TOP_PERCENT_THRESHOLD = 0.20  # locked in Phase 0.3


def binarize_heatmap(heatmap, method='top_percent', threshold=None, top_percent=TOP_PERCENT_THRESHOLD):
    """
    Binarize a heatmap for IoU computation.

    Args:
        heatmap (ndarray): [H, W] heatmap
        method (str): 'top_percent' (default, locked Phase 0.3), 'otsu', or 'threshold'
        threshold (float): Threshold value (only used for 'threshold' method)
        top_percent (float): Fraction of highest-activation pixels to keep (default 0.20)

    Returns:
        binary_map (ndarray): [H, W] binary mask
    """
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    if method == 'top_percent':
        # Keep the top-% highest-activation pixels — constant "attention budget"
        # across methods, regardless of each method's raw value scale/distribution
        cutoff = np.percentile(heatmap, (1 - top_percent) * 100)
        binary = (heatmap >= cutoff).astype(np.uint8)
    elif method == 'otsu':
        _, binary = cv2.threshold((heatmap * 255).astype(np.uint8), 0, 1, cv2.THRESH_OTSU)
    elif method == 'threshold':
        binary = (heatmap > threshold).astype(np.uint8)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


def compute_iou(heatmap1, heatmap2, method='top_percent', top_percent=TOP_PERCENT_THRESHOLD):
    """
    Compute Spatial Intersection-over-Union between two heatmaps.

    Returns:
        iou (float): IoU score in [0, 1]
    """
    binary1 = binarize_heatmap(heatmap1, method=method, top_percent=top_percent)
    binary2 = binarize_heatmap(heatmap2, method=method, top_percent=top_percent)

    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def compute_spearman_correlation(heatmap1, heatmap2):
    """
    Compute pixel-level rank correlation (Spearman's rho) — flattened heatmap
    correlation, locked in Phase 0.3.

    Returns:
        rho (float): Spearman's rho correlation coefficient
    """
    flat1 = heatmap1.flatten()
    flat2 = heatmap2.flatten()
    rho, _ = spearmanr(flat1, flat2)
    return rho


def compute_all_pairwise_scores(heatmaps, top_percent=TOP_PERCENT_THRESHOLD):
    """
    Compute IoU and Spearman's rho disagreement for all 6 method-pairs.

    Args:
        heatmaps (dict): {'gradcam': hm, 'ig': hm, 'occlusion': hm, 'gradientsshap': hm}

    Returns:
        dict: {
            'iou_disagreements': {pair_name: 1-iou, ...},   # 6 entries
            'spearman_disagreements': {pair_name: 1-rho, ...},  # 6 entries
        }
    """
    methods = list(heatmaps.keys())
    iou_disagreements = {}
    spearman_disagreements = {}

    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            pair_name = f"{m1}_{m2}"
            iou = compute_iou(heatmaps[m1], heatmaps[m2], top_percent=top_percent)
            rho = compute_spearman_correlation(heatmaps[m1], heatmaps[m2])
            iou_disagreements[pair_name] = 1 - iou
            spearman_disagreements[pair_name] = 1 - rho

    return {
        'iou_disagreements': iou_disagreements,
        'spearman_disagreements': spearman_disagreements,
    }


def aggregate_disagreement(pairwise_scores, agg='mean'):
    """
    Aggregate a dict of pairwise disagreement scores into one value.

    Args:
        pairwise_scores (dict): {pair_name: score, ...}
        agg (str): 'mean' (primary, RQ1/RQ2) or 'max' (Phase 4 Conflict Detector)

    Returns:
        float: aggregated disagreement score
    """
    values = list(pairwise_scores.values())
    if agg == 'mean':
        return float(np.mean(values))
    elif agg == 'max':
        return float(np.max(values))
    else:
        raise ValueError(f"Unknown aggregation: {agg}")


def compute_defect_severity(mask):
    """
    Compute defect severity as normalized mask area.

    Args:
        mask (ndarray): [H, W] mask (grayscale or binary)

    Returns:
        severity (float): Normalized defect area in [0, 1]
    """
    total_pixels = mask.size
    defect_pixels = np.sum(mask > 127)
    return defect_pixels / total_pixels


def correlation_analysis(disagreements, severities):
    """
    Run correlation analysis: disagreement vs. defect severity.

    Args:
        disagreements (list or ndarray): Per-image disagreement scores
        severities (list or ndarray): Per-image severity scores

    Returns:
        rho (float): Spearman's rho
        p_value (float): Significance p-value
    """
    rho, p_value = spearmanr(disagreements, severities)
    return rho, p_value