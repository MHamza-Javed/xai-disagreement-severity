"""
Conflict Detector Module (Phase 4)
- Flags predictions where XAI methods disagree significantly (max aggregation, per Phase 0.3)
- Builds a weighted consensus heatmap, weighted by each method's historical
  localization accuracy (mean IoU against ground-truth masks)
- Validated against ground-truth mask IoU: does the consensus beat the best
  single method at localizing the actual defect?
"""

import numpy as np

from src.metrics import binarize_heatmap, TOP_PERCENT_THRESHOLD


def compute_mask_iou(heatmap, mask, top_percent=TOP_PERCENT_THRESHOLD):
    """
    Compute IoU between a binarized heatmap and the ground-truth defect mask.
    This measures LOCALIZATION accuracy (heatmap vs. reality), distinct from
    the inter-method disagreement IoU used in Phase 3 (heatmap vs. heatmap).

    Args:
        heatmap (ndarray): [H, W] heatmap
        mask (ndarray): [H, W] ground-truth mask (grayscale, >127 = defect)
        top_percent (float): binarization threshold, locked in Phase 0.3

    Returns:
        iou (float): IoU in [0, 1]
    """
    binary_heatmap = binarize_heatmap(heatmap, method='top_percent', top_percent=top_percent)
    binary_mask = (mask > 127).astype(np.uint8)

    intersection = np.logical_and(binary_heatmap, binary_mask).sum()
    union = np.logical_or(binary_heatmap, binary_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def compute_method_weights(mask_ious_by_method):
    """
    Convert each method's mean mask-IoU (localization accuracy) into consensus weights.

    Args:
        mask_ious_by_method (dict): {method_name: [iou1, iou2, ...]} across the dataset

    Returns:
        dict: {method_name: weight}, weights sum to 1.0
    """
    mean_ious = {method: float(np.mean(ious)) for method, ious in mask_ious_by_method.items()}
    total = sum(mean_ious.values())

    if total <= 0:
        # Fallback: uniform weights if all methods score zero IoU (degenerate case)
        n = len(mean_ious)
        return {method: 1.0 / n for method in mean_ious}

    return {method: iou / total for method, iou in mean_ious.items()}


def generate_consensus_heatmap(heatmaps, weights):
    """
    Generate a weighted consensus heatmap from multiple methods.

    Args:
        heatmaps (dict): {'gradcam': hm, 'ig': hm, 'occlusion': hm, 'gradientsshap': hm}
        weights (dict): {method_name: weight}, from compute_method_weights

    Returns:
        consensus (ndarray): [H, W] weighted average heatmap, normalized to [0, 1]
    """
    consensus = np.zeros_like(list(heatmaps.values())[0], dtype=np.float64)
    for method, heatmap in heatmaps.items():
        consensus += weights.get(method, 0.0) * heatmap

    max_val = consensus.max()
    if max_val > 1e-8:
        consensus = consensus / max_val
    return consensus


class ConflictDetector:
    """
    Lightweight conflict detector for XAI disagreement.

    Flags predictions where XAI methods disagree significantly (using MAX
    pairwise disagreement, per the Phase 0.3 decision — mean is reserved for
    RQ1/RQ2 severity analysis, max is for this worst-case-conflict use case),
    and generates a weighted consensus heatmap for flagged and unflagged cases alike.
    """

    def __init__(self, disagreement_threshold, method_weights):
        """
        Args:
            disagreement_threshold (float): max-pairwise disagreement above this is flagged
            method_weights (dict): {method_name: weight} for consensus generation
        """
        self.disagreement_threshold = disagreement_threshold
        self.method_weights = method_weights
        self.flagged_count = 0
        self.total_count = 0

    @classmethod
    def from_percentile(cls, disagreement_scores, method_weights, percentile=75):
        """
        Build a ConflictDetector with a threshold learned as a percentile of
        observed disagreement scores (e.g., 75th percentile flags the top 25%
        most-disagreeing predictions as low-trust).
        """
        threshold = float(np.percentile(disagreement_scores, percentile))
        return cls(disagreement_threshold=threshold, method_weights=method_weights)

    def detect_conflict(self, disagreement_score):
        """Return True if this image's disagreement exceeds the threshold."""
        is_conflicted = disagreement_score > self.disagreement_threshold
        self.total_count += 1
        if is_conflicted:
            self.flagged_count += 1
        return is_conflicted

    def get_consensus(self, heatmaps):
        """Generate this detector's weighted consensus heatmap for one image."""
        return generate_consensus_heatmap(heatmaps, self.method_weights)

    def get_statistics(self):
        """Return flag-rate statistics."""
        if self.total_count == 0:
            return {'flagged': 0, 'total': 0, 'flag_rate': 0.0}
        return {
            'flagged': self.flagged_count,
            'total': self.total_count,
            'flag_rate': self.flagged_count / self.total_count,
        }