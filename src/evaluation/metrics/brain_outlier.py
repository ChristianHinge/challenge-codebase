"""
Brain outlier robustness metric.
"""

import numpy as np
from .common import compute_k_value, compute_auc_of_K
import nibabel as nib

def compute_brain_outlier_score(pred_paths, gt_paths, totalseg_paths):

    thresholds = [0.05, 0.10, 0.15]
    auc_scores = []

    for threshold in thresholds:
        k_values = []

        for pred, gt, seg in zip(pred_paths, gt_paths, totalseg_paths):

            k = compute_k_value(pred, gt, seg, threshold=threshold)

            k_values.append(k)

        auc_scores.append(compute_auc_of_K(k_values))

    return np.mean(auc_scores)