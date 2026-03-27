"""
Brain outlier robustness metric.
"""

import numpy as np
from .common import compute_k_value, compute_auc_of_K
import nibabel as nib

def compute_brain_outlier_score(pred_paths, gt_paths, organ_seg_paths):
    """
    Dataset-level brain outlier robustness score (1 - AUC of K(t)) across subjects.

    For each subject, computes the fraction of brain voxels within a relative error
    threshold t, then integrates the curve over t ∈ {5%, 10%, 15%}. A score of 0
    means perfect predictions; higher is worse.

    Parameters
    ----------
    pred_paths      : list of paths to predicted PET NIfTIs
    gt_paths        : list of pet-label/pet.nii.gz paths
    organ_seg_paths : list of pet-label/organ_seg.nii.gz paths
    """

    thresholds = [0.05, 0.10, 0.15]
    auc_scores = []

    for threshold in thresholds:
        k_values = []

        for pred, gt, seg in zip(pred_paths, gt_paths, organ_seg_paths):

            k = compute_k_value(pred, gt, seg, threshold=threshold)

            k_values.append(k)

        auc_scores.append(compute_auc_of_K(k_values))

    return 1-np.mean(auc_scores)