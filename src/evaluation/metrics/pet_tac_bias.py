"""
TAC bias metric.
"""

import numpy as np
import nibabel as nib
from .common import compute_region_auc

AORTA_LABEL = None      # TODO: set totalseg aorta label
BRAIN_LABEL_IDS = []    # TODO: set totalseg brain region label IDs


def compute_tac_bias(
    pred_path,
    gt_path,
    organ_seg_path,
    brain_seg_path,
    frame_durations,
    epsilon=1e-6,
):
    """
    MARE of integrated time-activity curve (TAC) AUC values across aorta and brain regions.

    NOTE: This metric is used only for the final testing phase. It requires dynamic
    (4D) PET reconstructions, which are not part of the provided dataset.

    Parameters
    ----------
    pred_path       : path to predicted 4D PET NIfTI
    gt_path         : path to ground-truth 4D PET NIfTI
    organ_seg_path  : pet-label/organ_seg.nii.gz  (for aorta)
    brain_seg_path  : pet-label/brain_seg.nii.gz  (for brain regions)
    frame_durations : list of frame durations in seconds
    """

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    organ_seg = nib.load(organ_seg_path).get_fdata()
    brain_seg = nib.load(brain_seg_path).get_fdata()

    assert pred.ndim == 4
    assert len(frame_durations) == pred.shape[-1]

    mare_values = []

    # Aorta
    aorta_mask = organ_seg == AORTA_LABEL
    if np.sum(aorta_mask) > 0:
        auc_pred = compute_region_auc(pred, aorta_mask, frame_durations)
        auc_gt = compute_region_auc(gt, aorta_mask, frame_durations)
        mare_values.append(np.abs(auc_pred - auc_gt) / (np.abs(auc_gt) + epsilon))

    # Brain regions
    for label_id in BRAIN_LABEL_IDS:
        region_mask = brain_seg == label_id
        if np.sum(region_mask) == 0:
            continue

        auc_pred = compute_region_auc(pred, region_mask, frame_durations)
        auc_gt = compute_region_auc(gt, region_mask, frame_durations)
        mare_values.append(np.abs(auc_pred - auc_gt) / (np.abs(auc_gt) + epsilon))

    if not mare_values:
        raise ValueError("No valid TAC regions found.")

    return np.mean(mare_values)