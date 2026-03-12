"""
Whole-body CT MAE metric.
"""

import numpy as np
import nibabel as nib


def compute_whole_body_ct_mae(pred_ct_path, gt_ct_path, body_mask_path):
    """
    Compute voxel-wise MAE in HU between predicted and ground-truth CT,
    restricted to the body mask.
    """

    pred = nib.load(pred_ct_path).get_fdata()
    gt = nib.load(gt_ct_path).get_fdata()
    body_mask = nib.load(body_mask_path).get_fdata() > 0

    return np.mean(np.abs(pred - gt)[body_mask])
