"""
Whole-body SUV MAE metric.
"""

import json
import numpy as np
import nibabel as nib
from .suv_utils import compute_suv_factor, suv_sanity_check


def compute_whole_body_suv_mae(
    pred_pet_path,
    gt_pet_path,
    body_mask_path,
    liver_mask_path,
    json_path,
    exclusion_cm=4.0,
    debug=False,
):
    """
    Compute voxel-wise MAE of SUV inside body,
    excluding ±4 cm around superior liver slice.
    """

    with open(json_path, "r") as f:
        meta = json.load(f)

    weight_kg = meta["weight"]

    gt_img = nib.load(gt_pet_path)
    suv_factor = compute_suv_factor(weight_kg, gt_img)

    pred = nib.load(pred_pet_path).get_fdata() * suv_factor
    gt = gt_img.get_fdata() * suv_factor

    body_mask = nib.load(body_mask_path).get_fdata() > 0
    liver_mask = nib.load(liver_mask_path).get_fdata() > 0

    if debug:
        suv_sanity_check(pred, body_mask, "Prediction")
        suv_sanity_check(gt, body_mask, "Ground Truth")

    slice_thickness_mm = nib.load(pred_pet_path).header.get_zooms()[2]
    exclusion_slices = int(round((exclusion_cm * 10.0) / slice_thickness_mm))

    superior_slice = np.max(np.where(liver_mask)[2])

    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(pred.shape[2], superior_slice + exclusion_slices)

    exclusion_mask = np.zeros_like(body_mask, dtype=bool)
    exclusion_mask[:, :, z_min:z_max] = True

    eval_mask = body_mask & (~exclusion_mask)

    return np.mean(np.abs(pred - gt)[eval_mask])
