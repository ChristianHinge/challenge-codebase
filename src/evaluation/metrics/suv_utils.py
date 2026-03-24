"""
SUV Utilities

Provides SUV normalization factor computation from patient weight and GT PET.
"""

import numpy as np


def compute_suv_factor(gt_pet_img, body_mask_img):
    """
    Compute SUV normalization factor from GT PET and body mask.

    Total activity is estimated by summing GT PET voxels × voxel volume.
    Body weight is estimated from body mask volume (density ≈ 1.0 g/mL → kg).

    Parameters
    ----------
    gt_pet_img : nibabel image
        Ground-truth PET NIfTI image (any activity unit).
    body_mask_img : nibabel image
        Body segmentation mask (binary or label > 0).

    Returns
    -------
    float
        Factor to multiply raw PET values by to obtain SUV.
        SUV = PET * factor
    """
    zooms = gt_pet_img.header.get_zooms()
    voxel_vol_mL = np.prod(zooms[:3]) / 1000.0  # mm^3 → mL
    total_activity = np.sum(gt_pet_img.get_fdata()) * voxel_vol_mL

    body_zooms = body_mask_img.header.get_zooms()
    body_voxel_vol_mL = np.prod(body_zooms[:3]) / 1000.0
    body_volume_mL = np.sum(body_mask_img.get_fdata() > 0) * body_voxel_vol_mL
    weight_kg = body_volume_mL / 1000.0  # mL → kg (density ≈ 1.0 g/mL)

    return weight_kg / total_activity


def suv_sanity_check(pet_suv, body_mask, name="PET"):
    """
    Debug helper to verify SUV magnitude (~1 inside body).
    """
    mean_suv = np.mean(pet_suv[body_mask])
    print(f"[DEBUG] {name} mean SUV (body): {mean_suv:.4f}")

    if mean_suv < 0.01 or mean_suv > 50:
        print(
            "[WARNING] SUV mean appears incorrect. "
            "Check PET units or normalization."
        )
