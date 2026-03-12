"""
SUV Utilities

Provides SUV normalization factor computation from patient weight and GT PET.
"""

import numpy as np


def compute_suv_factor(weight_kg, gt_pet_img):
    """
    Compute SUV normalization factor from patient weight and GT PET image.

    Total dose is estimated by summing GT PET voxels multiplied by voxel volume,
    avoiding the need for InjectedRadioactivity in metadata.

    Parameters
    ----------
    weight_kg : float
        Patient weight in kg.
    gt_pet_img : nibabel image
        Ground-truth PET NIfTI image (any activity unit).

    Returns
    -------
    float
        Factor to multiply raw PET values by to obtain SUV.
        SUV = PET * factor
    """
    zooms = gt_pet_img.header.get_zooms()
    voxel_vol_mL = np.prod(zooms[:3]) / 1000.0  # mm^3 → mL
    total_activity = np.sum(gt_pet_img.get_fdata()) * voxel_vol_mL
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
