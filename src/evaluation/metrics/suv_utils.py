"""
SUV Utilities

Provides SUV normalization factor computation from patient weight and GT PET.

NOTE: Since the units of the STIR reconstructed PET images are not Bq/mL,
we cannot directly convert to SUV using the recorded injected dose and recorded subject weight.
Instead, we estimate the injected dose by integrating the ground-truth PET image itself,
and the subject weight by integrating the body mask.
The suv_factor is then computed as (estimated subject weight in g) / (estimated injected dose),
and applied to both the predicted and ground-truth PET images before computing any metric.
"""

import numpy as np


def compute_suv_factor(gt_pet_img, body_seg_img):
    """
    Compute SUV normalization factor from GT PET and body mask.

    Total activity is estimated by summing GT PET voxels × voxel volume.
    Body weight is estimated from body mask volume (density ≈ 1.0 g/mL → g).

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

    body_zooms = body_seg_img.header.get_zooms()
    body_voxel_vol_mL = np.prod(body_zooms[:3]) / 1000.0
    body_volume_mL = np.sum(body_seg_img.get_fdata() > 0) * body_voxel_vol_mL
    weight_g = body_volume_mL * 1.0  # mL → g (density ≈ 1.0 g/mL)

    return 1.0 / (total_activity / weight_g)

