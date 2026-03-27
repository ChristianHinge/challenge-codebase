"""
Organ bias metric (SUV-mean MARE).
"""

import numpy as np
import nibabel as nib
from .suv_utils import compute_suv_factor


def compute_organ_bias(
    pred_path,
    gt_path,
    organ_seg_path,
    tissue_seg_path,
    body_seg_path,
    epsilon=1e-6,
):
    """
    Mean absolute relative error (MARE) of SUV-mean across 8 regions.

    Regions and their segmentation source:
      organ_seg  : brain (90), liver (5), spleen (1), heart (51), pancreas (7)
      tissue_seg : muscle (3), adipose (1+2)
      body_seg   : extremities (2)

    Parameters
    ----------
    pred_path       : path to predicted PET NIfTI
    gt_path         : pet-label/pet.nii.gz
    organ_seg_path  : pet-label/organ_seg.nii.gz
    tissue_seg_path : pet-label/tissue_seg.nii.gz
    body_seg_path   : pet-label/body_seg.nii.gz
    """

    gt_img = nib.load(gt_path)
    suv_factor = compute_suv_factor(gt_img, nib.load(body_seg_path))

    pred = nib.load(pred_path).get_fdata() * suv_factor
    gt = gt_img.get_fdata() * suv_factor

    organ_seg   = nib.load(organ_seg_path).get_fdata()
    tissue_seg  = nib.load(tissue_seg_path).get_fdata()
    body_seg    = nib.load(body_seg_path).get_fdata()

    organ_masks = {
        "brain":       organ_seg == 90,
        "liver":       organ_seg == 5,
        "spleen":      organ_seg == 1,
        "heart":       organ_seg == 51,
        "pancreas":    organ_seg == 7,
        "muscle":      tissue_seg == 3,
        "adipose":     (tissue_seg == 1) | (tissue_seg == 2),
        "extremities": body_seg == 2,
    }

    mare_values = []

    for name, mask in organ_masks.items():
        assert np.sum(mask) > 0, f"Mask for {name} is empty"

        pred_mean = np.mean(pred[mask])
        gt_mean = np.mean(gt[mask])

        mare = 100 * np.abs(pred_mean - gt_mean) / (np.abs(gt_mean) + epsilon)
        mare_values.append(mare)

    if not mare_values:
        raise ValueError("No valid organs found.")

    return np.mean(mare_values)
