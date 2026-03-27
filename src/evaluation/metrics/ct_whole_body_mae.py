"""
Whole-body CT MAE metric.
"""

import numpy as np
import nibabel as nib

import numpy as np
import nibabel as nib

def hu_to_mu(ct_path, kvp=120):
    """Carney et al. 2006 (Med Phys 33:976-983) bilinear HU to mu at 511 keV."""
    bone_slope = {80: 3.84e-5, 100: 4.56e-5, 120: 5.10e-5, 140: 5.64e-5}

    ct = nib.load(ct_path)
    hu = ct.get_fdata(dtype=np.float32)

    mu = np.where(hu <= 0,
                  9.6e-5 * (hu + 1000),
                  9.6e-5 * 1000 + bone_slope[kvp] * hu)
    mu = np.clip(mu, 0, None)

    return nib.Nifti1Image(mu, ct.affine, ct.header)


LIVER_LABEL = 5


def compute_whole_body_mu_mae(
    pred_ct_path,
    gt_ct_path,
    body_seg_path,
    organ_seg_path,
    exclusion_cm=4.0,
    save_mask_path=None,
):
    """
    Voxel-wise MAE of linear attenuation coefficient (mu, cm⁻¹) across the body,
    excluding ±4 cm around the superior liver slice to reduce sensitivity to
    respiratory misalignment. CT images are converted from HU → mu using the
    Carney et al. 2006 bilinear model at 511 keV (see hu_to_mu()).

    Parameters
    ----------
    pred_ct_path   : path to predicted CT NIfTI (in HU)
    gt_ct_path     : ct-label/ct.nii.gz
    body_seg_path  : ct-label/body_seg.nii.gz
    organ_seg_path : ct-label/organ_seg.nii.gz
    save_mask_path : optional path to save the evaluation mask as NIfTI
    """

    # Convert CT → mu
    pred = hu_to_mu(pred_ct_path).get_fdata()
    gt = hu_to_mu(gt_ct_path).get_fdata()

    body_mask = nib.load(body_seg_path).get_fdata() > 0
    liver_mask = nib.load(organ_seg_path).get_fdata() == LIVER_LABEL

    # Slice thickness
    slice_thickness_mm = nib.load(pred_ct_path).header.get_zooms()[2]
    exclusion_slices = int(round((exclusion_cm * 10.0) / slice_thickness_mm))

    # Superior liver slice
    superior_slice = np.max(np.where(liver_mask)[2])

    z_min = max(0, superior_slice - exclusion_slices)
    z_max = min(pred.shape[2], superior_slice + exclusion_slices)

    exclusion_mask = np.zeros_like(body_mask, dtype=bool)
    exclusion_mask[:, :, z_min:z_max] = True

    eval_mask = body_mask & (~exclusion_mask)

    if save_mask_path is not None:
        nib.save(nib.Nifti1Image(eval_mask.astype(np.uint8), nib.load(body_seg_path).affine), save_mask_path)

    return np.mean(np.abs(pred - gt)[eval_mask])
