#!/usr/bin/env python3
"""
CT Face Anonymizer — Volumetric (NIfTI)
----------------------------------------
Pixelates the face region in a NIfTI CT image using 3-D blocks defined
in millimetres. Block size in voxels is derived per-axis from the voxel
spacing in the NIfTI header, so the cubes are isotropic in mm-space.

Any cube that touches the mask at all is fully replaced with the mean HU
of all voxels in that cube. The expanded cube mask is also saved.

Usage:
  python anonymize_ct.py <ct.nii.gz> <mask.nii.gz> [options]

Examples:
  python anonymize_ct.py scan.nii.gz face_mask.nii.gz
  python anonymize_ct.py scan.nii.gz face_mask.nii.gz --block-mm 20 -o anon.nii.gz

Outputs:
  <stem>_anon.nii.gz       — anonymized CT
  <stem>_anon_mask.nii.gz  — expanded cube mask (1 where pixelated, 0 elsewhere)

Dependencies:
  pip install nibabel numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel is required:  pip install nibabel")


def voxel_spacing(img) -> np.ndarray:
    """Return voxel size in mm for the first three axes."""
    zooms = np.array(img.header.get_zooms()[:3], dtype=float)
    if np.any(zooms <= 0):
        sys.exit("Could not read valid voxel spacing from NIfTI header.")
    return zooms


def mm_to_voxels(block_mm: float, spacing: np.ndarray) -> np.ndarray:
    """Convert a block size in mm to per-axis voxel counts (minimum 1)."""
    counts = np.maximum(1, np.round(block_mm / spacing)).astype(int)
    return counts


def pixelate_volume(ct: np.ndarray, mask: np.ndarray, block_vox: np.ndarray):
    """
    For every block that contains at least one masked voxel, replace ALL
    voxels in that block with the block's mean HU value.

    Returns:
        result    — anonymized CT array
        cube_mask — binary array marking every pixelated block (uint8)
    """
    result    = ct.copy()
    cube_mask = np.zeros(ct.shape, dtype=np.uint8)
    sx, sy, sz = ct.shape
    bx, by, bz = block_vox

    for x in range(0, sx, bx):
        for y in range(0, sy, by):
            for z in range(0, sz, bz):
                sl = np.s_[x:x + bx, y:y + by, z:z + bz]
                if not mask[sl].any():
                    continue
                mean_val   = ct[sl].mean()
                result[sl] = mean_val
                cube_mask[sl] = 1

    return result, cube_mask


def resolve_paths(ct_path: Path, output_arg: str | None):
    stem = ct_path.name.replace(".nii.gz", "").replace(".nii", "")
    base = ct_path.parent

    if output_arg:
        ct_out    = Path(output_arg)
        mask_stem = ct_out.name.replace(".nii.gz", "").replace(".nii", "")
        mask_out  = ct_out.parent / f"{mask_stem}_mask.nii.gz"
    else:
        ct_out   = base / f"{stem}_anon.nii.gz"
        mask_out = base / f"{stem}_anon_mask.nii.gz"

    return ct_out, mask_out


def main():
    parser = argparse.ArgumentParser(
        description="Volumetrically pixelate the face region in a NIfTI CT for anonymization."
    )
    parser.add_argument("ct_image",  help="Input CT image (.nii or .nii.gz)")
    parser.add_argument("face_mask", help="Face mask NIfTI (.nii or .nii.gz), non-zero = face region")
    parser.add_argument("--block-mm", type=float, default=10.0,
                        help="Isotropic block size in mm (default: 10.0)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CT path (default: <input_stem>_anon.nii.gz)")
    args = parser.parse_args()

    ct_path   = Path(args.ct_image)
    mask_path = Path(args.face_mask)

    if not ct_path.exists():
        sys.exit(f"CT image not found: {ct_path}")
    if not mask_path.exists():
        sys.exit(f"Face mask not found: {mask_path}")

    ct_out, mask_out = resolve_paths(ct_path, args.output)

    ct_img   = nib.load(ct_path)
    mask_img = nib.load(mask_path)

    ct_data   = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    if ct_data.shape != mask_data.shape:
        sys.exit(f"Shape mismatch: CT {ct_data.shape} vs mask {mask_data.shape}")

    spacing   = voxel_spacing(ct_img)
    block_vox = mm_to_voxels(args.block_mm, spacing)

    print(f"CT image   : {ct_path}")
    print(f"Face mask  : {mask_path}")
    print(f"Voxel size : {spacing[0]:.2f} x {spacing[1]:.2f} x {spacing[2]:.2f} mm")
    print(f"Block size : {args.block_mm} mm  →  {block_vox[0]} x {block_vox[1]} x {block_vox[2]} voxels")
    print(f"Volume     : {ct_data.shape}")
    print(f"CT output  : {ct_out}")
    print(f"Mask output: {mask_out}")

    result, cube_mask = pixelate_volume(ct_data, mask_data, block_vox)

    n_voxels = int(cube_mask.sum())
    print(f"Pixelated  : {n_voxels:,} voxels anonymized")

    nib.save(nib.Nifti1Image(result,    ct_img.affine, ct_img.header), ct_out)
    nib.save(nib.Nifti1Image(cube_mask, ct_img.affine, ct_img.header), mask_out)

    print(f"Saved CT   : {ct_out}")
    print(f"Saved mask : {mask_out}")


if __name__ == "__main__":
    main()
