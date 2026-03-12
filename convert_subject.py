#!/usr/bin/env python3
"""
convert_subject.py — Stage 1: Assemble subject data into the shared layout.

Usage:
    python convert_subject.py <subject_id> <output_dir>

Example:
    python convert_subject.py sub-000 /depict/u/hinge/shared/bic-mac-data/sub-000-new
"""

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

# ---------------------------------------------------------------------------
# Source roots — edit if your paths differ
# ---------------------------------------------------------------------------
RAW    = Path("/depict/data/hedit/raw_temp")
DERIV  = Path("/depict/data/hedit/derivatives")
PRIVATE = Path("/depict/users/hinge/private")


def resample_to_ref(moving: nib.Nifti1Image, ref: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """Resample moving to ref voxel grid via affine transform (no resampling library needed).

    order=1  linear interpolation  (intensity images)
    order=0  nearest-neighbour     (label/segmentation images)
    """
    # Maps ref voxels → moving voxels
    T = np.linalg.inv(moving.affine) @ ref.affine
    data = np.asanyarray(moving.dataobj)
    resampled = affine_transform(
        data, T[:3, :3], offset=T[:3, 3],
        output_shape=ref.shape[:3], order=order, mode="constant", cval=0,
    )
    return nib.Nifti1Image(resampled, ref.affine)


def resample_to_ref_2d(moving: nib.Nifti1Image, ref: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """Resample a 2D-in-3D image (one singleton axis) to ref in-plane grid only.

    The singleton axis is detected automatically. Output has ref's shape in the two
    non-singleton axes and the original size (1) in the singleton axis.
    """
    data = np.asanyarray(moving.dataobj)
    thin_ax = int(np.argmin(data.shape[:3]))

    out_shape = list(ref.shape[:3])
    out_shape[thin_ax] = data.shape[thin_ax]

    T = np.linalg.inv(moving.affine) @ ref.affine
    resampled = affine_transform(
        data, T[:3, :3], offset=T[:3, 3],
        output_shape=tuple(out_shape), order=order, mode="constant", cval=0,
    )

    # Affine: use ref columns for in-plane axes, original column for the thin axis
    # so the physical position of the single slice is preserved.
    new_affine = ref.affine.copy()
    new_affine[:, thin_ax] = moving.affine[:, thin_ax]
    return nib.Nifti1Image(resampled, new_affine)


def anonymize_inplace(
    img_path: Path,
    face_mask: np.ndarray,
    block_mm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixelate the face region of img_path in-place.

    Returns (cube_mask, affine) where cube_mask is the expanded block mask (uint8)
    and affine is the image's affine matrix (for saving derived images).
    """
    img = nib.load(img_path)
    data = np.asanyarray(img.dataobj, dtype=float)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    block_vox = np.maximum(1, np.round(block_mm / spacing)).astype(int)
    bx, by, bz = block_vox
    sx, sy, sz = data.shape[:3]

    result = data.copy()
    cube_mask = np.zeros(data.shape[:3], dtype=np.uint8)
    for x in range(0, sx, bx):
        for y in range(0, sy, by):
            for z in range(0, sz, bz):
                sl = np.s_[x : x + bx, y : y + by, z : z + bz]
                if not face_mask[sl].any():
                    continue
                result[sl] = data[sl].mean()
                cube_mask[sl] = 1

    nib.save(nib.Nifti1Image(result, img.affine, img.header), img_path)
    return cube_mask, img.affine


def cp(src: Path, dst: Path) -> None:
    if not src.exists():
        sys.exit(f"ERROR  missing file: {src}")
    shutil.copy2(src, dst)
    print(f"  OK  {src.name}  ->  {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: assemble subject data into shared bic-mac layout."
    )
    parser.add_argument("subject", help="Subject ID, e.g. sub-000")
    parser.add_argument("output_dir", help="Output root, e.g. /depict/u/hinge/shared/bic-mac-data/sub-000-new")
    parser.add_argument("--debug", action="store_true",
                        help="Skip copying large .s sinogram files (mult_factors.s, add_factors.s)")
    args = parser.parse_args()

    sub   = args.subject
    debug = args.debug
    out = Path(args.output_dir)

    # Source prefixes
    raw_quadra  = RAW / sub / "ses-quadra"
    raw_vida    = RAW / sub / "ses-vida"
    raw_seg     = RAW / "derivatives" / "totalsegmentator" / sub / "ses-quadra" / "ct"
    raw_reg     = RAW / "derivatives" / "registration_matrices" / sub
    deriv_seg   = DERIV / "totalsegmentator" / sub / "anat"
    proc        = PRIVATE / sub / "processing"

    # Output subdirs
    ct_dir   = out / "ct-label"
    feat_dir = out / "features"
    pet_dir  = out / "pet-label"

    # -----------------------------------------------------------------------
    # Create output layout
    # -----------------------------------------------------------------------
    print(f"==> Creating output directories under {out}")
    for d in (ct_dir, feat_dir, pet_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # ct-label/
    # -----------------------------------------------------------------------
    print("\n==> ct-label/")
    cp(
        raw_quadra / "ct" / f"{sub}_ses-quadra_acq-LOWDOSE_ce-none_rec-ac_ct.nii.gz",
        ct_dir / "ct.nii.gz",
    )
    cp(
        raw_seg / f"{sub}_ses-quadra_acq-LOWDOSE_ce-none_rec-ac_seg-body_dseg.nii.gz",
        ct_dir / "body_seg.nii.gz",
    )
    cp(
        raw_seg / f"{sub}_ses-quadra_acq-LOWDOSE_ce-none_rec-ac_seg-total_dseg.nii.gz",
        ct_dir / "organ_seg.nii.gz",
    )
    cp(
        deriv_seg / f"{sub}_acq-LOWDOSE_ce-none_rec-ac_seg-face_dseg.nii.gz",
        ct_dir / "face_seg.nii.gz",
    )

    # -----------------------------------------------------------------------
    # features/ — PET & topogram
    # -----------------------------------------------------------------------
    print("\n==> features/ (PET / topogram)")
    cp(
        raw_quadra / "pet" / f"{sub}_ses-quadra_trc-18FFDG_rec-nacstatOSEM_pet.nii.gz",
        feat_dir / "nacpet.nii.gz",
    )
    cp(
        raw_quadra / "ct" / f"{sub}_ses-quadra_acq-TOPOGRAM_ce-none_rec-tr20f_Xray.nii.gz",
        feat_dir / "topogram.nii.gz",
    )

    # -----------------------------------------------------------------------
    # features/ — DIXON MRI (copy all chunks, rename with clean names)
    # -----------------------------------------------------------------------
    print("\n==> features/ (DIXON MRI)")
    vida_anat = raw_vida / "anat"
    dixon_files = sorted(vida_anat.glob(f"{sub}_ses-vida_task-breathhold_acq-DIXONbody*.nii.gz"))

    if not dixon_files:
        sys.exit(f"ERROR  no DIXON files found under {vida_anat}")
    else:
        chunk_re = re.compile(r"acq-DIXONbody(IN|OUT)(?:_chunk-(\d+))?_T1w\.nii\.gz$")
        for src in dixon_files:
            m = chunk_re.search(src.name)
            if not m:
                print(f"  SKIP (unrecognised pattern)  {src.name}")
                continue
            phase = "in" if m.group(1) == "IN" else "out"
            chunk = m.group(2)  # None for the combined (no chunk token)
            dst_name = (
                f"mri_chunk_{chunk}_{phase}_phase.nii.gz"
                if chunk is not None
                else f"mri_combined_{phase}_phase.nii.gz"
            )
            cp(src, feat_dir / dst_name)

    # MR face segmentation (chunk-0 IN-phase)
    cp(
        deriv_seg / f"{sub}_task-breathhold_acq-DIXONbodyIN_chunk-0_seg-face_mr_dseg.nii.gz",
        feat_dir / "face_seg.nii.gz",
    )

    # -----------------------------------------------------------------------
    # features/ — apply MR→PET-CT rigid registration to affine (no resampling)
    # -----------------------------------------------------------------------
    print("\n==> features/ (apply mr2petct affine)")
    reg_matrix = raw_reg / "mr2petct_body.txt"
    if not reg_matrix.exists():
        sys.exit(f"ERROR  missing registration matrix: {reg_matrix}")
    else:
        pre_aff = np.loadtxt(reg_matrix)
        mri_targets = sorted(feat_dir.glob("mri_*.nii.gz")) + [feat_dir / "face_seg.nii.gz"]
        for p in mri_targets:
            if not p.exists():
                continue
            img = nib.load(p)
            img = nib.Nifti1Image(np.asanyarray(img.dataobj), pre_aff @ img.affine, img.header)
            nib.save(img, p)
            print(f"  OK  affine updated  {p.name}")

    # -----------------------------------------------------------------------
    # pet-label/
    # -----------------------------------------------------------------------
    print("\n==> pet-label/")
    cp(proc / "mult_factors_forSTIR_noacf_SSRB.hs", pet_dir / "mult_factors.hs")
    cp(proc / "additive_term_SSRB.hs",              pet_dir / "add_factors.hs")
    pet_src = proc / "manual_acf_20.nii"
    if not pet_src.exists():
        sys.exit(f"ERROR  missing file: {pet_src}")
    pet_img = nib.load(pet_src)
    nib.save(pet_img, pet_dir / "pet.nii.gz")
    print(f"  OK  {pet_src.name}  ->  {pet_dir / 'pet.nii.gz'}")
    if debug:
        print("  SKIP (--debug)  mult_factors.s  add_factors.s")
    else:
        cp(proc / "mult_factors_forSTIR_noacf_SSRB.s", pet_dir / "mult_factors.s")
        cp(proc / "additive_term_noacf_SSRB.s",               pet_dir / "add_factors.s")

    # -----------------------------------------------------------------------
    # features/ — resample all features to CT grid
    # -----------------------------------------------------------------------
    print("\n==> features/ (resample to CT)")
    ct_ref_path = ct_dir / "ct.nii.gz"
    if not ct_ref_path.exists():
        sys.exit(f"ERROR  missing CT reference for resampling: {ct_ref_path}")
    ct_ref = nib.load(ct_ref_path)

    # (filename, interpolation order, 2D-aware)
    feat_targets: list[tuple[str, int, bool]] = [
        ("nacpet.nii.gz",   1, False),
        ("topogram.nii.gz", 1, True),   # singleton axis → 2D resampling
        ("face_seg.nii.gz", 0, False),  # label image → nearest neighbour
    ]
    feat_targets += [(p.name, 1, False) for p in sorted(feat_dir.glob("mri_*.nii.gz"))]

    for fname, order, is_2d in feat_targets:
        p = feat_dir / fname
        if not p.exists():
            sys.exit(f"ERROR  missing feature file for resampling: {p}")
        img = nib.load(p)
        resampled = resample_to_ref_2d(img, ct_ref, order) if is_2d else resample_to_ref(img, ct_ref, order)
        nib.save(resampled, p)
        print(f"  OK  resampled ({'2D' if is_2d else '3D'}, order={order})  {fname}")

    # -----------------------------------------------------------------------
    # Anonymize CT (ct-label/)
    # -----------------------------------------------------------------------
    print("\n==> Anonymizing CT (ct-label/)")
    ct_img_path      = ct_dir / "ct.nii.gz"
    ct_face_seg_path = ct_dir / "face_seg.nii.gz"
    for p in (ct_img_path, ct_face_seg_path):
        if not p.exists():
            sys.exit(f"ERROR  missing file for CT anonymization: {p}")

    face_mask = np.asanyarray(nib.load(ct_face_seg_path).dataobj)
    cube_mask, affine = anonymize_inplace(ct_img_path, face_mask)
    print(f"  OK  anonymized  ct.nii.gz  ({int(cube_mask.sum()):,} voxels)")

    # Overwrite face_seg with expanded cube mask
    nib.save(nib.Nifti1Image(cube_mask, affine), ct_face_seg_path)
    print(f"  OK  updated  face_seg.nii.gz")

    # Zero out face region in body_seg and organ_seg
    for seg_name in ("body_seg.nii.gz", "organ_seg.nii.gz"):
        seg_path = ct_dir / seg_name
        if not seg_path.exists():
            sys.exit(f"ERROR  missing file for CT anonymization: {seg_path}")
        seg_img = nib.load(seg_path)
        seg_data = np.asanyarray(seg_img.dataobj).copy()
        seg_data[cube_mask > 0] = 0
        nib.save(nib.Nifti1Image(seg_data, seg_img.affine, seg_img.header), seg_path)
        print(f"  OK  zeroed face region in  {seg_name}")

    # -----------------------------------------------------------------------
    # Resample defaced body_seg / organ_seg to PET space → pet-label/
    # -----------------------------------------------------------------------
    print("\n==> pet-label/ (seg masks resampled to PET space)")
    pet_ref_path = pet_dir / "pet.nii.gz"
    if not pet_ref_path.exists():
        sys.exit(f"ERROR  missing PET reference for seg resampling: {pet_ref_path}")
    pet_ref = nib.load(pet_ref_path)
    for seg_name in ("body_seg.nii.gz", "organ_seg.nii.gz"):
        seg_path = ct_dir / seg_name
        if not seg_path.exists():
            sys.exit(f"ERROR  missing seg for PET-space resampling: {seg_path}")
        resampled = resample_to_ref(nib.load(seg_path), pet_ref, order=0)
        nib.save(resampled, pet_dir / seg_name)
        print(f"  OK  resampled (order=0)  {seg_name}  ->  pet-label/")

    # -----------------------------------------------------------------------
    # Anonymize MRI (features/)  — nacpet and topogram are left untouched
    # -----------------------------------------------------------------------
    print("\n==> Anonymizing MRI (features/)")
    mri_face_seg_path = feat_dir / "face_seg.nii.gz"
    if not mri_face_seg_path.exists():
        sys.exit(f"ERROR  missing file for MRI anonymization: {mri_face_seg_path}")
    mri_paths = sorted(feat_dir.glob("mri_*.nii.gz"))
    if not mri_paths:
        sys.exit(f"ERROR  no mri_*.nii.gz files found in {feat_dir}")
    face_mask = np.asanyarray(nib.load(mri_face_seg_path).dataobj)
    union_mask = None
    ref_affine = None
    for mri_path in mri_paths:
        cube_mask, affine = anonymize_inplace(mri_path, face_mask)
        print(f"  OK  anonymized  {mri_path.name}  ({int(cube_mask.sum()):,} voxels)")
        union_mask = cube_mask if union_mask is None else (union_mask | cube_mask)
        ref_affine = affine

    # Overwrite face_seg with union of all expanded cube masks
    nib.save(nib.Nifti1Image(union_mask.astype(np.uint8), ref_affine), mri_face_seg_path)
    print(f"  OK  updated  features/face_seg.nii.gz")

    # -----------------------------------------------------------------------
    # features/metadata.json
    # -----------------------------------------------------------------------
    print("\n==> features/metadata.json")

    # --- participants.tsv ---
    participants_tsv = RAW / "participants.tsv"
    if not participants_tsv.exists():
        sys.exit(f"ERROR  missing file: {participants_tsv}")
    with participants_tsv.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        row = next((r for r in reader if r["participant_id"] == sub), None)
    if row is None:
        sys.exit(f"ERROR  subject {sub!r} not found in {participants_tsv}")

    # --- PET JSON sidecar ---
    pet_json = raw_quadra / "pet" / f"{sub}_ses-quadra_trc-18FFDG_rec-nacstatOSEM_pet.json"
    if not pet_json.exists():
        sys.exit(f"ERROR  missing file: {pet_json}")
    with pet_json.open() as f:
        pet_meta = json.load(f)
    if "InjectedRadioactivity" not in pet_meta:
        sys.exit(f"ERROR  'InjectedRadioactivity' not found in {pet_json}")

    metadata = {
        "sex":                    row["sex"],
        "age":                    int(row["age"]),
        "height":                 float(row["height"]),
        "weight":                 float(row["weight"]),
        "injected_radioactivity": pet_meta["InjectedRadioactivity"],
    }
    metadata_path = feat_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  OK  {metadata_path}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n==> Done.")


if __name__ == "__main__":
    main()
