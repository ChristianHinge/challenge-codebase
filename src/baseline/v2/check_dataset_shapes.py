import os
import nibabel as nib
import numpy as np

DATA_DIR = "/depict/users/hinge/shared/bic-mac-data/hf_dataset/train"

FILES = {
    "pet": "features/nacpet.nii.gz",
    "topogram": "features/topogram.nii.gz",
    "mri_in": "features/mri_combined_in_phase.nii.gz",
    "mri_out": "features/mri_combined_out_phase.nii.gz",
    "ct": "ct-label/ct.nii.gz",
}


def load_info(path):

    img = nib.load(path)

    shape = img.shape
    spacing = img.header.get_zooms()

    affine = img.affine
    det = np.linalg.det(affine[:3, :3])

    return shape, spacing, det


def main():

    subjects = sorted(os.listdir(DATA_DIR))

    print("\nChecking dataset:\n")

    for sub in subjects:

        print("=" * 70)
        print("Subject:", sub)

        base = os.path.join(DATA_DIR, sub)

        for key, relpath in FILES.items():

            path = os.path.join(base, relpath)

            if not os.path.exists(path):
                print(f"{key:10s} MISSING")
                continue

            try:

                shape, spacing, det = load_info(path)

                print(
                    f"{key:10s} | shape={str(shape):20s} "
                    f"| spacing={str(spacing):20s} "
                    f"| det={det:.4f}"
                )

                if det == 0:
                    print("  ⚠ WARNING: affine determinant is ZERO")

            except Exception as e:

                print(f"{key:10s} ERROR: {e}")

        print()

    print("\nDone.\n")


if __name__ == "__main__":
    main()