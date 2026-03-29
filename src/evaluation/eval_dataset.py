import argparse
import os

import numpy as np
from tqdm import tqdm 

try:
    from .eval_subject import evaluate_subject
    from .metrics import compute_brain_outlier_score
except ImportError:
    from eval_subject import evaluate_subject
    from metrics import compute_brain_outlier_score


def validate_pred_structure(pred_dir, subjects):
    """
    Validate that all subjects have a consistent prediction structure.
    Allowed layouts:
      (A) pet.nii.gz only
      (B) ct.nii.gz only
      (C) pet.nii.gz + ct.nii.gz

    Raises ValueError if subjects are inconsistent (some have pet, some don't).
    Prints found vs. expected structure on failure.
    """
    has_pet = {s: os.path.exists(os.path.join(pred_dir, s, "pet.nii.gz")) for s in subjects}
    has_ct  = {s: os.path.exists(os.path.join(pred_dir, s, "ct.nii.gz"))  for s in subjects}

    for label, found in (("pet.nii.gz", has_pet), ("ct.nii.gz", has_ct)):
        present = [s for s, h in found.items() if h]
        missing = [s for s, h in found.items() if not h]
        if present and missing:
            raise ValueError(
                f"Inconsistent predictions: {label} present for {present} but missing for {missing}.\n"
                f"\n"
                f"Expected one of these layouts (must be consistent across all subjects):\n"
                f"\n"
                f"  (A) PET only          (B) CT only           (C) PET + CT\n"
                f"  pred_dir/             pred_dir/             pred_dir/\n"
                f"    sub-000/              sub-000/              sub-000/\n"
                f"      pet.nii.gz            ct.nii.gz             pet.nii.gz\n"
                f"    sub-001/              sub-001/              sub-001/ ...  ct.nii.gz\n"
                f"      pet.nii.gz ...        ct.nii.gz ...\n"
            )

    return {s: (has_pet[s], has_ct[s]) for s in subjects}


def evaluate_dataset(dataset_path, pred_dir, subjects=None, quiet=False):
    """
    Evaluate predictions across multiple subjects.

    Parameters
    ----------
    dataset_path : str
        Root directory containing subject folders with ground-truth labels.
    pred_dir : str
        Directory containing one sub-folder per subject with ct.nii.gz and pet.nii.gz.
    subjects : list of str, optional
        Subject IDs to evaluate. Defaults to all sub-folders in pred_dir.

    Returns
    -------
    dict
        Aggregate scores: CT MAE, Whole-body SUV MAE, Brain Outlier Score, Organ Bias.
    """

    if subjects is None:
        subjects = sorted(
            d for d in os.listdir(pred_dir)
            if os.path.isdir(os.path.join(pred_dir, d))
        )

    if not subjects:
        raise ValueError(f"No subject folders found in {pred_dir}")

    if not quiet:
        print(f"Evaluating {len(subjects)} subject(s): {subjects}\n")

    pred_layout    = validate_pred_structure(pred_dir, subjects)
    per_subject    = {}
    pred_pet_paths = []
    gt_pet_paths   = []
    organ_seg_paths = []

    for subject_id, (with_pet, with_ct) in tqdm(pred_layout.items(), desc="Evaluating subjects"):
        subject_path = os.path.join(dataset_path, subject_id)
        pred_pet = os.path.join(pred_dir, subject_id, "pet.nii.gz") if with_pet else None
        pred_ct  = os.path.join(pred_dir, subject_id, "ct.nii.gz")  if with_ct  else None

        per_subject[subject_id] = evaluate_subject(subject_path, pred_pet, pred_ct, quiet=True)

        if with_pet:
            pred_pet_paths.append(pred_pet)
            gt_pet_paths.append(os.path.join(subject_path, "pet-label", "pet.nii.gz"))
            organ_seg_paths.append(os.path.join(subject_path, "pet-label", "organ_seg.nii.gz"))

    # Brain outlier — dataset-level, computed jointly across all subjects
    brain_outlier = compute_brain_outlier_score(
        pred_paths=pred_pet_paths,
        gt_paths=gt_pet_paths,
        organ_seg_paths=organ_seg_paths,
    ) if pred_pet_paths else float("nan")

    all_results = list(per_subject.values())
    aggregate = {key: float(np.mean([r[key] for r in all_results])) for key in all_results[0]}
    if pred_pet_paths:
        aggregate["pet_brain_outlier_score"] = float(brain_outlier)

    DISPLAY_NAMES = {
        "ct_mu_map_mae":           "[CT] Mu-map MAE",
        "pet_whole_body_suv_mae":  "[PET] Whole-body SUV MAE",
        "pet_brain_outlier_score": "[PET] Brain Outlier Score",
        "pet_organ_bias":          "[PET] Organ Bias",
    }
    UNITS = {"pet_organ_bias": "%"}

    if not quiet:
        print("\n================ Aggregate Results ================")
        for key, value in aggregate.items():
            label = DISPLAY_NAMES.get(key, key)
            unit  = UNITS.get(key, "")
            print(f"  {label} ↓{'':<{34 - len(label)}}: {value:.6f}{unit}")
        print("====================================================\n")

    return aggregate


def main():

    parser = argparse.ArgumentParser(
        description="BIC-MAC Dataset-level Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Expected directory structures:\n"
            "\n"
            "  --dataset_path                      --pred_dir\n"
            "  (BIC-MAC split, e.g. train/)        (your predictions)\n"
            "  bic-mac-data/train/                 pred_dir/\n"
            "  ├── sub-000/                        ├── sub-000/\n"
            "  │   ├── ct-label/                   │   ├── ct.nii.gz\n"
            "  │   ├── pet-label/                  │   └── pet.nii.gz (optional)\n"
            "  │   └── ...                         └── sub-001/ ...\n"
            "  └── sub-001/ ...\n"
            "\n"
            "  Each subject folder in pred_dir may contain ct.nii.gz only or ct.nii.gz+pet.nii.gz\n"
            "  but the choice must be consistent across all subjects.\n"
        ),
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to the downloaded BIC-MAC dataset split, e.g. bic-mac-data/train or bic-mac-data/val",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Root with predicted subject folders (see structure below)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit list of subject IDs to evaluate (default: all sub-folders in pred_dir)",
    )
    args = parser.parse_args()

    evaluate_dataset(args.dataset_dir, args.pred_dir, args.subjects)


if __name__ == "__main__":
    main()
