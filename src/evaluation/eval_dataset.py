"""
Dataset-level Evaluation Script

Evaluates predictions across multiple subjects and reports per-subject
and aggregate scores, matching the challenge leaderboard computation.

Brain Outlier Score is computed jointly across all subjects (as in the
challenge), not averaged from per-subject values.

Usage:
    python eval_dataset.py --dataset_path <dir> --pred_dir <dir>

    <dataset_path>  root directory containing subject folders (e.g. train/)
                    each subject must have ct-label/ and pet-label/ subdirs
    <pred_dir>      directory containing one folder per subject, each with
                    ct.nii.gz and pet.nii.gz

Example:
    python eval_dataset.py \\
        --dataset_path /data/bic-mac/train \\
        --pred_dir /results/my_method
"""

import argparse
import os

import numpy as np

from .eval_case import evaluate_case
from .metrics import compute_brain_outlier_score


def eval_dataset(dataset_path, pred_dir, subjects=None):
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

    print(f"Evaluating {len(subjects)} subject(s): {subjects}\n")

    per_subject     = {}
    pred_pet_paths  = []
    gt_pet_paths    = []
    organ_seg_paths = []

    for subject_id in subjects:
        subject_path = os.path.join(dataset_path, subject_id)
        pred_pet     = os.path.join(pred_dir, subject_id, "pet.nii.gz")
        pred_ct      = os.path.join(pred_dir, subject_id, "ct.nii.gz")

        results = evaluate_case(subject_path, pred_pet, pred_ct)
        per_subject[subject_id] = results

        print(f"  {subject_id}")
        for name, value in results.items():
            unit = "%" if name == "Organ Bias" else ""
            print(f"    {name:<25}: {value:.6f}{unit}")

        pred_pet_paths.append(pred_pet)
        gt_pet_paths.append(os.path.join(subject_path, "pet-label", "pet.nii.gz"))
        organ_seg_paths.append(os.path.join(subject_path, "pet-label", "organ_seg.nii.gz"))

    # Brain outlier — dataset-level, computed jointly across all subjects
    brain_outlier = compute_brain_outlier_score(
        pred_paths=pred_pet_paths,
        gt_paths=gt_pet_paths,
        totalseg_paths=organ_seg_paths,
    )

    all_results = list(per_subject.values())
    aggregate = {
        "CT MAE":              float(np.mean([r["CT MAE"]             for r in all_results])),
        "Whole-body SUV MAE":  float(np.mean([r["Whole-body SUV MAE"] for r in all_results])),
        "Brain Outlier Score": float(brain_outlier),
        "Organ Bias":          float(np.mean([r["Organ Bias"]         for r in all_results])),
    }

    print("\n================ Aggregate Results ================")
    for name, value in aggregate.items():
        unit = "%" if name == "Organ Bias" else ""
        print(f"  {name:<25}: {value:.6f}{unit}")
    print("====================================================\n")

    return aggregate


def main():

    parser = argparse.ArgumentParser(
        description="BIC-MAC Dataset-level Evaluation"
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Root directory containing subject folders with ground-truth labels",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing one sub-folder per subject with ct.nii.gz and pet.nii.gz",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Explicit list of subject IDs to evaluate (default: all sub-folders in pred_dir)",
    )
    args = parser.parse_args()

    eval_dataset(args.dataset_path, args.pred_dir, args.subjects)


if __name__ == "__main__":
    main()
