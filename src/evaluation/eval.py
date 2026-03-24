"""
Main Evaluation Script

Runs quantitative evaluation metrics for the PET attenuation
correction challenge.

Usage:
    python eval.py <subject_path> <pred_pet> <pred_ct> [-all | -specific_metric <name>]

Example:
    python eval.py /data/sub-000 /results/pet.nii.gz /results/ct.nii.gz -all
"""

import argparse
import os
import numpy as np

from metrics import (
    compute_whole_body_suv_mae,
    compute_organ_bias_from_totalseg,
    compute_whole_body_mu_mae,
)


def evaluate_subject(subject_path, pred_pet_path=None, pred_ct_path=None):
    """
    Run metrics for a single subject.

    Parameters
    ----------
    subject_path : str
        Path to the subject directory (must contain ct-label/ and pet-label/).
    pred_pet_path : str or None
        Path to predicted PET NIfTI. If given, runs PET metrics (SUV MAE, Organ Bias).
    pred_ct_path : str or None
        Path to predicted CT NIfTI. If given, runs CT MAE.

    Note
    ----
    Brain Outlier Score is a dataset-level metric and cannot be computed per-subject.
    Use compute_brain_outlier_score() directly with paths from multiple subjects.

    Returns
    -------
    dict
        {metric_name: float}
    """

    if pred_pet_path is None and pred_ct_path is None:
        raise ValueError("At least one of pred_pet_path or pred_ct_path must be provided.")

    ct_label_dir  = os.path.join(subject_path, "ct-label")
    pet_label_dir = os.path.join(subject_path, "pet-label")

    results = {}

    if pred_pet_path is not None:
        gt_pet        = os.path.join(pet_label_dir, "pet.nii.gz")
        body_seg_pet  = os.path.join(pet_label_dir, "body_seg.nii.gz")
        organ_seg_pet = os.path.join(pet_label_dir, "organ_seg.nii.gz")

        results["Whole-body SUV MAE"] = compute_whole_body_suv_mae(
            pred_pet_path=pred_pet_path,
            gt_pet_path=gt_pet,
            body_mask_path=body_seg_pet,
            organ_seg_path=organ_seg_pet,
        )

        results["Organ Bias"] = compute_organ_bias_from_totalseg(
            pred_path=pred_pet_path,
            gt_path=gt_pet,
            totalseg_path=organ_seg_pet,
            body_mask_path=body_seg_pet,
        )

    if pred_ct_path is not None:
        gt_ct        = os.path.join(ct_label_dir,  "ct.nii.gz")
        body_seg_ct  = os.path.join(ct_label_dir,  "body_seg.nii.gz")
        organ_seg_ct = os.path.join(ct_label_dir,  "organ_seg.nii.gz")

        results["CT MAE"] = compute_whole_body_mu_mae(
            pred_ct_path=pred_ct_path,
            gt_ct_path=gt_ct,
            body_mask_path=body_seg_ct,
            organ_seg_path=organ_seg_ct,
        )

    return results


def main():

    parser = argparse.ArgumentParser(
        description="PET Attenuation Correction Challenge — Evaluation"
    )

    parser.add_argument("--subject_path", required=True, help="Path to subject directory")
    parser.add_argument("--pred_pet",     default=None,  help="Path to predicted PET NIfTI")
    parser.add_argument("--pred_ct",      default=None,  help="Path to predicted CT NIfTI")

    args = parser.parse_args()

    if args.pred_pet is None and args.pred_ct is None:
        parser.error("At least one of --pred_pet or --pred_ct must be provided.")

    results = evaluate_subject(args.subject_path, args.pred_pet, args.pred_ct)

    print("\n================ Evaluation Results ================")
    print(f"Subject: {os.path.basename(args.subject_path)}")
    print("----------------------------------------------------")
    for name, value in results.items():
        unit = "%" if name == "Organ Bias" else ""
        print(f"{name:<25}: {value:.6f}{unit}")
    print("====================================================\n")

    return results


if __name__ == "__main__":
    main()
