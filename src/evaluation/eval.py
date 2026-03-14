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
import nibabel as nib

from metrics import (
    compute_whole_body_suv_mae,
    compute_brain_outlier_score,
    compute_organ_bias_from_totalseg,
    compute_whole_body_mu_mae,
)


def main():

    parser = argparse.ArgumentParser(
        description="PET Attenuation Correction Challenge — Evaluation"
    )

    parser.add_argument(
        "--subject_path",
        required=True,
        help="Path to subject directory"
    )

    parser.add_argument(
        "--pred_pet",
        required=True,
        help="Path to predicted PET NIfTI"
    )

    parser.add_argument(
        "--pred_ct",
        required=True,
        help="Path to predicted CT NIfTI"
)

    parser.add_argument(
        "-all",
        action="store_true",
        help="Run all metrics"
    )

    parser.add_argument(
        "-specific_metric",
        choices=[
            "whole_body_mae",
            "brain_outlier",
            "organ_bias",
            "ct_mae",
        ],
        help="Run specific metric only"
    )

    args = parser.parse_args()

    subject_path  = args.subject_path
    ct_label_dir  = os.path.join(subject_path, "ct-label")
    pet_label_dir = os.path.join(subject_path, "pet-label")
    features_dir  = os.path.join(subject_path, "features")

    gt_pet        = os.path.join(pet_label_dir, "pet.nii.gz")
    gt_ct         = os.path.join(ct_label_dir,  "ct.nii.gz")
    body_seg_pet  = os.path.join(pet_label_dir, "body_seg.nii.gz")
    organ_seg_pet = os.path.join(pet_label_dir, "organ_seg.nii.gz")
    body_seg_ct   = os.path.join(ct_label_dir,  "body_seg.nii.gz")
    meta_json     = os.path.join(features_dir,  "metadata.json")

    results = {}

    # =====================================================
    # 1. Whole-body SUV MAE
    # =====================================================

    if args.all or args.specific_metric == "whole_body_mae":

        results["Whole-body SUV MAE"] = compute_whole_body_suv_mae(
            pred_pet_path=args.pred_pet,
            gt_pet_path=gt_pet,
            body_mask_path=body_seg_pet,
            liver_mask_path=organ_seg_pet,
            json_path=meta_json,
        )

    # =====================================================
    # 2. Brain Outlier Score
    # =====================================================

    if args.all or args.specific_metric == "brain_outlier":

        results["Brain Outlier Score"] = compute_brain_outlier_score(
            pred_paths=[args.pred_pet],
            gt_paths=[gt_pet],
            totalseg_paths=[organ_seg_pet],
        )

    # =====================================================
    # 3. Organ Bias
    # =====================================================

    if args.all or args.specific_metric == "organ_bias":

        
        results["Organ Bias"] = compute_organ_bias_from_totalseg(
            pred_path=args.pred_pet,
            gt_path=gt_pet,
            totalseg_path=organ_seg_pet,
            json_path=meta_json,
        )

    
    # =====================================================
    # 4. CT MAE
    # =====================================================

    if args.all or args.specific_metric == "ct_mae":

        results["CT MAE"] = compute_whole_body_mu_mae(
            pred_ct_path=args.pred_ct,
            gt_ct_path=gt_ct,
            body_mask_path=body_seg_ct,
            liver_mask_path=organ_seg_pet,
        )

    # =====================================================
    # Print Results
    # =====================================================

    print("\n================ Evaluation Results ================")
    print(f"Subject: {os.path.basename(subject_path)}")
    print("----------------------------------------------------")

    if not results:
        print("No metric selected.")
    else:
        for name, value in results.items():
            if name == "Organ Bias":
                print(f"{name:<25}: {value:.6f}%")
            else:
                print(f"{name:<25}: {value:.6f}")

    print("====================================================\n")


if __name__ == "__main__":
    main()
