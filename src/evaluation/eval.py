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
    compute_tac_bias,
    compute_whole_body_mu_mae,
)


def main():

    parser = argparse.ArgumentParser(
        description="PET Attenuation Correction Challenge — Evaluation"
    )

    parser.add_argument(
        "subject_path",
        help="Path to subject directory"
    )

    parser.add_argument(
        "pred_pet",
        help="Path to predicted PET NIfTI"
    )

    parser.add_argument(
        "pred_ct",
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
            "tac_bias",
            "ct_mae",
        ],
        help="Run specific metric only"
    )

    args = parser.parse_args()

    subject_path  = args.subject_path
    ct_label_dir  = os.path.join(subject_path, "ct-label")
    pet_label_dir = os.path.join(subject_path, "pet-label")
    features_dir  = os.path.join(subject_path, "features")

    gt_pet        = os.path.join(pet_label_dir, "acpet.nii.gz")
    gt_ct         = os.path.join(ct_label_dir,  "ct.nii.gz")
    body_seg_pet  = os.path.join(pet_label_dir, "body_seg.nii.gz")
    organ_seg_pet = os.path.join(pet_label_dir, "organ_seg.nii.gz")
    body_seg_ct   = os.path.join(ct_label_dir,  "body_seg.nii.gz")
    synthseg      = os.path.join(pet_label_dir, "brain_seg.nii.gz")
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
            brain_mask_paths=[synthseg]
        )

    # =====================================================
    # 3. Organ Bias
    # =====================================================

    if args.all or args.specific_metric == "organ_bias":

        organ_labels = {
            "brain": 90,
            "liver": 5,
            "spleen": 1,
            "heart": 52,
            "pancreas": 10,
            "muscle": 200,
            "adipose": 201,
            "extremities": 300,
        }

        results["Organ Bias"] = compute_organ_bias_from_totalseg(
            pred_path=args.pred_pet,
            gt_path=gt_pet,
            totalseg_path=organ_seg_pet,
            organ_label_dict=organ_labels,
            json_path=meta_json,
        )

    # =====================================================
    # 4. TAC Bias (Dynamic Only)
    # =====================================================

    if args.all or args.specific_metric == "tac_bias":

        pet_data = nib.load(args.pred_pet).get_fdata()

        if pet_data.ndim != 4:
            print("TAC Bias skipped: PET is not dynamic (4D).")
        else:
            frame_durations = np.array([4.0] * pet_data.shape[-1])

            results["TAC Bias"] = compute_tac_bias(
                pred_path=args.pred_pet,
                gt_path=gt_pet,
                totalseg_path=organ_seg_pet,
                synthseg_path=synthseg,
                frame_durations=frame_durations,
                aorta_label=52,
                brain_label_ids=[3, 42, 10, 49, 8, 47]
            )

    # =====================================================
    # 5. CT MAE
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
            print(f"{name:<25}: {value:.6f}")

    print("====================================================\n")


if __name__ == "__main__":
    main()
