import argparse
import os

try:
    from .metrics import (
        compute_whole_body_suv_mae,
        compute_organ_bias,
        compute_whole_body_mu_mae,
    )
except ImportError:
    from metrics import (
        compute_whole_body_suv_mae,
        compute_organ_bias,
        compute_whole_body_mu_mae,
    )


def evaluate_subject(subject_path, pred_pet_path=None, pred_ct_path=None, quiet=False):
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
        gt_pet         = os.path.join(pet_label_dir, "pet.nii.gz")
        body_seg_pet   = os.path.join(pet_label_dir, "body_seg.nii.gz")
        organ_seg_pet  = os.path.join(pet_label_dir, "organ_seg.nii.gz")
        tissue_seg_pet = os.path.join(pet_label_dir, "tissue_seg.nii.gz")

        results["pet_whole_body_suv_mae"] = compute_whole_body_suv_mae(
            pred_pet_path=pred_pet_path,
            gt_pet_path=gt_pet,
            body_seg_path=body_seg_pet,
            organ_seg_path=organ_seg_pet,
        )

        results["pet_organ_bias"] = compute_organ_bias(
            pred_path=pred_pet_path,
            gt_path=gt_pet,
            organ_seg_path=organ_seg_pet,
            tissue_seg_path=tissue_seg_pet,
            body_seg_path=body_seg_pet,
        )

    if pred_ct_path is not None:
        gt_ct        = os.path.join(ct_label_dir,  "ct.nii.gz")
        body_seg_ct  = os.path.join(ct_label_dir,  "body_seg.nii.gz")
        organ_seg_ct = os.path.join(ct_label_dir,  "organ_seg.nii.gz")

        results["ct_mu_map_mae"] = compute_whole_body_mu_mae(
            pred_ct_path=pred_ct_path,
            gt_ct_path=gt_ct,
            body_seg_path=body_seg_ct,
            organ_seg_path=organ_seg_ct,
        )

    DISPLAY_NAMES = {
        "pet_whole_body_suv_mae": "[PET] Whole-body SUV MAE",
        "pet_organ_bias":         "[PET] Organ Bias",
        "ct_mu_map_mae":          "[CT] Mu-map MAE",
    }
    UNITS = {"pet_organ_bias": "%"}

    if not quiet:
        print("\n================ Evaluation Results ================")
        print(f"Subject: {os.path.basename(subject_path)}")
        print("----------------------------------------------------")
        for key, value in results.items():
            label = DISPLAY_NAMES.get(key, key)
            unit  = UNITS.get(key, "")
            print(f"{label} ↓{'':<{34 - len(label)}}: {value:.6f}{unit}")
        print("====================================================\n")

    return results


def main():

    parser = argparse.ArgumentParser(
        description="PET Attenuation Correction Challenge — Evaluation",
        epilog="Note: Brain Outlier Score is a dataset-level metric and is not computed here. Use eval_dataset.py to compute it across multiple subjects.",
    )

    parser.add_argument("--subject_path", required=True, help="Path to subject directory, e.g. /data/sub-000 (must contain ct-label/ if using --pred_ct, pet-label/ if using --pred_pet)")
    parser.add_argument("--pred_pet",     default=None,  help="Path to predicted PET NIfTI")
    parser.add_argument("--pred_ct",      default=None,  help="Path to predicted CT NIfTI")

    args = parser.parse_args()

    if args.pred_pet is None and args.pred_ct is None:
        parser.error("At least one of --pred_pet or --pred_ct must be provided.")

    evaluate_subject(args.subject_path, args.pred_pet, args.pred_ct)


if __name__ == "__main__":
    main()
