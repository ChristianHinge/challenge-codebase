# PET Evaluation Tool

This repository provides a **quantitative evaluation tool** for the
**Big Cross-Modal Attenuation Correction Challenge**.

It evaluates predicted **pseudo-CTAC PET** and **pseudo-CT** images
against the **reference PET and CT images** provided in the dataset.

The tool computes several metrics commonly used in **PET attenuation
correction research**.

------------------------------------------------------------------------

# Implemented Metrics

The following evaluation metrics are implemented:

### 1. Whole-body SUV MAE

Mean absolute error of PET SUV values inside the body mask, excluding a
region around the liver.

### 2. Brain Outlier Robustness Score

Measures robustness of the prediction in brain regions by identifying
extreme voxel errors.

### 3. Organ Bias (SUV-mean MARE)

Mean absolute relative error of SUV mean values across multiple organs.

### 4. CT μ-MAE

Mean absolute error of the **attenuation coefficient μ (511 keV)**
derived from CT after HU→μ conversion.

------------------------------------------------------------------------

# Requirements

-   Python **3.12**
-   [`uv`](https://github.com/astral-sh/uv) for environment and
    dependency management

------------------------------------------------------------------------

# Installation

Clone the repository and install dependencies:

``` bash
git clone <repository_url>
cd <repository_folder>
uv sync
```

------------------------------------------------------------------------

# Dataset Structure

Each subject directory must follow the structure below:

    subject/
    │
    ├── ct-label/
    │   ├── ct.nii.gz
    │   └── body_seg.nii.gz
    │
    ├── pet-label/
    │   ├── pet.nii.gz
    │   ├── body_seg.nii.gz
    │   └── organ_seg.nii.gz
    │
    └── features/
        └── metadata.json

## File Description

  File                 Description
  -------------------- ------------------------------------------
  `pet.nii.gz`         Reference PET image
  `ct.nii.gz`          Reference CT image
  `body_seg.nii.gz`    Body mask
  `organ_seg.nii.gz`   Organ segmentation from TotalSegmentator
  `metadata.json`      Subject metadata used for SUV conversion

------------------------------------------------------------------------

# Prediction Requirements

Participants must provide:

-   **Predicted PET volume**
-   **Predicted CT volume**

Both must:

-   Be in **NIfTI format (`.nii.gz`)**
-   Have the **same shape, spacing, orientation, and field-of-view** as
    the reference images.

------------------------------------------------------------------------

# Running the Evaluation

There are two entry points: `eval_case.py` for a single subject and `eval_dataset.py` for a
full dataset (this matches the challenge leaderboard computation, including the dataset-level
Brain Outlier Score).

## Single subject

``` bash
python eval_case.py \
  --subject_path <subject_path> \
  --pred_pet <pred_pet.nii.gz> \
  --pred_ct <pred_ct.nii.gz>
```

`--pred_pet` and `--pred_ct` are both optional — omit either to skip the corresponding metrics.

Note: Brain Outlier Score is a dataset-level metric and is not computed by `eval_case.py`.

## Full dataset

``` bash
python eval_dataset.py \
  --dataset_path <dataset_path> \
  --pred_dir <predictions_dir>
```

`<predictions_dir>` must contain one sub-folder per subject, each with `ct.nii.gz` and `pet.nii.gz`.

------------------------------------------------------------------------

# Arguments

## `eval_case.py`

  Argument             Description
  -------------------- ---------------------------------
  `--subject_path`     Path to the subject directory (must contain `ct-label/` and `pet-label/`)
  `--pred_pet`         Path to the predicted PET NIfTI (optional)
  `--pred_ct`          Path to the predicted CT NIfTI (optional)

## `eval_dataset.py`

  Argument             Description
  -------------------- -------------------------------------------------------
  `--dataset_path`     Root directory containing subject folders with ground-truth labels
  `--pred_dir`         Directory with one sub-folder per subject (each containing `ct.nii.gz` and `pet.nii.gz`)
  `--subjects`         Optional explicit list of subject IDs (default: all sub-folders in pred_dir)

------------------------------------------------------------------------

# Example

Evaluate a single subject:

``` bash
python eval_case.py \
  --subject_path /data/sub-000 \
  --pred_pet /results/sub-000/pet.nii.gz \
  --pred_ct /results/sub-000/ct.nii.gz
```

Evaluate a full dataset:

``` bash
python eval_dataset.py \
  --dataset_path /data/bic-mac/train \
  --pred_dir /results/my_method
```

------------------------------------------------------------------------

# Example Output

    ================ Evaluation Results ================
    Subject: sub-000
    ----------------------------------------------------
    Whole-body SUV MAE        : 0.124512
    Organ Bias                : 6.382100%
    CT MAE                    : 0.000218
    ====================================================

------------------------------------------------------------------------

# Notes

-   PET values are converted to **SUV** using information from
    `metadata.json`.
-   CT images are converted from **HU to attenuation coefficient μ (511
    keV)** before computing CT μ-MAE.
-   Brain masks are derived from **TotalSegmentator segmentation**.
-   The evaluation excludes a region around the **superior liver slice
    (±4 cm)** to avoid bias from high physiological uptake.
-   Predicted images must be **aligned with the reference images** (same
    orientation, resolution, and field of view).
