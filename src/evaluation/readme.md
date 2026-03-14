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

-   Python **3.10+**
-   [`uv`](https://github.com/astral-sh/uv) for environment and
    dependency management

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone <repository_url>
cd <repository_folder>
```

Create and activate a virtual environment:

``` bash
uv venv
source .venv/bin/activate
```

Install required dependencies:

``` bash
uv pip install numpy nibabel
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

Run the evaluation script with:

``` bash
python eval.py --subject_path <subject_path> --pred_pet <pred_pet> --pred_ct <pred_ct> [-all | -specific_metric <metric_name>]
```

------------------------------------------------------------------------

# Arguments

  Argument             Description
  -------------------- ---------------------------------
  `--subject_path`     Path to the subject directory
  `--pred_pet`         Path to the predicted PET NIfTI
  `--pred_ct`          Path to the predicted CT NIfTI
  `-all`               Run all evaluation metrics
  `-specific_metric`   Run only a single metric

------------------------------------------------------------------------

# Example

Run all metrics:

``` bash
python eval.py --subject_path /data/sub-000 --pred_pet /results/pred_pet.nii.gz --pred_ct /results/pred_ct.nii.gz -all
```

Run only CT μ-MAE:

``` bash
python eval.py --subject_path /data/sub-000 --pred_pet /results/pred_pet.nii.gz --pred_ct /results/pred_ct.nii.gz -specific_metric ct_mae
```

------------------------------------------------------------------------

# Available Metrics

The following metrics can be executed individually:

    whole_body_mae
    brain_outlier
    organ_bias
    ct_mae

Example:

``` bash
python eval.py --subject_path <subject_path> --pred_pet <pred_pet> --pred_ct <pred_ct> -specific_metric whole_body_mae
```

------------------------------------------------------------------------

# Example Output

    ================ Evaluation Results ================
    Subject: sub-000
    ----------------------------------------------------
    Whole-body SUV MAE        : 0.124512
    Brain Outlier Score       : 0.912341
    Organ Bias                : 6.382100%
    CT μ-MAE                  : 0.000218
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
