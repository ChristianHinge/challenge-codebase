# PET Evaluation Tool

This repository provides a **quantitative evaluation tool** for the **Big Cross-Modal Attenuation Correction Challenge**.
It evaluates predicted **pseudo-CTAC PET** and **pseudo-CT** images against the **reference CTAC PET and CT images**.

The tool computes several metrics commonly used in PET attenuation correction research.

---

# Implemented Metrics

The following evaluation metrics are implemented:

1. **Whole-body SUV MAE**
   Mean absolute error of PET SUV values inside the body mask, excluding a region around the liver.

2. **Brain Outlier Robustness Score**
   Measures robustness of the prediction in brain regions by identifying extreme voxel errors.

3. **Organ Bias (SUV-mean MARE)**
   Mean absolute relative error of SUV mean values across several organs.

4. **TAC Bias (Dynamic AUC MARE)**
   Mean absolute relative error of time-activity curve AUC values for dynamic PET.

5. **CT μ-MAE**
   Mean absolute error of the attenuation coefficient (μ at 511 keV) derived from CT.

---

# Requirements

* Python **3.10 or newer**
* [`uv`](https://github.com/astral-sh/uv) for environment and dependency management

---

# Installation

Clone the repository:

```bash
git clone <repository_url>
cd <repository_folder>
```

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install required dependencies:

```bash
uv pip install numpy nibabel
```

---

# Dataset Structure

Each subject directory should follow this structure:

```
subject/
│
├── ct-label/
│   ├── ct.nii.gz
│   └── body_seg.nii.gz
│
├── pet-label/
│   ├── acpet.nii.gz
│   ├── body_seg.nii.gz
│   ├── organ_seg.nii.gz
│   └── brain_seg.nii.gz
│
└── features/
    └── metadata.json
```

---

# Running the Evaluation

Run the evaluation script with:

```
python eval.py <subject_path> <pred_pet> <pred_ct> [-all | -specific_metric <metric_name>]
```

### Arguments

| Argument           | Description                          |
| ------------------ | ------------------------------------ |
| `subject_path`     | Path to the subject directory        |
| `pred_pet`         | Path to the predicted PET NIfTI file |
| `pred_ct`          | Path to the predicted CT NIfTI file  |
| `-all`             | Run all evaluation metrics           |
| `-specific_metric` | Run only a single metric             |

---

# Example

Run all metrics:

```
python eval.py /data/sub-000 /results/pred_pet.nii.gz /results/pred_ct.nii.gz -all
```

Run only CT MAE:

```
python eval.py /data/sub-000 /results/pred_pet.nii.gz /results/pred_ct.nii.gz -specific_metric ct_mae
```

---

# Available Metrics

You can run a specific metric using:

```
-specific_metric <metric_name>
```

Available options:

```
whole_body_mae
brain_outlier
organ_bias
tac_bias
ct_mae
```

---

# Example Output

```
================ Evaluation Results ================
Subject: sub-000
----------------------------------------------------
Whole-body SUV MAE        : 0.124512
Brain Outlier Score       : 0.912341
Organ Bias                : 0.038221
TAC Bias                  : 0.052131
CT MAE                    : 0.000218
====================================================
```

---

# Notes

* PET values are converted to **SUV** using subject metadata.
* CT images are converted from **HU to attenuation coefficient μ at 511 keV** before computing CT MAE.
* The evaluation excludes a region around the **superior liver slice** to avoid bias from high physiological uptake.

---
