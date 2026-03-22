# BIC-MAC Challenge Codebase

**Big Cross-Modal Attenuation Correction** — synthesize pseudo-CT from multi-modal PET/MRI input to enable CT-less PET reconstruction.

[Challenge website](https://bic-mac-challenge.github.io/)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Data Format](#data-format)
- [Baseline](#baseline-srcbaseline)
- [Reconstruction Pipeline](#reconstruction-pipeline-srcrecon)
- [Evaluation](#evaluation-srcevaluation)
- [Submission](#submission)
- [Tips & FAQ](#tips--faq)
- [Organizers](#organizers)

---

## 🧠 Overview

Your algorithm receives the files under `features/` for each subject and must output a predicted CT volume as a NIfTI file in Hounsfield units (HU). Predictions are evaluated two ways:

1. **CT accuracy** — predicted CT is compared directly against the ground-truth CT
2. **PET accuracy** — predicted CT is fed into the reconstruction pipeline to produce an attenuation-corrected PET (ACPET) image, which is then compared against the ground-truth PET

The dataset (100 subjects, Siemens Biograph Vision Quadra + MAGNETOM Vida) is split as follows:

| Split | Subjects | Contents |
|-------|----------|----------|
| `train/` (full) | 8 | `features/` + `ct-label/` + `recon/` + `pet-label/` |
| `train/` (no recon) | 68 | `features/` + `ct-label/` |
| `val/` | 4 | `features/` + `recon/` |

All train subjects have CT labels. The 8 fully-equipped subjects additionally include sinogram data and PET labels, enabling closed-loop local evaluation. Validation subjects have sinogram data but no labels — submit reconstructed PET to Codabench.

---

## 📁 Repository Structure

```
src/
├── baseline/       # Baseline pseudo-CT model (patch-based MONAI 3D UNet, NAC-PET input)
├── evaluation/     # Five-metric evaluation suite
└── recon/          # PET reconstruction pipeline (STIR-based, Dockerized)
```

---

## 🚀 Getting Started

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv), Docker

```bash
uv sync
```

**Dataset:** Download from Hugging Face Hub (link available at challenge launch — see [website](https://bic-mac-challenge.github.io/)).

---

## 🗂️ Data Format

```
train/
└── sub-000/
    ├── features/                          # model inputs (all subjects)
    │   ├── nacpet.nii.gz                  # non-attenuation-corrected PET
    │   ├── topogram.nii.gz                # 2D scout X-ray resampled to CT grid
    │   ├── mri_chunk_0_in_phase.nii.gz    # DIXON MRI bed position 0, in-phase
    │   ├── mri_chunk_0_out_phase.nii.gz   # DIXON MRI bed position 0, out-of-phase
    │   ├── mri_chunk_1_in_phase.nii.gz    #   ... (chunks 0–3 for each phase)
    │   ├── mri_chunk_1_out_phase.nii.gz
    │   ├── mri_chunk_2_in_phase.nii.gz
    │   ├── mri_chunk_2_out_phase.nii.gz
    │   ├── mri_chunk_3_in_phase.nii.gz
    │   ├── mri_chunk_3_out_phase.nii.gz
    │   ├── mri_combined_in_phase.nii.gz   # stitched whole-body MRI, in-phase
    │   ├── mri_combined_out_phase.nii.gz  # stitched whole-body MRI, out-of-phase
    │   ├── face_seg.nii.gz                # face mask (MRI space)
    │   └── metadata.json                  # {sex, age, height, weight}
    ├── ct-label/                          # ground-truth CT (train only)
    │   ├── ct.nii.gz                      # anonymized CT in HU
    │   ├── body_seg.nii.gz                # body mask
    │   ├── organ_seg.nii.gz               # TotalSegmentator organ labels
    │   └── face_seg.nii.gz                # face mask
    ├── recon/                             # sinogram data (labeled train + val)
    │   ├── mult_nac_rd85.hs/.s            # multiplicative correction sinogram
    │   ├── add_nac_rd85.hs/.s             # additive correction sinogram (scatter + randoms)
    │   ├── prompts_rd85.hs/.s             # prompt (raw) sinogram
    │   ├── offset.json                    # bed position and gantry offset
    │   ├── ct_face_and_bed.nii.gz         # GT CT values at face + scanner bed (for swap-back)
    │   └── face_and_bed_mask.nii.gz       # face + scanner bed mask
    └── pet-label/                         # ground-truth PET (labeled train only)
        ├── acpet.nii.gz                   # CT-attenuation-corrected PET (reference)
        ├── body_seg.nii.gz                # body mask in PET space
        └── organ_seg.nii.gz               # organ labels in PET space
```

---

## 📦 Baseline (`src/baseline/`)

A simple patch-based MONAI 3D UNet that predicts pseudo-CT from NAC-PET only. It is provided as a starting-point reference — participants are expected to improve on it by incorporating MRI and topogram inputs.

A pre-built Docker image is available for download (see [website](https://bic-mac-challenge.github.io/)).

**Direct Python usage:**

```bash
python src/baseline/model.py <input_dir> <output_ct.nii.gz>
# Example:
python src/baseline/model.py data/sub-000/features/ results/sub-000/ct_pred.nii.gz
```

---

## ⚙️ Reconstruction Pipeline (`src/recon/`)

Converts a predicted pseudo-CT into a reconstructed ACPET image using [STIR](http://stir.sourceforge.net/) (Software for Tomographic Image Reconstruction). The pipeline:

1. Validates CT shape, affine, and HU range
2. Swaps face and scanner bed region back from ground-truth CT (to avoid evaluating face/bed prediction)
3. Converts HU → linear attenuation coefficients (μ-map) at 511 keV using the Carney et al. (2006) bilinear model
4. Smooths the μ-map (4mm FWHM Gaussian)
5. Resamples the μ-map to STIR format (ring spacing 3.29114 mm)
6. Computes the ACF (attenuation correction factor) sinogram
7. Applies ACF to the additive sinogram
8. Applies ACF to the multiplicative sinogram
9. Reconstructs using OSEM (ordered subsets expectation maximisation, with post-filter)
10. Converts to NIfTI with correct bed/gantry offset origin

### Option 1: Docker (recommended)

A pre-built image with STIR and all dependencies is available (see [website](https://bic-mac-challenge.github.io/)).

```bash
docker pull ghcr.io/bic-mac-challenge/recon:latest  # placeholder — final name on website

docker run --rm \
  -v /path/to/sub-000/recon:/data/recon \
  -v /path/to/ct_pred.nii.gz:/data/ct/ct.nii.gz \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
```

The reconstructed PET is written to `/data/output/pet.nii.gz`. Intermediate files (mu-map, ACF sinogram, etc.) are written to `/data/output/intermediates/` and a full debug log to `/data/output/intermediates/recon.log`.

The pipeline resumes from any existing intermediates automatically; set `OVERWRITE=1` to forcefully restart from scratch:

```bash
docker run --rm \
  -e OVERWRITE=1 \
  -v /path/to/sub-000/recon:/data/recon \
  -v /path/to/ct_pred.nii.gz:/data/ct/ct.nii.gz \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
```

Set `VERBOSE=1` to stream STIR subprocess output to the terminal in addition to the log file.

### Option 2: Direct Python (requires local STIR)

```bash
python src/recon/main.py <recon_dir> <ct.nii.gz> <output_dir> [-w] [-v]
```

`pet.nii.gz` and `intermediates/` are written inside `output_dir`. Use `-w`/`--overwrite` to rerun from scratch and `-v`/`--verbose` to stream STIR output to the terminal.

---

## 📊 Evaluation (`src/evaluation/`)

Five metrics compare predicted PET and CT outputs against the ground truth:

| Metric | Flag | Description | Region |
|--------|------|-------------|--------|
| Whole-body SUV MAE | `whole_body_mae` | Mean absolute error in standardised uptake value (SUV = activity × weight / total dose) | Body mask, excluding ±4 cm around liver |
| Brain Outlier Score | `brain_outlier` | AUC of fraction of brain voxels within relative error thresholds (5%, 10%, 15%) | Brain |
| Organ Bias (MARE) | `organ_bias` | Mean absolute relative error of mean SUV in 8 organs: brain, liver, spleen, heart, pancreas, muscle, adipose, extremities | TotalSegmentator organ labels |
| CT MAE | `ct_mae` | Mean absolute error of attenuation coefficients (μ at 511 keV) between predicted and ground-truth CT after HU→μ conversion | Body mask, excluding ±4 cm around liver|

**Run all metrics:**

```bash
python src/evaluation/eval.py <subject_dir> <pred_pet.nii.gz> <pred_ct.nii.gz> -all
```

**Run a single metric:**

```bash
python src/evaluation/eval.py <subject_dir> <pred_pet.nii.gz> <pred_ct.nii.gz> \
  -specific_metric <metric>
# <metric>: whole_body_mae | brain_outlier | organ_bias | ct_mae
```

---

## 📬 Submission

Wrap your algorithm in a Docker container. The evaluation system will run your container with two mounts:

- `/data/features/` — read-only input directory (contents of `features/` for the subject)
- `/data/output/` — write directory for your predictions

Your container must write the predicted CT to `/data/output/ct.nii.gz` as a NIfTI file in Hounsfield units (HU), with the same affine and shape as the input CT space.

The exact command used to run your container is:

```bash
docker run --rm \
  -v /path/to/sub-XXX/features:/data/features:ro \
  -v /path/to/output:/data/output \
  <your-image>
```

No other files or directories are mounted. Your container must not require network access at inference time.

Submit your image name and tag via Codabench (see [website](https://bic-mac-challenge.github.io/) for registration and submission instructions).

---

## 💡 Tips & FAQ

_Coming soon._

---

## 🏛️ Organizers

This challenge is organized by [Rigshospitalet](https://www.rigshospitalet.dk/), [Technical University of Denmark (DTU)](https://www.dtu.dk/), [University of Copenhagen (KU)](https://www.ku.dk/), and [KU Leuven](https://www.kuleuven.be/), with support from [QIM](https://qim.dk/) and [SyneRBI](https://www.synergistic-biomedical-imaging.eu/).

For full details, timeline, and registration: [bic-mac-challenge.github.io](https://bic-mac-challenge.github.io/)
