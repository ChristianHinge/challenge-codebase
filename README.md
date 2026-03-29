# BIC-MAC Challenge Codebase

**Big Cross-Modal Attenuation Correction** — synthesize pseudo-CT from multi-modal PET/MRI input to enable CT-less PET reconstruction.

[Challenge website](https://bic-mac-challenge.github.io/)

---

## Table of Contents

- [Overview](#overview)
- [Documentation](#documentation)
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
2. **PET accuracy** — predicted CT is fed into the reconstruction pipeline to produce an attenuation-corrected PET image, which is then compared against the ground-truth PET

Note that no PET reconstruction experience is needed to participate in the challenge, and the main purpose of the reconstruction is to enable clinically meaningful metrics. 

The dataset comprises 99 subject-unique cases, with 20 reserved for testing and the remaining 79 available on huggingface and split as follows:

| Split | Subjects | Contents |
|-------|----------|----------|
| `train/` (full) | 8 | `features/` + `ct-label/` + `recon/` + `pet-label/` |
| `train/` (no recon) | 67 | `features/` + `ct-label/` |
| `val/` | 4 | `features/` + `recon/` |

All train cases have CT labels, but due to the size of the sinograms, only 8 include the recon and pet-label folders needed for closed loop reconstruction. Validation subjects have sinogram data but no labels — submit predicted CTs and reconstructed PET to Codabench to get live leaderboard metrics throughout the challenge.

---

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [PET Background](docs/pet-background.md) | PET physics and attenuation correction — start here if you're new to PET |
| [Submission Guide](docs/submission-guide.md) | Validation, dry-run, and final submission phases explained |
| [Docker Packaging](docs/docker-packaging.md) | How to containerize your model, with baseline as a worked example |

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
All images are resampled to the label CT image (tensor size: 512x512x531, voxel size 1.52x1.52,2.00mm^3) and structured in four folders per case. 
- `features/` All the files you can use as input to your generative CT model at inference.
- 
```


train/
└── sub-000/
    ├── features/                          # generative model inputs
    │   ├── nacpet.nii.gz                  # non-attenuation-corrected PET. 
    │   ├── topogram.nii.gz                # 2D scout X-ray
    │   ├── mri_chunk_{0-3}_{in/out}_phase.nii.gz    # DIXON MRI bed position (0-3), in-phase and out-phase
    │   ├── mri_combined_{in/out}_phase.nii.gz  # stitched whole-body MRI, out-of-phase
    │   ├── mri_face_mask.nii.gz           # binary anonymization mask
    │   └── metadata.json                  # {sex, age, height, weight}
    ├── ct-label/                          # ground-truth CT
    │   ├── ct.nii.gz                      # in HU this is what your algorithm should predict
    │   ├── body_seg.nii.gz                # TotalSegmentator body seg.
    │   ├── organ_seg.nii.gz               # TotalSegmentator organ seg.
    │   └── prediction_mask.nii.gz         # The generative model should focus only on these voxels (face + scanner are excluded)
    ├── recon/                             # sinogram data
    │   ├── mult_nac_rd85.hs/.s            # multiplicative sinogram
    │   ├── add_nac_rd85.hs/.s             # additive sinogram
    │   ├── prompts_rd85.hs/.s             # raw sinogram
    │   ├── offset.json                    # bed position and gantry offset
    │   ├── ct_face_and_bed.nii.gz         # GT CT values at face + scanner bed (automatically superimposed on your prediction before reconstruction)
    │   └── face_and_bed_mask.nii.gz       # binary face + scanner bed mask
    └── pet-label/                         # ground-truth PET
        ├── pet.nii.gz                     # CT-attenuation-corrected PET (reference)
        ├── body_seg.nii.gz                # body mask in PET space
        └── organ_seg.nii.gz               # organ labels in PET space
```

---

## 📦 Baseline (`src/baseline/`)

A simple patch-based MONAI 3D UNet that predicts pseudo-CT from NAC-PET only. It is provided as a starting-point reference — participants are expected to improve on it by incorporating MRI and topogram inputs.

**Direct Python usage:**

```bash
python src/baseline/predict.py --features_dir <features_dir> --output_ct <ct.nii.gz>
# Example:
python src/baseline/predict.py --features_dir data/sub-000/features/ --output_ct results/sub-000/ct.nii.gz
```

**Docker usage:**

```bash
docker pull ghcr.io/bic-mac-challenge/baseline:latest

docker run --rm \
  --gpus all \
  -v /path/to/sub-XXX/features:/data/features:ro \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/baseline:latest
```

The predicted CT is written to `/data/output/ct.nii.gz`. All weights and dependencies are baked into the image — no internet access needed at runtime.

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
docker pull ghcr.io/bic-mac-challenge/recon:latest 

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
python src/recon/main.py --recon_dir <recon_dir> --ct <ct.nii.gz> --output_dir <output_dir> [-w] [-v]
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

**Evaluate a single subject:**

```bash
python src/evaluation/eval_subject.py \
  --subject_dir <subject_dir> \
  --pred_pet <pred_pet.nii.gz> \
  --pred_ct <pred_ct.nii.gz>
```

`--pred_pet` and `--pred_ct` are both optional — omit either to skip PET or CT metrics.
Note: Brain Outlier Score is a dataset-level metric and requires multiple subjects (see below).

**Evaluate a full dataset (matches challenge leaderboard):**

```bash
python src/evaluation/eval_dataset.py \
  --dataset_dir <dataset_dir> \
  --pred_dir <predictions_dir>
```

`<predictions_dir>` must contain one sub-folder per subject, each with `ct.nii.gz` and `pet.nii.gz`.

---

## 📬 Submission

There are three submission phases — **Validation** (NIfTI upload to Codabench), **Dry Run** (container sanity check), and **Final Test** (container + full recon + evaluation). Validation and Dry Run run concurrently during the pre-evaluation period (May 15 – Jun 15).

See [docs/submission-guide.md](docs/submission-guide.md) for full instructions on each phase.

For phases requiring a Docker container, your image must:

- Read from `/data/features/` (read-only mount)
- Write `ct.nii.gz` to `/data/output/`
- Run within 5 minutes, with 128 GB RAM and no network access

See [docs/docker-packaging.md](docs/docker-packaging.md) for a step-by-step guide to building and testing your container, with the baseline as a worked example.

---

## 💡 Tips & FAQ

_Coming soon._

---

## 🏛️ Organizers

This challenge is organized by [Rigshospitalet](https://www.rigshospitalet.dk/), [Technical University of Denmark (DTU)](https://www.dtu.dk/), [University of Copenhagen (KU)](https://www.ku.dk/), and [KU Leuven](https://www.kuleuven.be/), with support from [QIM](https://qim.dk/) and [SyneRBI](https://www.synergistic-biomedical-imaging.eu/).

For full details, timeline, and registration: [bic-mac-challenge.github.io](https://bic-mac-challenge.github.io/)
