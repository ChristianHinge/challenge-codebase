# BIC-MAC Challenge Codebase

**Big Cross-Modal Attenuation Correction** — synthesize pseudo-CT from multi-modal PET/MRI input to enable CT-less PET reconstruction.

[Challenge website](https://bic-mac-challenge.github.io/)

---

## Overview

PET/CT scanners combine functional PET imaging with anatomical CT. The CT is used to compute an attenuation correction factor (ACF) — a correction for how much the body absorbs the 511 keV photons emitted during PET scanning. While PET tracer doses have dropped significantly with modern scanners, the CT component remains the primary radiation source in a PET/CT exam.

This challenge asks participants to synthesize a pseudo-CT from three radiation-free inputs: non-attenuation-corrected PET (NAC-PET), whole-body MRI, and a 2D topogram (scout X-ray). The predicted CT is then used as a drop-in replacement in the standard reconstruction pipeline, producing an attenuation-corrected PET (ACPET) image without a volumetric CT scan. Eliminating the CT dose is particularly valuable for radiation-sensitive populations such as children and pregnant patients.

The dataset comprises 100 healthy volunteers acquired on a **Siemens Biograph Vision Quadra** (PET/CT) and **MAGNETOM Vida** (MRI), with 75 training cases and 4 validation cases. Both static and dynamic acquisition protocols are included.

---

## Repository Structure

```
src/
├── baseline/       # Baseline pseudo-CT model (patch-based MONAI 3D UNet, NAC-PET input)
├── evaluation/     # Five-metric evaluation suite
└── recon/          # PET reconstruction pipeline (STIR-based, Dockerized)
```

---

## Getting Started

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv), Docker

```bash
uv sync
```

**Dataset:** Download from Hugging Face Hub (link available at challenge launch — see [website](https://bic-mac-challenge.github.io/)).

---

## Data Format

Each subject is a directory with four subdirectories:

```
sub-000/
├── features/           # Inputs to your model
├── ct-label/           # Ground-truth CT and segmentations
├── recon/              # Sinogram data for reconstruction
└── pet-label/          # Ground-truth ACPET and segmentations
```

### `features/` — model inputs

| File | Description |
|------|-------------|
| `nacpet.nii.gz` | Non-attenuation-corrected PET (NAC-PET) |
| `topogram.nii.gz` | 2D scout X-ray (topogram) resampled to 3D CT grid |
| `mri_chunk_*_*.nii.gz` | DIXON MRI chunks (in-phase and out-of-phase) |
| `mri_combined_*_phase.nii.gz` | Combined whole-body DIXON MRI |
| `metadata.json` | Demographics: `{sex, age, height, weight}` |

### `ct-label/` — ground-truth CT

| File | Description |
|------|-------------|
| `ct.nii.gz` | Anonymized ground-truth CT in Hounsfield units (HU) |
| `body_seg.nii.gz` | Binary body mask |
| `organ_seg.nii.gz` | TotalSegmentator organ labels |
| `face_seg.nii.gz` | Face mask (used for anonymization) |

### `recon/` — reconstruction inputs

| File | Description |
|------|-------------|
| `mult_factors_forSTIR_SSRB.hs/.s` | Multiplicative correction sinogram |
| `additive_term_SSRB.hs/.s` | Additive correction sinogram (scatter + randoms) |
| `prompts_SSRB.hs/.s` | Prompt (raw) sinogram |
| `offset.json` | Bed position and gantry offset for origin alignment |
| `ct_face.nii.gz` | Ground-truth CT face region (for optional face swap) |
| `face_mask.nii.gz` | Face mask for anonymization |

### `pet-label/` — ground-truth ACPET

| File | Description |
|------|-------------|
| `acpet.nii.gz` | Ground-truth CT-attenuation-corrected PET (reference) |
| `body_seg.nii.gz` | Body mask resampled to PET space |
| `organ_seg.nii.gz` | Organ labels resampled to PET space |
| `brain_seg.nii.gz` | SynthSeg brain segmentation |

---

## Baseline (`src/baseline/`)

A simple patch-based MONAI 3D UNet that predicts pseudo-CT from NAC-PET only. It is provided as a starting-point reference — participants are expected to improve on it by incorporating MRI and topogram inputs.

A pre-built Docker image is available for download (see [website](https://bic-mac-challenge.github.io/)).

**Direct Python usage:**

```bash
python src/baseline/model.py <input_dir> <output_ct.nii.gz>
# Example:
python src/baseline/model.py data/sub-000/features/ results/sub-000/ct_pred.nii.gz
```

---

## Reconstruction Pipeline (`src/recon/`)

Converts a predicted pseudo-CT into a reconstructed ACPET image using [STIR](http://stir.sourceforge.net/) (Software for Tomographic Image Reconstruction). The pipeline:

1. Validates CT shape, affine, and HU range
2. Converts HU → linear attenuation coefficients (μ-map) at 511 keV using the Carney et al. (2006) bilinear model
3. Smooths the μ-map (4mm FWHM Gaussian)
4. Resamples the μ-map to STIR sinogram format
5. Computes the ACF (attenuation correction factor) sinogram
6. Applies ACF to multiplicative/additive sinograms
7. Reconstructs using OSEM (ordered subsets expectation maximisation)
8. Applies 4mm post-reconstruction filter
9. Converts to NIfTI with correct bed/gantry offset origin

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

The reconstructed PET is written to `/data/output/pet.nii.gz`.

Set `OVERWRITE=1` to re-run over existing outputs:

```bash
docker run --rm -e OVERWRITE=1 \
  -v /path/to/sub-000/recon:/data/recon \
  ...
```

### Option 2: Direct Python (requires local STIR)

```bash
python src/recon/main.py <recon_dir> <ct.nii.gz> <pet_out.nii.gz> \
  [--overwrite] [--intermediates_dir <dir>]
```

---

## Evaluation (`src/evaluation/`)

Five metrics compare predicted PET and CT outputs against the ground truth:

| Metric | Flag | Description | Region |
|--------|------|-------------|--------|
| Whole-body SUV MAE | `whole_body_mae` | Mean absolute error in standardised uptake value (SUV = activity × weight / total dose) | Body mask, excluding ±4 cm around liver |
| Brain Outlier Score | `brain_outlier` | AUC of fraction of brain voxels within relative error thresholds (5%, 10%, 15%) | Brain |
| Organ Bias (MARE) | `organ_bias` | Mean absolute relative error of mean SUV in 8 organs: brain, liver, spleen, heart, pancreas, muscle, adipose, extremities | TotalSegmentator organ labels |
| TAC Bias | `tac_bias` | AUC MARE for time-activity curves in aorta + 6 brain regions (4D dynamic PET only) | Aorta + brain regions |
| CT MAE | `ct_mae` | Mean absolute error in HU between predicted and ground-truth CT | Body mask |

**Run all metrics:**

```bash
python src/evaluation/eval.py <subject_dir> <pred_pet.nii.gz> <pred_ct.nii.gz> -all
```

**Run a single metric:**

```bash
python src/evaluation/eval.py <subject_dir> <pred_pet.nii.gz> <pred_ct.nii.gz> \
  -specific_metric <metric>
# <metric>: whole_body_mae | brain_outlier | organ_bias | tac_bias | ct_mae
```

---

## Tips & FAQ

_Coming soon._

---

## Organizers

This challenge is organized by [Rigshospitalet](https://www.rigshospitalet.dk/), [Technical University of Denmark (DTU)](https://www.dtu.dk/), [University of Copenhagen (KU)](https://www.ku.dk/), and [KU Leuven](https://www.kuleuven.be/), with support from [QIM](https://qim.dk/) and [SyneRBI](https://www.synergistic-biomedical-imaging.eu/).

For full details, timeline, and registration: [bic-mac-challenge.github.io](https://bic-mac-challenge.github.io/)
