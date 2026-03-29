# PET Reconstruction Pipeline

The reconstruction pipeline (`src/recon/`) converts a pseudo-CT into an attenuation-corrected PET image using [STIR](http://stir.sourceforge.net/) (Software for Tomographic Image Reconstruction). You do **not** need to understand or modify the pipeline to participate — it is run by the challenge organisers on your submissions. This guide is for participants who want to run it locally for closed-loop training or debugging.

---

## Pipeline Steps

Given a CT (ground-truth or pseudo-CT) and the subject's sinogram data (`recon/`), the pipeline produces a reconstructed PET NIfTI:

1. **Validate CT** — checks shape, affine, and HU range against the ground-truth CT
2. **Swap face and bed** — replaces the face and scanner bed region with ground-truth CT values (so face/bed prediction is not penalised)
3. **HU → μ-map** — converts Hounsfield units to linear attenuation coefficients at 511 keV using the Carney et al. (2006) bilinear model at 120 kVp
4. **Smooth μ-map** — applies a 4 mm FWHM Gaussian to match scanner resolution
5. **Resample to STIR** — resamples the μ-map onto the STIR z-axis grid (ring spacing 3.29114 mm)
6. **Compute ACF sinogram** — forward-projects the μ-map to produce the attenuation correction factor (ACF) sinogram
7. **Apply ACF to additive sinogram** — multiplies ACF into the scatter+randoms estimate
8. **Apply ACF to multiplicative sinogram** — multiplies ACF into the detector normalisation sinogram
9. **OSEM reconstruction** — reconstructs PET via ordered-subsets expectation maximisation with a 4 mm post-filter
10. **Convert to NIfTI** — writes the reconstructed PET with the correct bed/gantry offset origin

Intermediate outputs (μ-map, ACF sinogram, STIR-format files) are written to `output_dir/intermediates/`. The pipeline skips steps whose outputs already exist, so it resumes automatically from a partial run.

---

> [!WARNING]
> Running reconstruction requires **~20 GB of RAM** and takes **20–120 minutes** depending on CPU speed. The `intermediates/` folder uses **~50 GB** of additional disk space — consider deleting it after a successful reconstruction.

## Running the Pipeline

### Option 1: Docker (recommended)

A pre-built image with STIR and all dependencies is available.

```bash
docker pull ghcr.io/bic-mac-challenge/recon:latest

docker run --rm \
  -v /path/to/sub-000/recon:/data/recon \
  -v /path/to/ct.nii.gz:/data/ct/ct.nii.gz \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
```

The reconstructed PET is written to `/data/output/pet.nii.gz`. A full debug log is written to `/data/output/intermediates/recon.log`.

**Environment variables:**

| Variable | Default | Effect |
|----------|---------|--------|
| `OVERWRITE` | `0` | Set to `1` to ignore existing intermediates and rerun from scratch |
| `VERBOSE` | `0` | Set to `1` to stream STIR subprocess output to the terminal |

```bash
docker run --rm \
  -e OVERWRITE=1 \
  -e VERBOSE=1 \
  -v /path/to/sub-000/recon:/data/recon \
  -v /path/to/ct.nii.gz:/data/ct/ct.nii.gz \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
```

### Option 2: Direct Python (requires local STIR)

```bash
python src/recon/main.py \
  --recon_dir <subject_recon_dir> \
  --ct <ct.nii.gz> \
  --output_dir <output_dir> \
  [-w] [-v]
```

| Argument | Description |
|----------|-------------|
| `--recon_dir` | Subject's `recon/` directory (contains sinograms, offset.json, face mask) |
| `--ct` | CT NIfTI file in Hounsfield units |
| `--output_dir` | Directory where `pet.nii.gz` and `intermediates/` will be written |
| `-w` / `--overwrite` | Rerun from scratch, ignoring existing intermediates |
| `-v` / `--verbose` | Stream STIR subprocess output to the terminal |

---

## Expected `recon/` Contents

```
recon/
├── add_nac_rd85.hs / .s       # additive sinogram (scatter + randoms)
├── mult_nac_rd85.hs / .s      # multiplicative sinogram (normalisation, decay)
├── prompts_rd85.hs / .s       # raw prompt coincidences
├── offset.json                # bed position and gantry offset
├── ct_face_and_bed.nii.gz     # ground-truth CT values at face + scanner bed
└── face_and_bed_mask.nii.gz   # binary face + scanner bed mask
```

Only subjects in the full `train/` split and the `val/` split include `recon/` data (see the main README).

---

## Further Reading

- Carney et al. (2006) — *"Method for Transforming CT Images for Attenuation Correction in PET/CT Scanners"*, Medical Physics. The bilinear HU→μ model used in this pipeline.
- Thielemans et al. (2012) — *"STIR: Software for Tomographic Image Reconstruction Release 2"*, Physics in Medicine and Biology. The reconstruction library used here.
