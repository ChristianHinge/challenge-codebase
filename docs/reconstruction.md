# PET Reconstruction Pipeline

The reconstruction pipeline (`src/recon/`) converts a predicted pseudo-CT into an attenuation-corrected PET image using [STIR](http://stir.sourceforge.net/) (Software for Tomographic Image Reconstruction). You do **not** need to understand reconstruction or attenuation correction to participate, however, having an intuition of the first reconstruction steps can be a **significant advantage** when designing your pseudo-CT algorithm and loss function.

---

## Reconstruction steps
The reconstruction 
Given a CT (ground-truth or pseudo-CT) and the subject's sinogram data (`recon/`), the pipeline produces a reconstructed PET NIfTI:

1. **Superimpose bed pixelated face** - The pseudo-CT face is replaced by a pre-saved pixelated face. Likewise, everything outside a ~1cm rim (pillows, bed, hair, air) is replaced by the ground truth image (see `ct_face_and_bed.nii.gz` and `face_and_bed_mask.nii.gz`). Consequently, the pseudo-CT algorithm will not benefit from trying to predict these areas. The `prediction_mask.nii.gz` under `ct-label` is the inverse mask of `face_and_bed_mask.nii.gz` and may be used to restrict training to the body region. 

2. **HU → μ-map** — The pseudo-CT is a volume of Hounsfield units (HU), a relative X-ray density scale where air = −1000 and water = 0. PET reconstruction needs instead the *linear attenuation coefficient* μ (cm⁻¹) at the 511 keV photon energy of PET annihilation events.

   The conversion uses a **bilinear model** (Carney et al. 2006): one linear segment maps soft tissue (HU ≤ 0, dominated by water) and a steeper segment maps bone (HU > 0). The kink at 0 HU means the functional form of the error changes depending on which side of the boundary a voxel falls on.

   **Why this matters for your model:** Errors in HU are not equally costly. A 100 HU error in dense bone (HU ≈ 700) changes μ roughly three times more than the same 100 HU error in soft tissue. Optimising purely for MAE in HU space may down-weight the tissue type that matters most. 

3. **Smooth μ-map** — A 4 mm FWHM Gaussian blur is applied to the μ-map before any sinogram operations. This matches the intrinsic spatial resolution of the scanner and prevents ringing artefacts from sharp CT edges propagating into the reconstructed PET.

   **Why this matters for your model:** Fine structural detail in your pseudo-CT (below ~4 mm) is blurred away before it ever influences the ACF sinogram. Spending model capacity or loss weight on sub-4mm sharpness is unlikely to improve downstream PET metrics.

4. **Resample to STIR** — Resamples the μ-map onto STIR's z-axis grid (ring spacing 3.29114 mm), snapping the origin to the STIR coordinate system. A technical prerequisite for STIR's forward projection.

5. **Compute ACF sinogram** — The μ-map is *forward projected* along every line of response (LOR) in the PET scanner geometry, computing the total integrated attenuation each annihilation photon pair experiences along that path. The result is the **attenuation correction factor (ACF)** sinogram — a per-LOR multiplier that later undoes the attenuation bias.

   Conceptually, this step is analogous to a Radon transform: the μ-map is "smeared" through the scanner geometry. An error in a single voxel affects *every* LOR passing through it, so a localised μ error (e.g. a missed bone structure) introduces a spatially coherent bias pattern across the reconstructed PET — not just a local blip.

6.–7. **Apply ACF to sinograms** — The ACF sinogram is multiplied into both the *multiplicative* sinogram (detector normalisation, decay correction) and the *additive* sinogram (scatter + random coincidences). This encodes the predicted attenuation into the reconstruction inputs.

   **Intuition:** If your pseudo-CT underestimates attenuation in a region (e.g. predicts soft tissue where bone should be), the ACF will under-correct: the reconstructor will "see" too few photons and reconstruct *lower* activity than reality in that region — even if no functional change is present. The relationship is roughly proportional: Δμ → proportional bias in reconstructed SUV.

8. **OSEM reconstruction** — Ordered Subsets Expectation Maximisation (OSEM) is an iterative algorithm for solving the Poisson maximum-likelihood reconstruction problem. It is conceptually similar to minibatch stochastic gradient ascent on the log-likelihood of the measured sinogram given the image. Each "subset" is a partition of the LORs; each iteration updates the image using one subset. The result is followed by a 4 mm Gaussian post-filter to suppress Poisson noise.

   **Why this matters for your model:** OSEM is non-linear and iterative, so ACF errors do not produce simple closed-form artefacts. However, the dominant effect of underestimated attenuation is a systematic *underestimate of SUV*, especially in deep structures (brain, heart) where photons travel through more tissue to reach the detectors. Improving your pseudo-CT in the trunk — not just the periphery — has an outsized effect on reconstruction accuracy.

9. **Convert to NIfTI** — Writes the reconstructed PET volume as a NIfTI file with the correct origin, accounting for the scanner bed position and gantry offset stored in `offset.json`.

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
