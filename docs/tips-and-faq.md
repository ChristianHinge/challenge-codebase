# Tips & FAQ

---

## General

**Do I need to understand PET reconstruction to participate?**

No. You only need to predict a pseudo-CT from the input features. The reconstruction pipeline is provided and run for you. See [pet-background.md](pet-background.md) if you want to understand what the reconstruction does and why CT quality matters for PET accuracy.

**Can I use any model architecture?**

Yes — any architecture is allowed. The baseline is a 3D UNet trained on NAC-PET only. You are expected to improve on it, especially by incorporating MRI and topogram inputs.

**Can I use external data or pre-trained weights?**

Yes, as long as it is publicly available and you disclose it in your submission.

---

## Data

**What subjects have sinogram data for local reconstruction?**

Only subjects in the full `train/` split (8 subjects) and all 4 `val/` subjects include `recon/` data. The remaining 67 training subjects have `features/` and `ct-label/` only — you can train and evaluate CT metrics on them, but cannot run closed-loop PET reconstruction locally.

**What does the face mask do?**

The face region is replaced with ground-truth CT values before reconstruction (step 2 of the pipeline). This means errors in face prediction do not affect PET or CT metrics — your model is only evaluated on the body region. `features/mri_face_mask.nii.gz` marks the corresponding region in MRI space if you want to apply the same masking during training.

**Why is `prediction_mask.nii.gz` in `ct-label/`?**

It marks the voxels your model is responsible for predicting (body minus face and scanner bed). During training you may want to restrict your loss to this mask so the model is not penalised for face/bed regions that are overwritten anyway.

**The MRI comes in chunks — do I need to stitch them?**

Pre-stitched versions (`mri_combined_in_phase.nii.gz`, `mri_combined_out_phase.nii.gz`) are provided if you want a single whole-body volume. The individual chunks (`mri_chunk_{0-3}_{in/out}_phase.nii.gz`) are available if you prefer to work per bed position.

---

## Model Output

**What format does my pseudo-CT need to be in?**

A NIfTI file (`.nii.gz`) in Hounsfield units, with the same shape and affine as `features/nacpet.nii.gz`. Copying the header directly from the NAC-PET when saving is the safest approach:

```python
ref = nib.load("features/nacpet.nii.gz")
nib.save(nib.Nifti1Image(pred_hu, ref.affine, ref.header), "ct.nii.gz")
```

**What HU range should my output cover?**

Roughly −1000 (air) to +3000 HU (dense bone). The reconstruction pipeline will raise an error if the minimum HU is below −1100 or the maximum exceeds a plausible upper bound, as this usually indicates a unit or scaling error.

**My affine does not exactly match — will the pipeline reject it?**

The pipeline checks for exact affine equality. Floating-point drift from resampling operations is a common cause of mismatches. Copying the affine from the NAC-PET rather than deriving it from an intermediate volume avoids this.

---

## Reconstruction

**Do I need to install STIR locally?**

Not for submission. The Docker image (`ghcr.io/bic-mac-challenge/recon:latest`) includes STIR and all dependencies. Only use the direct Python path (`src/recon/main.py`) if you have a local STIR build.

**Reconstruction is slow — how long should I expect it to take?**

Roughly 10–20 minutes per subject on a modern CPU, dominated by the OSEM reconstruction step (step 9). Intermediate outputs are cached, so re-runs resume from where they left off unless `OVERWRITE=1` is set.

**How do I debug a failed reconstruction?**

Check `output_dir/intermediates/recon.log` for the full STIR log. Rerun with `VERBOSE=1` (Docker) or `-v` (Python) to stream STIR output to the terminal in real time.

---

## Evaluation

**Which metrics are reported on the leaderboard?**

Four metrics in total: Whole-body SUV MAE, Brain Outlier Score, Organ Bias, and CT μ-MAE. See the evaluation section of the main README for descriptions. The Brain Outlier Score is a dataset-level metric — it cannot be computed per subject.

**Can I evaluate without running reconstruction?**

Yes — CT μ-MAE only requires your pseudo-CT and the ground-truth CT. Pass `--pred_ct` without `--pred_pet` to `eval_subject.py`:

```bash
python src/evaluation/eval_subject.py \
  --subject_dir data/sub-000 \
  --pred_ct outputs/sub-000/ct.nii.gz
```

**My training subjects don't have `pet-label/` — can I still compute PET metrics?**

Only on the 8 fully-labelled training subjects that include `pet-label/` and `recon/`. On the other 67 you can run CT metrics only.

---

## Submission

**Do I need to submit both a pseudo-CT and a PET?**

For the Validation phase (NIfTI upload), you can submit CT-only, PET-only, or both. Submitting both unlocks all four metrics. For the Final Test phase, the organizers run reconstruction themselves from your pseudo-CT, so you only submit the container — no PET is needed.

**My dry run failed — how many times can I resubmit?**

There is no limit on dry-run submissions during the pre-evaluation period (May 15 – Jun 15). Fix the issue and resubmit.

**Does my final container need to be the same as the dry-run container?**

No. You can retrain or update your model between the dry run and the final submission deadline (August 15, 2026).

---

## Common Pitfalls

**Shape or affine mismatch at reconstruction time**
Always derive the output shape and affine from `features/nacpet.nii.gz`, not from an intermediate resampled volume.

**Network downloads at runtime**
The container runs with `--network none`. Pre-download all weights and bake them into the image during `docker build`. Calls to `torch.hub`, `huggingface_hub`, or any HTTP request will fail silently or raise an error.

**Hardcoded training paths in the container**
Ensure your container reads from `/data/features/` and writes to `/data/output/ct.nii.gz`. Test locally with the exact `docker run` command from [docker-packaging.md](docker-packaging.md) before submitting.

**HU scaling errors**
A common mistake is outputting values in a different scale (e.g., normalised 0–1, or in μ units). Check that air voxels are near −1000 HU and soft tissue near 0–100 HU before submitting.
