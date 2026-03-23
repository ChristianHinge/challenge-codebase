# Submission Guide

This guide explains how to submit predictions at each phase of the BIC-MAC Challenge.

---

## Phase Overview

| Phase | Period | What you submit | What we run |
|-------|--------|----------------|-------------|
| **Validation** | May 15 – Jun 15 | NIfTI predictions uploaded to Codabench | Evaluation metrics directly |
| **Dry Run** | May 15 – Jun 15 | Docker container via email | Container only (no recon, no scoring) |
| **Final Test** | Jun 15 – Aug 15 | Docker container via email | Container + full recon + evaluation |

Phases 1 (Validation) and 2 (Dry Run) run **concurrently**. You can — and should — do both during the pre-evaluation period before the final test opens.

---

## Phase 1: Validation — NIfTI Upload (n=4)

This phase lets you evaluate your model on the 4 validation subjects using Codabench, without needing a Docker container.

### What you have locally

Validation subjects include all the inputs your model needs plus the sinogram data required for reconstruction:

```
val/sub-XXX/
├── features/     # model inputs (nacpet, MRI, topogram, metadata)
└── recon/        # sinogram data (mult, add, prompts sinograms + offset.json)
```

Labels are **not** included — Codabench holds the ground truth.

### Step 1: Generate your pseudo-CT predictions

Run your model on each of the 4 validation subjects and save the output as `ct.nii.gz` in Hounsfield units.

```bash
# Example — adapt to your model's interface
python your_model.py val/sub-068/features/ outputs/sub-068/ct.nii.gz
python your_model.py val/sub-074/features/ outputs/sub-074/ct.nii.gz
python your_model.py val/sub-081/features/ outputs/sub-081/ct.nii.gz
python your_model.py val/sub-087/features/ outputs/sub-087/ct.nii.gz
```

**Output requirements:**
- NIfTI format (`.nii.gz`)
- Same shape and affine as `features/nacpet.nii.gz`
- Values in Hounsfield units (valid range approximately −1000 to +3000 HU)

### Step 2: Optionally generate PET predictions (for PET metrics)

If you want PET-based metrics (Whole-body SUV MAE, Brain Outlier Score, Organ Bias), you also need to run the reconstruction pipeline on your pseudo-CT to produce `pet.nii.gz`.

Use the organizer-provided Docker image (see [website](https://bic-mac-challenge.github.io/)):

```bash
docker run --rm \
  -v val/sub-068/recon:/data/recon \
  -v outputs/sub-068/ct.nii.gz:/data/ct/ct.nii.gz \
  -v outputs/sub-068/recon_out:/data/output \
  ghcr.io/bic-mac-challenge/recon:latest
# reconstructed PET → outputs/sub-068/recon_out/pet.nii.gz
```

See the [main README](../README.md#reconstruction-pipeline-srcrecon) for full reconstruction instructions.

### Step 3: Evaluate locally (optional)

You can sanity-check your results on the 8 fully-labelled training subjects (which have ground-truth CT and PET labels) before uploading:

```bash
python src/evaluation/eval.py \
  train/sub-000 \
  outputs/sub-000/pet.nii.gz \
  outputs/sub-000/ct.nii.gz \
  -all
```

See [src/evaluation/readme.md](../src/evaluation/readme.md) for full usage.

### Step 4: Upload to Codabench

Organize your predictions into a zip archive with this structure:

```
submission.zip
├── sub-068/
│   ├── ct.nii.gz
│   └── pet.nii.gz   # optional, needed for PET metrics
├── sub-074/
│   ├── ct.nii.gz
│   └── pet.nii.gz
├── sub-081/
│   ├── ct.nii.gz
│   └── pet.nii.gz
└── sub-087/
    ├── ct.nii.gz
    └── pet.nii.gz
```

Upload this zip to the Codabench competition page (see [website](https://bic-mac-challenge.github.io/) for the direct link).

---

## Phase 2: Dry Run — Container Check (concurrent with Phase 1)

The dry run verifies that your Docker container runs correctly on organizer hardware **before** the final test. No reconstruction or scoring is done — a pass/fail result with any error logs is returned.

### What you need

A Docker container that:
- Reads inputs from `/data/features/` (read-only mount)
- Writes `ct.nii.gz` to `/data/output/`
- Runs within 5 minutes on the validation subjects
- Does not require network access

See [docker-packaging.md](docker-packaging.md) for how to build and test your container locally.

### How to submit

1. Save your image to a tar archive:

```bash
docker save my-model:latest | gzip > my-model.tar.gz
```

2. Email the archive (or a link to it) to **bic-mac-challenge@github.io** with subject line:

```
[DRY-RUN] <TeamName>
```

Include in the email body:
- Team name
- Short description of your approach
- Docker image name and tag
- Any environment variables or flags required at runtime (if any)

### What you receive back

- **Pass**: Your container ran successfully on all 4 validation subjects — you are cleared for the final test.
- **Fail**: Error logs showing where the container failed. Fix the issue and resubmit.

There is no limit on dry-run submissions during the pre-evaluation period.

---

## Phase 3: Final Test — Container Submission (Jun 15 – Aug 15)

The final evaluation runs your container on the **unseen test set**. You do not have access to these subjects' features or recon data.

The organizers will:

1. Run your container on each test subject (same `docker run` command as the dry run)
2. Validate the output `ct.nii.gz` (shape, affine, HU range)
3. Run the full STIR reconstruction pipeline on each pseudo-CT
4. Evaluate all metrics against ground-truth CT and PET

Results and winner announcements: **September 1, 2026**.

### How to submit

Same process as the dry run — email your container to **bic-mac-challenge@github.io** with subject:

```
[FINAL] <TeamName>
```

**Deadline: August 15, 2026.**

You may update your container after the dry run (e.g., to fix issues found, retrain with more data). The container submitted for the final test does not need to be the same as the dry-run container.

---

## Resource Constraints (Phases 2 & 3)

These constraints are enforced identically in both the dry run and the final test:

| Resource | Specification |
|----------|--------------|
| GPU | 1× NVIDIA A40 |
| CPU | 2× Intel Xeon Gold 6346 @ 3.10 GHz |
| RAM | 128 GB |
| Wall-clock time per subject | 5 minutes |
| Network access | None (`--network none`) |

All model weights and dependencies must be baked into the image. No downloads at inference time.

---

## Common Mistakes

- **Output affine mismatch**: Your `ct.nii.gz` must have the same shape and affine as `features/nacpet.nii.gz`. Copying the header from the NAC-PET is the safest approach.
- **Hardcoded paths**: Make sure your container reads from `/data/features/` and writes to `/data/output/`, not from local training paths.
- **Network downloads at runtime**: The container runs with `--network none`. Download weights during `docker build`, not during inference.
- **Output not written**: If your script crashes silently, the output file may not exist. Always verify with a quick test locally (see [docker-packaging.md](docker-packaging.md)).
