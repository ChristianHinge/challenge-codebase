# Submission Guide

This guide explains how to submit predictions at each phase of the BIC-MAC Challenge.

---

## Phase Overview

| Phase | Period | What you submit | What we return |
|-------|--------|----------------|----------------|
| **Validation** | May 15 – Aug 15 | Zip of NIfTI predictions uploaded to Codabench | All metrics on the 4 validation subjects |
| **Dry Run** | May 15 – Aug 15 | Docker container via email | CT metrics on the 4 validation subjects (or error logs if the container failed) |
| **Final Test** | Aug 15 | Docker container via email | Full evaluation on the unseen test set |

Validation and Dry Run run **concurrently** throughout the challenge. Use them to iterate on your model before the final deadline. There is no limit on submissions during either phase.

---

## Phase 1: Validation — NIfTI Upload

Submit your predictions directly as NIfTI files. No Docker container needed.

### What to submit

Run your model on the 4 validation subjects (you have both `features/` and `recon/` for these) and produce predictions:

1. **Pseudo-CT** (`ct.nii.gz`) — run your model on `features/`
2. **Reconstructed PET** (`pet.nii.gz`, optional) — run the reconstruction pipeline on your pseudo-CT using the provided Docker image (see [reconstruction.md](reconstruction.md))

If you only submit `ct.nii.gz`, you will receive CT metrics only. Submitting both unlocks all four metrics.

### Output requirements

- NIfTI format (`.nii.gz`)
- Same shape and affine as `features/nacpet.nii.gz`
- CT values in Hounsfield units (valid range approximately −1000 to +3000 HU)

### Zip structure

```
submission.zip
├── sub-068/
│   ├── ct.nii.gz
│   └── pet.nii.gz   # optional
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

Upload to the [Codabench competition page](https://www.codabench.org/competitions/12555/).

---

## Phase 2: Dry Run — Container Check

The dry run verifies that your Docker container runs correctly on organizer hardware **before** the final deadline. Submit as early as possible to leave time to fix issues.

We run your container on the 4 validation subjects and return either:
- **CT metrics** — your container ran successfully and produced valid pseudo-CTs
- **Error logs** — the container failed, with details of what went wrong

See [docker-packaging.md](docker-packaging.md) for how to build and test your container locally before submitting.

### Container requirements

Your container must:
- Read inputs from `/data/features/` (read-only mount)
- Write `ct.nii.gz` to `/data/output/`
- Complete within 5 minutes per subject
- Not require network access

### How to submit

Save your image and email it (or a download link) to **bic-mac-challenge@github.io**:

```bash
docker save my-model:latest | gzip > my-model.tar.gz
```

Subject line: `[DRY-RUN] <TeamName>`

Include in the body: team name, Docker image name and tag, and a short description of your approach.

---

## Phase 3: Final Test

Submit your Docker container by **August 15, 2026**. The container does not need to be the same as the one used for the dry run — you can continue to improve your model right up to the deadline.

We will:
1. Run your container on each unseen test subject
2. Validate the output `ct.nii.gz` (shape, affine, HU range)
3. Run the full reconstruction pipeline on each pseudo-CT
4. Evaluate all metrics against ground-truth CT and PET

Results and winner announcements: **September 1, 2026**.

### How to submit

Same as the dry run — email your container to **bic-mac-challenge@github.io** with subject:

```
[FINAL] <TeamName>
```

---

## Hardware Constraints (Phases 2 & 3)

| Resource | Specification |
|----------|--------------|
| GPU | 1× NVIDIA A40 |
| CPU | 2× Intel Xeon Gold 6346 @ 3.10 GHz |
| RAM | 128 GB |
| Wall-clock time per subject | 5 minutes |
| Network access | None (`--network none`) |

All weights and dependencies must be baked into the image. No downloads at inference time.
