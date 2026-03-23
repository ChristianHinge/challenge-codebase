# Docker Packaging Guide

Your container is run once per subject with two volume mounts:

```bash
docker run --rm \
  --memory 128g \
  --network none \
  --gpus all \
  -v /path/to/subject/features:/data/features:ro \
  -v /path/to/output:/data/output \
  <your-image>
```

| Mount | Mode | Contents |
|-------|------|---------|
| `/data/features/` | read-only | NAC-PET, MRI, topogram, metadata for one subject |
| `/data/output/` | read-write | write your prediction here |

Your container must write the predicted CT to **`/data/output/ct.nii.gz`** — a NIfTI file in Hounsfield units with the same shape and affine as `features/nacpet.nii.gz`.

---

## Requirements

- All model weights baked into the image — network is disabled at runtime (`--network none`)
- Output within 5 minutes (hardware: 1× NVIDIA A40, 2× Xeon Gold 6346, 128 GB RAM)
- No other mounts — do not rely on paths outside `/data/`

---

## Baseline as a Starting Point

The baseline Dockerfile lives at `src/baseline/v2/inference/docker/Dockerfile`. It uses a PyTorch base image, installs dependencies from `requirements.txt`, copies code and weights, and sets the inference script as the entrypoint. Use it as a template.

The key adaptation for any submission is reading inputs from `/data/features/` and writing output to `/data/output/`:

```python
FEATURES_DIR = Path("/data/features")
OUTPUT_DIR   = Path("/data/output")

# ...run your model...

# Copy affine from NAC-PET to guarantee shape/affine match
ref = nib.load(str(FEATURES_DIR / "nacpet.nii.gz"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
nib.save(nib.Nifti1Image(pred_hu, ref.affine, ref.header), str(OUTPUT_DIR / "ct.nii.gz"))
```

---

## Submitting

Save your image and email it (or a download link) to **bic-mac-challenge@github.io**:

```bash
docker save my-model:latest | gzip > my-model.tar.gz
```

Subject line: `[DRY-RUN] <TeamName>` or `[FINAL] <TeamName>`

See [submission-guide.md](submission-guide.md) for phase details.

---

## Common Pitfalls

**Hardcoded paths** — make sure your container reads from `/data/features/`, not from training-time paths.

**Affine mismatch** — always copy the header from `features/nacpet.nii.gz` when saving output; don't derive it from an intermediate resampled volume.

**Network downloads at runtime** — `torch.hub`, `huggingface_hub`, etc. will fail. Bake weights in during `docker build`.
