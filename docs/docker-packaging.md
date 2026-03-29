# Docker Packaging Guide

Your container is run once per subject with two volume mounts:

```bash
timeout 300 docker run --rm \
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

The baseline Dockerfile lives at `src/baseline/Dockerfile`. It uses a PyTorch base image, installs dependencies from `requirements.txt`, copies code and weights, and sets the inference script as the entrypoint. You can use it as a template.


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

## Validating Before Submission

Before sending us your image, run it on the 4 validation subjects and upload the predictions to Codabench to confirm your container works end-to-end and your scores look reasonable. We cannot debug containers that fail silently on our infrastructure, so this step is required.

---

## Submitting

See [submission-guide.md](submission-guide.md) for full submission instructions, including how to share your image via Docker Hub or a compressed archive.

---
