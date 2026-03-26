# Baseline v2 — 3D U-Net (NAC-PET only)

A 3D U-Net that predicts pseudo-CT from NAC-PET alone. Serves as the baseline for the BIC-MAC challenge.

## Model

- **Architecture**: 3D U-Net with residual blocks, instance norm, LeakyReLU
- **Input**: NAC-PET (single channel, normalized nonzero channel-wise)
- **Output**: Pseudo-CT in Hounsfield Units (HU range: −1000 to 2000)
- **Loss**: L1, trained for 250 epochs with AdamW + cosine annealing
- **Patch size**: 192³ voxels

Despite MRI and topogram being available, this baseline uses only NAC-PET to keep the implementation minimal.

## Inference

The baseline model has already been pretrained

Run with Docker:

```bash
docker run --rm \
  --memory 120g \
  -v /path/to/sub-XXX/features:/data/features:ro \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/baseline
```

Or without Docker:

```bash
pip install -r requirements.txt
python predict.py /path/to/sub-XXX/features /path/to/output/ct.nii.gz
```

## Training

Edit `config.yaml` to set your data path, then:

```bash
python train.py
```

Weights are saved to `outputs/checkpoints/best_model.pth` (lowest validation loss).

Move new weights to `weights/best_model.pth` and build the image:

```bash
docker build -t my-baseline .
```
