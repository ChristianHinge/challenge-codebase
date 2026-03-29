# Baseline v2 — 3D U-Net (NAC-PET only)

A 3D U-Net that predicts pseudo-CT from NAC-PET alone. Serves as the baseline for the BIC-MAC challenge.

## Model

- **Architecture**: 3D U-Net with residual blocks, instance norm, LeakyReLU
- **Input**: NAC-PET 
- **Output**: Pseudo-CT in Hounsfield Units 
- **Loss**: L1, trained for 250 epochs with AdamW 
- **Patch size**: `192x192x192`

Despite MRI and topogram being available, this baseline uses only NAC-PET to keep the implementation minimal.

## Inference

The baseline model has already been pretrained

Run with Docker:

```bash
docker run --rm \
  --memory 120g \
  -v /path/to/sub-000/features:/data/features:ro \
  -v /path/to/output:/data/output \
  ghcr.io/bic-mac-challenge/baseline
```

Or without Docker:

```bash
pip install -r requirements.txt
python predict.py --features_dir /path/to/sub-000/features --output_ct pseudo-ct.nii.gz
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
