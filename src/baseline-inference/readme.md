# 🧠 Baseline Inference Docker (Pseudo-CT)

This repository provides a **Dockerized baseline inference pipeline** for generating pseudo-CT images from NAC PET input using a trained 3D U-Net model.

This container corresponds to the **official baseline model** of the BIC-MAC challenge.

---

## 🚀 Features

* ✅ GPU-accelerated inference (CUDA)
* ✅ Sliding window inference (overlap = 0.75)
* ✅ PET-only input (baseline setting)
* ✅ Outputs CT in Hounsfield Units (HU)
* ✅ Clean CLI interface
* ✅ Runtime + inference timing logs
* ✅ Fully self-contained (no external downloads)

---

## 📦 Requirements

* Docker installed
* NVIDIA GPU + drivers
* NVIDIA Container Toolkit (`--gpus all` support)

---

## 📦 Pull Prebuilt Image (Recommended)

```bash
docker pull ghcr.io/bic-mac-challenge/inference:latest
```

---

## 🏗️ Build Docker Image (Optional)

If you want to build locally:

```bash
docker build -t ghcr.io/bic-mac-challenge/inference:latest .
```

---

## ▶️ Run Inference

```bash
docker run --rm \
  --gpus all \
  --user $(id -u):$(id -g) \
  -v /absolute/path/to/data:/data \
  ghcr.io/bic-mac-challenge/inference:latest \
  --input /data/sub-XXX/features/nacpet.nii.gz \
  --output /data/sub-XXX/pseudo_ct.nii.gz
```

---

## 📁 Input Structure

Your data should follow:

```
data/
└── sub-XXX/
    └── features/
        └── nacpet.nii.gz
```

---

## 📁 Output

The output will be saved as:

```
data/
└── sub-XXX/
    └── pseudo_ct.nii.gz
```

---

## ⚠️ Important Notes

### 🔸 Path Mounting

Docker cannot access host paths directly. You must mount them using `-v`.

Example:

```bash
-v /home/user/data:/data
```

---

### 🔸 Permissions Fix

If you get a **permission denied error**, use:

```bash
--user $(id -u):$(id -g)
```

---

### 🔸 GPU Usage

Make sure to include:

```bash
--gpus all
```

Otherwise inference will run on CPU.

---

### 🔸 Self-contained Model

* All model weights are **baked into the container**
* No internet access is required
* Suitable for **offline evaluation environments (e.g., Codabench)**

---

## ⏱️ Output Logs

Example output:

```
Using device: cuda
Inference time: 3.21 seconds
Total runtime: 5.12 seconds
Saved: /data/sub-XXX/pseudo_ct.nii.gz
```

---

## 🧪 Debugging

### Check mounted data inside container:

```bash
docker run --rm -it \
  -v /your/data:/data \
  ghcr.io/bic-mac-challenge/inference:latest bash

ls /data
```

---

## 🛠️ Customization

You can modify:

* Patch size → `PATCH_SIZE`
* Sliding window batch → `SW_BATCH`
* Overlap → `OVERLAP`

inside `inference.py`.

---

## 📌 Summary

| Step | Command                                              |
| ---- | ---------------------------------------------------- |
| Pull | `docker pull ghcr.io/bic-mac-challenge/inference`    |
| Run  | `docker run ... ghcr.io/bic-mac-challenge/inference` |

---

## 🧠 Relation to Challenge

This container represents the **baseline pseudo-CT model** provided in:

```
src/baseline/
```

Participants are expected to **build upon and improve this baseline** using additional modalities such as MRI and topogram.
