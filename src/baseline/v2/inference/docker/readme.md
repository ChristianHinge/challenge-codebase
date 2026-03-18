# 🧠 Pseudo-CT Inference Docker

This repository provides a **Dockerized inference pipeline** for generating pseudo-CT images from NAC PET input using a trained 3D U-Net model.

---

## 🚀 Features

* ✅ GPU-accelerated inference (CUDA)
* ✅ Sliding window inference (overlap = 0.75)
* ✅ PET-only input
* ✅ Outputs CT in Hounsfield Units (HU)
* ✅ Clean CLI interface
* ✅ Runtime + inference timing logs

---

## 📦 Requirements

* Docker installed
* NVIDIA GPU + drivers
* NVIDIA Container Toolkit (`--gpus all` support)

---

## 🏗️ Build Docker Image

Navigate to the docker directory:

```bash
cd inference/docker
```

Build the image:

```bash
docker build -t pseudoct .
```

---

## ▶️ Run Inference

### 🔹 Basic Usage

```bash
docker run --rm \
  --gpus all \
  --user $(id -u):$(id -g) \
  -v /absolute/path/to/data:/data \
  pseudoct \
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
  pseudoct bash

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

| Step  | Command                                            |
| ----- | -------------------------------------------------- |
| Build | `docker build -t pseudoct .`                       |
| Run   | `docker run ... pseudoct --input ... --output ...` |

---
