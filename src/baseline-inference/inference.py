import argparse
import time
import warnings

import torch
import nibabel as nib
from pathlib import Path

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
)

from models.unet import build_model

warnings.filterwarnings("ignore")

# -----------------------------
# ARGUMENTS
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to nacpet.nii.gz")
parser.add_argument("--output", required=True, help="Path to save pseudo CT")
args = parser.parse_args()

total_start = time.time()


INPUT_PATH = Path(args.input)
OUTPUT_PATH = Path(args.output)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "weights/best_model.pth"
PATCH_SIZE = (192,192,192)
SW_BATCH = 2
OVERLAP = 0.5


# -----------------------------
# TRANSFORMS (PET only)
# -----------------------------

transforms = Compose([
    LoadImaged(keys=["pet"]),
    EnsureChannelFirstd(keys=["pet"]),
    NormalizeIntensityd(keys=["pet"], nonzero=True, channel_wise=True),
    ConcatItemsd(keys=["pet"], name="input"),
    EnsureTyped(keys=["input"])
])


# -----------------------------
# MODEL
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = build_model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# -----------------------------
# INFERENCE
# -----------------------------

data = {"pet": INPUT_PATH}

data = transforms(data)

x = data["input"].unsqueeze(0).to(device)

with torch.no_grad():

    pred = sliding_window_inference(
        x,
        PATCH_SIZE,
        SW_BATCH,
        model,
        overlap=OVERLAP,
        mode="gaussian",
        progress=True
    )

pred = pred.cpu().numpy()[0,0]

# convert back to HU
pred = pred * 3000 - 1000

ref = nib.load(str(INPUT_PATH))

nib.save(
    nib.Nifti1Image(pred, ref.affine, ref.header),
    str(OUTPUT_PATH)
)

print("Saved:", OUTPUT_PATH)

total_end = time.time()
print(f"Total runtime: {total_end - total_start:.2f} seconds")