import torch
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    EnsureTyped,
    Lambdad,
)

from models.unet import build_model


# -----------------------------
# CONFIG
# -----------------------------

DATA_ROOT = Path("/depict/users/hinge/shared/bic-mac-data/hf_dataset/train")
MODEL_PATH = "outputs/checkpoints/model_epoch_99.pth"
OUTPUT_DIR = Path("outputs/pseudo_ct")

PATCH_SIZE = (128,128,128)
SW_BATCH = 2
OVERLAP = 0.75

SUBJECTS = [
    "sub-068",
    "sub-074",
    "sub-081",
    "sub-087",
    "sub-093",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# PREPROCESS (match training)
# -----------------------------

transforms = Compose(
[
    LoadImaged(keys=["pet","topogram","mri_in","mri_out"]),

    EnsureChannelFirstd(keys=["pet","topogram","mri_in","mri_out"]),

    # repeat topogram depth (same as training)
    Lambdad(keys=["topogram"], func=lambda x: x.repeat(1,1,1,531)),

    # normalize PET + MRI (same as training)
    NormalizeIntensityd(keys=["pet","mri_in","mri_out"]),

    # combine modalities
    ConcatItemsd(
        keys=["pet","topogram","mri_in","mri_out"],
        name="input"
    ),

    EnsureTyped(keys=["input"])
]
)


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

for name in tqdm(SUBJECTS):

    sub = DATA_ROOT / name

    data = {
        "pet": sub/"features/nacpet.nii.gz",
        "topogram": sub/"features/topogram.nii.gz",
        "mri_in": sub/"features/mri_combined_in_phase.nii.gz",
        "mri_out": sub/"features/mri_combined_out_phase.nii.gz",
    }

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
        )

    pred = pred.cpu().numpy()[0,0]

    # convert normalized prediction back to HU
    pred = pred * 3000 - 1000

    ref = nib.load(str(sub/"features/nacpet.nii.gz"))

    out_file = OUTPUT_DIR / f"{name}_pseudo_ct.nii.gz"

    nib.save(
        nib.Nifti1Image(pred, ref.affine, ref.header),
        str(out_file)
    )

    print("Saved:", out_file)