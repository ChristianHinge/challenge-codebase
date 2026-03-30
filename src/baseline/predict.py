import argparse
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

from unet import build_model
from dataset import get_case_features


MODEL_PATH = Path(__file__).parent / "weights/best_model.pth"
MODEL_PATH = Path("/sonne/hinge/Projects/challenge-codebase/src/baseline/outputs4/checkpoints/best_model.pth")
PATCH_SIZE = (192, 192, 192)
SW_BATCH = 2
OVERLAP = 0.5


def predict(features_dir, out_path):

    transforms = Compose([
        LoadImaged(keys=["nacpet"]),
        EnsureChannelFirstd(keys=["nacpet"]),
        NormalizeIntensityd(keys=["nacpet"], nonzero=False, subtrahend=[0], channel_wise=True),
        ConcatItemsd(keys=["nacpet"], name="input"),
        EnsureTyped(keys=["input"]),
    ])

    device = "cuda"

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    case = get_case_features(features_dir)
    data = transforms(case)

    x = data["input"].unsqueeze(0).to(device)
    print("Sliding window inference...")
    with torch.no_grad():
        pred = sliding_window_inference(
            x, PATCH_SIZE, SW_BATCH, model,
            overlap=OVERLAP, mode="gaussian", progress=True,
        )

    pred_hu = pred.cpu().numpy()[0, 0] * 3000 - 1000
    affine = data["nacpet"].meta["affine"].numpy()

    print("Saving...")
    nib.save(nib.Nifti1Image(pred_hu, affine), out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_dir", help="Path to the subject's features/ folder")
    parser.add_argument("out_path", help="Path to save the predicted ct.nii.gz")
    args = parser.parse_args()

    predict(args.features_dir, args.out_path)
