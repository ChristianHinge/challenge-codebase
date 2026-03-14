import torch
import nibabel as nib
import numpy as np

from monai.inferers import sliding_window_inference

from models.unet import build_model


def run_inference(input_tensor, checkpoint):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)

    model.load_state_dict(torch.load(checkpoint))

    model.eval()

    with torch.no_grad():

        pred = sliding_window_inference(

            inputs=input_tensor,

            roi_size=(128,128,128),

            sw_batch_size=1,

            predictor=model,

        )

    return pred


def save_prediction(pred, path):

    arr = pred.cpu().numpy()[0,0]

    nib.save(
        nib.Nifti1Image(arr, np.eye(4)),
        path,
    )