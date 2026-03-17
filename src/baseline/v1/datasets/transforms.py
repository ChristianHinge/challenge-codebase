from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    ConcatItemsd,
    RandCropByPosNegLabeld,
    RandFlipd,
    EnsureTyped,
    Lambdad
)

import torch


def repeat_topogram(data):
    """
    Convert topogram from (1,H,W,1) -> (1,H,W,D)
    so it matches PET/MRI depth.
    """

    topogram = data["topogram"]
    depth = data["pet"].shape[-1]

    if topogram.shape[-1] == 1:
        topogram = topogram.repeat(1, 1, 1, depth)

    data["topogram"] = topogram

    return data


def get_train_transforms(patch_size, spacing):

    transforms = Compose(

        [

            LoadImaged(keys=["pet", "topogram", "mri_in", "mri_out", "ct"]),

            EnsureChannelFirstd(keys=["pet", "topogram", "mri_in", "mri_out", "ct"]),

            # repeat topogram depth
            Lambdad(keys=["topogram"], func=lambda x: x.repeat(1, 1, 1, 531)),

            # normalize PET + MRI
            NormalizeIntensityd(keys=["pet", "mri_in", "mri_out"]),

            # normalize CT to 0-1 range
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # combine modalities into input tensor
            ConcatItemsd(
                keys=["pet", "topogram", "mri_in", "mri_out"],
                name="input",
            ),

            # sample useful patches instead of random air
            RandCropByPosNegLabeld(
                keys=["input", "ct"],
                label_key="ct",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
            ),

            RandFlipd(
                keys=["input", "ct"],
                spatial_axis=0,
                prob=0.5,
            ),

            RandFlipd(
                keys=["input", "ct"],
                spatial_axis=1,
                prob=0.5,
            ),

            EnsureTyped(keys=["input", "ct"]),

        ]
    )

    return transforms