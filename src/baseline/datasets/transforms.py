from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ConcatItemsd,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
    MapTransform,
)

import torch


class RepeatTopogramZ(MapTransform):
    """
    Repeat the topogram slice along the Z axis
    so it matches the depth of the PET volume.
    """

    def __call__(self, data):

        d = dict(data)

        pet = d["pet"]
        topo = d["topogram"]

        target_depth = pet.shape[-1]

        # topo shape: (C,H,W,1)
        topo = topo.repeat(1, 1, 1, target_depth)

        d["topogram"] = topo

        return d


def get_train_transforms(patch_size, spacing):

    transforms = Compose(

        [

            LoadImaged(keys=["pet","topogram","mri_in","mri_out","ct"]),

            EnsureChannelFirstd(keys=["pet","topogram","mri_in","mri_out","ct"]),

            EnsureTyped(keys=["pet","topogram","mri_in","mri_out","ct"]),

            # repeat scout radiograph along Z
            RepeatTopogramZ(keys=["topogram"]),

            NormalizeIntensityd(
                keys=["pet","topogram","mri_in","mri_out"],
                nonzero=True,
                channel_wise=True
            ),

            ConcatItemsd(
                keys=["pet","topogram","mri_in","mri_out"],
                name="input",
            ),

            RandSpatialCropd(
                keys=["input","ct"],
                roi_size=patch_size,
                random_size=False
            ),

            RandFlipd(
                keys=["input","ct"],
                spatial_axis=0,
                prob=0.5
            ),

            RandFlipd(
                keys=["input","ct"],
                spatial_axis=1,
                prob=0.5
            ),

            EnsureTyped(keys=["input","ct"]),

        ]
    )

    return transforms