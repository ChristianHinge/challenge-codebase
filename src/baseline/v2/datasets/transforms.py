from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    ScaleIntensityRanged,
    ConcatItemsd,
    RandSpatialCropd,
    RandFlipd,
    EnsureTyped,
)


def get_train_transforms(patch_size, spacing):

    transforms = Compose(
        [

            LoadImaged(keys=["pet", "ct"]),

            EnsureChannelFirstd(keys=["pet", "ct"]),

            NormalizeIntensityd(
                keys=["pet"],
                nonzero=True,
                channel_wise=True
            ),

            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # now input is just PET
            ConcatItemsd(
                keys=["pet"],
                name="input"
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