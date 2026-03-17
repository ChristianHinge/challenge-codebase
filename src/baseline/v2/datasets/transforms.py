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


def get_train_transforms(patch_size, spacing):

    transforms = Compose(
        [

            LoadImaged(keys=["pet", "topogram", "mri_in", "mri_out", "ct"]),

            EnsureChannelFirstd(keys=["pet", "topogram", "mri_in", "mri_out", "ct"]),

            # expand topogram depth
            Lambdad(keys=["topogram"], func=lambda x: x.repeat(1,1,1,531)),

            # normalize PET and MRI
            NormalizeIntensityd(
                keys=["pet","mri_in","mri_out"],
                nonzero=True,
                channel_wise=True
            ),

            # normalize CT
            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            # combine modalities
            ConcatItemsd(
                keys=["pet","topogram","mri_in","mri_out"],
                name="input"
            ),

            # patch sampling
            RandCropByPosNegLabeld(
                keys=["input","ct"],
                label_key="ct",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1
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