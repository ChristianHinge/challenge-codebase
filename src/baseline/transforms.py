from monai.transforms import *


#NOTE: The baseline uses only the NAC-PET as input
# however, your model may use all images and metadata available
# under the /features folder

def get_transforms(patch_size, num_samples=2):

    transforms = Compose(
        [

            LoadImaged(keys=["nacpet", "ct", "prediction_mask"]),

            EnsureChannelFirstd(keys=["nacpet", "ct", "prediction_mask"]),

            NormalizeIntensityd(
                keys=["nacpet"],
                nonzero=False,
                channel_wise=True,
                subtrahend=[0]
                
            ),

            ScaleIntensityRanged(
                keys=["ct"],
                a_min=-1000,
                a_max=2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),

            RandGaussianNoised(keys=["nacpet"], prob=0.5, mean=0.0, std=0.05),
            RandScaleIntensityd(keys=["nacpet"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["nacpet"], offsets=0.1, prob=0.5),
            RandGaussianSmoothd(
                keys=["nacpet"],
                sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
                prob=0.3,
            ),

            RandAffined(
                keys=["nacpet", "ct", "prediction_mask"],
                prob=0.5,
                rotate_range=(0.087, 0.087, 0.087),  # ±5°
                scale_range=(0.05, 0.05, 0.05),       # ±5%
                mode=("bilinear", "bilinear", "nearest"),
                padding_mode="border",
            ),

            # now input is just nacpet
            ConcatItemsd(
                keys=["nacpet"],
                name="input"
            ),

            RandSpatialCropSamplesd(
                keys=["input", "ct", "prediction_mask"],
                roi_size=patch_size,
                random_size=False,
                num_samples=num_samples
            ),

            EnsureTyped(keys=["input", "ct", "prediction_mask"]),

        ]
    )

    return transforms