from monai.transforms import *


#NOTE: The baseline uses only the NAC-PET as input
# however, your model may use all images and metadata available
# under the /features folder

def get_train_transforms(patch_size):

    transforms = Compose(
        [

            LoadImaged(keys=["nacpet", "ct"]),

            EnsureChannelFirstd(keys=["nacpet", "ct"]),

            NormalizeIntensityd(
                keys=["nacpet"],
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

            # now input is just nacpet
            ConcatItemsd(
                keys=["nacpet"],
                name="input"
            ),

            RandSpatialCropSamplesd(
                keys=["input","ct"],
                roi_size=patch_size,
                random_size=False,
                num_samples=2
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