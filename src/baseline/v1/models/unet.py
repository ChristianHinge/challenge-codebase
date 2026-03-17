import torch
from monai.networks.nets import UNet


def build_model():

    model = UNet(

        spatial_dims=3,

        in_channels=4,
        out_channels=1,

        channels=(32,64,128,256),

        strides=(2,2,2),

        num_res_units=2,

    )

    return model