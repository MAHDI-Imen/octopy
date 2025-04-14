from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def dense_layer(in_channels, out_channels, apply_activation=True):
    layer = [nn.Linear(in_channels, out_channels)]
    if apply_activation:
        layer += [nn.LeakyReLU(0.02)]
    return layer


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride_size,
        apply_norm=True,
        apply_activation=True,
    ):
        super().__init__()

        padding_size = (
            kernel_size // 2
            if isinstance(kernel_size, int)
            else (kernel_size[0] // 2, kernel_size[1] // 2)
        )

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride_size,
            padding_size,
            padding_mode="replicate",
        )

        if apply_norm:
            self.norm = nn.GroupNorm(1, out_channels, affine=True)

        if apply_activation:
            self.activation = nn.LeakyReLU(0.02)

    def forward(
        self, ft: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = self.conv(ft)

        if hasattr(self, "norm"):
            out = self.norm(out)

        if hasattr(self, "activation"):
            out = self.activation(out)

        return out


def upsample_layer(
    in_channels,
    out_channels,
    kernel_size=(3, 3),
    stride_size=(1, 1),
    apply_norm=True,
    apply_activation=True,
):
    return [
        ConvLayer(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride_size=stride_size,
            apply_norm=apply_norm,
            apply_activation=apply_activation,
        ),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
    ]
