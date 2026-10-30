"""PyTorch implementation of the original 2015 Caffe U-Net architecture.

The network intentionally uses *valid* (padding=0) 3x3 convolutions.  This
matches ``phseg_v5-train.prototxt`` from the supplied u-net-release exactly:
the prediction is therefore smaller than the input and decoder skip features
are centre-cropped before concatenation.

Adapted from: O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional
networks for biomedical image segmentation,” in Proc. MICCAI, 2015,
pp. 234-241.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Two valid 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # Apply two consecutive valid convolutions with ReLU activations
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


def center_crop(source: Tensor, target: Tensor) -> Tensor:
    """Centre-crop ``source`` spatially to match ``target``.

    The Caffe ``CROP`` layers crop the encoder feature map to the decoder
    feature map.  A useful explicit error is preferable to an opaque concat
    error when an input slice is too small for this valid-convolution U-Net.
    """

    # Calculate the spatial dimensions and ensure the source can be cropped
    source_h, source_w = source.shape[-2:]
    target_h, target_w = target.shape[-2:]
    if source_h < target_h or source_w < target_w:
        raise ValueError(
            "Cannot crop skip feature map of size "
            f"{source_h}x{source_w} to {target_h}x{target_w}. "
            "Use input slices large enough for four valid-convolution U-Net levels."
        )
    # Calculate the central crop offsets and match the target dimensions
    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2
    return source[..., top : top + target_h, left : left + target_w]


class UpBlock(nn.Module):
    """Caffe-style deconvolution, crop/concat skip, then two valid convolutions."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        # Upsample decoder features and combine them with cropped encoder features
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.activation = nn.ReLU(inplace=True)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.activation(self.up(x))
        skip = center_crop(skip, x)
        return self.conv(torch.cat((x, skip), dim=1))


class UNet2D(nn.Module):
    """The original Ronneberger et al. U-Net topology used by u-net-release.

    ``base_channels`` defaults to 64, giving the precise 64/128/256/512/1024
    feature widths of the source model.  It is configurable only to make
    memory-constrained smoke tests possible.
    """

    def __init__(
        self, in_channels: int = 1, num_classes: int = 6, base_channels: int = 64
    ) -> None:
        super().__init__()
        c = base_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

        # Define the encoder blocks, progressively increasing feature channels
        self.down0 = DoubleConv(in_channels, c)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(c, 2 * c)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(2 * c, 4 * c)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = DoubleConv(4 * c, 8 * c)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define the bottleneck with the largest feature-channel dimension
        self.bottom = DoubleConv(8 * c, 16 * c)
        self.dropout4 = nn.Dropout2d(p=0.5)

        # Define decoder blocks that upsample and combine encoder skip features
        self.up3 = UpBlock(16 * c, 8 * c, 8 * c)
        self.up2 = UpBlock(8 * c, 4 * c, 4 * c)
        self.up1 = UpBlock(4 * c, 2 * c, 2 * c)
        # The reference prototxt has 128 output channels in this up-convolution
        self.up0 = UpBlock(2 * c, c, c)

        # Map the final feature maps to the six segmentation classes
        self.final = nn.Conv2d(c, num_classes, kernel_size=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # Validate the input tensor dimensions and channel count
        if x.ndim != 4:
            raise ValueError(f"Expected input [N, C, H, W], got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")

        # Pass through the encoder and bottleneck, retaining skip features
        d0 = self.down0(x)
        d1 = self.down1(self.pool0(d0))
        d2 = self.down2(self.pool1(d1))
        d3 = self.dropout3(self.down3(self.pool2(d2)))
        bottom = self.dropout4(self.bottom(self.pool3(d3)))

        # Pass through the decoder, combining features from the encoder
        x = self.up3(bottom, d3)
        x = self.up2(x, d2)
        x = self.up1(x, d1)
        x = self.up0(x, d0)
        return self.final(x)

class PaperUNet2D(UNet2D):
    """Fixed-width paper U-Net with He-normal weight initialisation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=64,
        )
        # Apply He-normal initialisation to the convolutional layers
        self.apply(paper_he_initialisation)


def crop_mask_to_logits(mask: Tensor, logits: Tensor) -> Tensor:
    """Centre-crop a segmentation mask to the model output size.

    Supports class-index masks [N, H, W] and one-hot masks
    [N, C, H, W].
    """
    # Validate mask dimensions and determine the required output size
    if mask.ndim not in (3, 4):
        raise ValueError(
            f"Expected mask [N, H, W] or [N, C, H, W], "
            f"got shape {tuple(mask.shape)}"
        )

    target_h, target_w = logits.shape[-2:]
    h, w = mask.shape[-2:]

    if h < target_h or w < target_w:
        raise ValueError("Mask is smaller than the network output.")

    # Centre-crop the mask to align with the network output dimensions
    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return mask[..., top:top + target_h, left:left + target_w]

def paper_he_initialisation(module: nn.Module) -> None:
    """Initialise convolution weights using He-normal scheme."""
    # Apply He-normal weight initialisation to convolutional layers
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        kernel_h, kernel_w = module.kernel_size
        fan_in = (
            module.in_channels // module.groups
        ) * kernel_h * kernel_w

        nn.init.normal_(
            module.weight,
            mean=0.0,
            std=math.sqrt(2.0 / fan_in),
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

