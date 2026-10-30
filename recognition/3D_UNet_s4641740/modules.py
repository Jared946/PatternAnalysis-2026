"""3D counterpart of the 2D baseline's five-level U-Net.

It retains two convolutions per level, 4 pools, 4 transpose-convolution
upsamples, dropout, concatenated skips, and a 1x1 head.  Convolutions are
padded because HipMRI depth is below the 188 voxels required by an all-valid
3D U-Net.

Adapted from Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger,
“3D U-Net: Learning dense volumetric segmentation from sparse annotation,”
in Proc. MICCAI, 2016, pp. 424-432.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor

class DoubleConv3D(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        # Two padded 3D convolutions, each followed by ReLU
        self.layers = nn.Sequential(nn.Conv3d(inp,out,3,padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out,out,3,padding=1),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

class UpBlock3D(nn.Module):
    def __init__(self, inp, skip, out):
        super().__init__()

        # Upsample the input and prepare it for concatenation with the skip
        self.up=nn.ConvTranspose3d(inp,out,2,2)
        self.relu=nn.ReLU(inplace=True)
        self.conv=DoubleConv3D(out+skip,out)

    def forward(self, x, skip):
        x=self.relu(self.up(x))

        # Ensure the upsampled features align with the encoder features
        if x.shape[-3:] != skip.shape[-3:]:
            raise ValueError("D/H/W must each be divisible by 16.")
        
        # Concatenate encoder features with the upsampled decoder features
        return self.conv(torch.cat((x,skip),dim=1))

class UNet3D(nn.Module):
    """Five-level 3D U-Net with configurable base channel width."""
    def __init__(self, in_channels=1, num_classes=6, base_channels=16):
        super().__init__()
        c=base_channels
        self.in_channels,self.num_classes,self.base_channels=in_channels,num_classes,c

        # Encoder: progressively increase feature channels while reducing
        # the spatial dimensions through max pooling
        self.down0,self.pool0=DoubleConv3D(in_channels,c),nn.MaxPool3d(2)
        self.down1,self.pool1=DoubleConv3D(c,2*c),nn.MaxPool3d(2)
        self.down2,self.pool2=DoubleConv3D(2*c,4*c),nn.MaxPool3d(2)

        # Deeper encoder stages use dropout for regularisation
        self.down3,self.drop3,self.pool3 = (
            DoubleConv3D(4*c,8*c),
            nn.Dropout3d(.5),
            nn.MaxPool3d(2),
        )
        self.bottom,self.drop4 = (
            DoubleConv3D(8*c,16*c),
            nn.Dropout3d(0.5),
        )

        # Decoder: upsample features and combine them with matching
        # encoder features through skip connections
        self.up3=UpBlock3D(16*c,8*c,8*c)
        self.up2=UpBlock3D(8*c,4*c,4*c)
        self.up1=UpBlock3D(4*c,2*c,2*c)
        self.up0=UpBlock3D(2*c,c,c)

        # Map the final feature channels to one output channel per class
        self.final=nn.Conv3d(c,num_classes,1)

    def forward(self,x:Tensor):
        # Validate the expected batch, channel and spatial dimensions
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [N,{self.in_channels},D,H,W], got {tuple(x.shape)}")
        if any(n<16 or n%16 for n in x.shape[-3:]):
            raise ValueError("Each D/H/W dimension must be a multiple of 16 and >=16.")
        
        # Encoder: save feature maps for the decoder's skip connections
        d0=self.down0(x)
        d1=self.down1(self.pool0(d0))
        d2=self.down2(self.pool1(d1))
        d3=self.drop3(self.down3(self.pool2(d2)))

        # Bottleneck: deepest feature representation after the final pool
        x=self.drop4(self.bottom(self.pool3(d3)))

        # Decoder: progressively recover spatial resolution and produce
        # per-voxel class logits
        return self.final(self.up0(self.up1(self.up2(self.up3(x,d3),d2),d1),d0))
