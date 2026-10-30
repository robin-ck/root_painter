"""
Multiclass segmentation model taken from pythonpanic
(https://github.com/nvdeijk/pythonpanic, source/model_aspp.py):
a segmentation_models_pytorch U-Net with a ResNet34 encoder and an
ASPP block (similar to Wu et al.) between encoder and decoder.

The network outputs one logit channel per foreground class (sigmoid,
no background channel). Background is a pixel where no class is above
the threshold.

RootPainter feeds the network tiles of in_w and expects out_w = in_w - 72
back (the original UNetGNRes uses valid convolutions). TiledSegModel
wraps the U-Net so it fits that tiling scheme: it pads the input up to a
multiple of 32 (needed by the ResNet encoder) and crops the centre out_w
region of the output, so the 36 px context margin is kept.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""
# pylint: disable=C0111, R0913, W0221
import math

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

# in_w - out_w = 72 for every valid patch size, so 36 px each side.
TILE_MARGIN = 36


class SeparableAtrousConv(nn.Module):
    """ Depthwise separable atrous convolution """
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   padding=dilation, dilation=dilation,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.pointwise(self.depthwise(x))))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=128,
                 dilation_rates=(6, 12, 18), dropout=0.5):
        super().__init__()
        # 1x1 convolution: local information
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )
        # Atrous convolutions: different spatial scales
        self.branch2 = SeparableAtrousConv(in_channels, out_channels, dilation_rates[0])
        self.branch3 = SeparableAtrousConv(in_channels, out_channels, dilation_rates[1])
        self.branch4 = SeparableAtrousConv(in_channels, out_channels, dilation_rates[2])
        # Global image context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ELU(inplace=True)
        )
        # Combine all 5 branches
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ELU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        size = x.shape[-2:]
        b5 = F.interpolate(self.global_pool(x), size=size,
                           mode="bilinear", align_corners=False)
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x),
                       self.branch4(x), b5], dim=1)
        return self.project(x)


class UnetASPP(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=4):
        super().__init__()
        self.unet = smp.Unet(encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             in_channels=in_channels,
                             classes=classes,
                             activation=None)
        # deepest encoder layer, 512 for ResNet34
        bottleneck_channels = self.unet.encoder.out_channels[-1]
        self.aspp = ASPP(in_channels=bottleneck_channels, out_channels=128,
                         dilation_rates=(6, 12, 18), dropout=0.5)

    def forward(self, x):
        features = list(self.unet.encoder(x))
        # ASPP only on the deepest encoder representation
        features[-1] = self.aspp(features[-1])
        try:
            decoder_output = self.unet.decoder(features)
        except TypeError:
            # older segmentation_models_pytorch versions take *features
            decoder_output = self.unet.decoder(*features)
        return self.unet.segmentation_head(decoder_output)


class TiledSegModel(nn.Module):
    """ Make the U-Net return the centre (in_w - 72) region of each tile """
    def __init__(self, num_classes, encoder_weights="imagenet"):
        super().__init__()
        self.num_classes = num_classes
        self.model = UnetASPP(encoder_weights=encoder_weights,
                              classes=num_classes)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = math.ceil(h / 32) * 32 - h
        pad_w = math.ceil(w / 32) * 32 - w
        top, left = pad_h // 2, pad_w // 2
        if pad_h or pad_w:
            x = F.pad(x, (left, pad_w - left, top, pad_h - top), mode='reflect')
        out = self.model(x)
        y0 = top + TILE_MARGIN
        x0 = left + TILE_MARGIN
        return out[:, :, y0:y0 + h - 2 * TILE_MARGIN, x0:x0 + w - 2 * TILE_MARGIN]


def create_model(num_classes, pretrained=True):
    """ New model with ImageNet encoder weights (as in pythonpanic).
        Falls back to random weights if they can't be downloaded. """
    if pretrained:
        try:
            return TiledSegModel(num_classes, encoder_weights="imagenet")
        except Exception as e:
            print('Could not load ImageNet encoder weights, '
                  'using random weights instead:', e)
    return TiledSegModel(num_classes, encoder_weights=None)


def num_classes_from_state_dict(state_dict):
    for key, val in state_dict.items():
        if key.endswith('segmentation_head.0.weight'):
            return val.shape[0]
    raise ValueError('Model file is not a multiclass U-Net ASPP model '
                     '(no segmentation head found). Models from the original '
                     'RootPainter U-Net can not be used with this trainer.')
