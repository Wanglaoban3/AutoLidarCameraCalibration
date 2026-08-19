"""Minimal TEED inference model.

Adapted from https://github.com/xavysp/TEED at commit
40fa4b1391dc6424f88989d0ca75d5b592c8681d (MIT License).
Copyright (c) 2022 Xavier Soria Poma.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def smish(value):
    return value * torch.tanh(torch.log(1 + torch.sigmoid(value)))


class Smish(nn.Module):
    def forward(self, value):
        return smish(value)


class DoubleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.DWconv1 = nn.Conv2d(3, 24, 3, padding=1, groups=3)
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.AF = Smish()

    def forward(self, value):
        first = self.PSconv1(self.DWconv1(self.AF(value)))
        second = self.PSconv1(self.DWconv2(self.AF(first)))
        return smish((first + second).sum(1, keepdim=True))


class _DenseLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.conv1 = nn.Conv2d(input_features, output_features, 3, padding=2)
        self.smish1 = Smish()
        self.conv2 = nn.Conv2d(output_features, output_features, 3)

    def forward(self, inputs):
        value, shortcut = inputs
        value = self.conv2(self.smish1(self.conv1(smish(value))))
        return 0.5 * (value + shortcut), shortcut


class _DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.denselayer1 = _DenseLayer(32, 48)

    def forward(self, inputs):
        return self.denselayer1(inputs)


class SingleConvBlock(nn.Module):
    def __init__(self, input_features, output_features, stride):
        super().__init__()
        self.conv = nn.Conv2d(input_features, output_features, 1, stride=stride)

    def forward(self, value):
        return self.conv(value)


class UpConvBlock(nn.Module):
    def __init__(self, input_features, scale):
        super().__init__()
        layers = []
        pads = [0, 0, 1, 3, 7]
        for index in range(scale):
            output_features = 1 if index == scale - 1 else 16
            layers.extend([
                nn.Conv2d(input_features, output_features, 1), Smish(),
                nn.ConvTranspose2d(output_features, output_features, 2 ** scale, stride=2, padding=pads[scale]),
            ])
            input_features = output_features
        self.features = nn.Sequential(*layers)

    def forward(self, value):
        return self.features(value)


class DoubleConvBlock(nn.Module):
    def __init__(self, input_features, middle_features, output_features=None, stride=1, use_activation=True):
        super().__init__()
        output_features = output_features or middle_features
        self.conv1 = nn.Conv2d(input_features, middle_features, 3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(middle_features, output_features, 3, padding=1)
        self.smish = Smish()
        self.use_activation = use_activation

    def forward(self, value):
        value = self.conv2(self.smish(self.conv1(value)))
        return self.smish(value) if self.use_activation else value


class TEED(nn.Module):
    """Tiny Efficient Edge Detector compatible with the official BIPED weight."""
    def __init__(self):
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_activation=False)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)
        self.dblock_3 = _DenseBlock()
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)
        self.block_cat = DoubleFusion()

    def forward(self, value):
        block_1 = self.block_1(value)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_3, _ = self.dblock_3((block_2_down + self.side_1(block_1), self.pre_dense_3(block_2_down)))
        outputs = [self.up_block_1(block_1), self.up_block_2(block_2), self.up_block_3(block_3)]
        outputs.append(self.block_cat(torch.cat(outputs, dim=1)))
        return outputs
