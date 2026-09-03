# -*- coding: utf-8 -*-
"""A small 3D CNN regressor for 63x63x63 binary microstructures."""

import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions, then halve the resolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.body(x)


class StressCNN(nn.Module):
    """Voxel field -> one scalar (the standardised stress).

    Four stride-2 stages take 63^3 down to 3^3, a global average pool removes
    the remaining spatial dimensions, and a small head produces the scalar.
    Pooling globally also makes the network translation invariant, which
    matches the periodic microstructures it is fed.
    """

    def __init__(self, base_channels=None, dropout=None, in_channels=1):
        super().__init__()
        base = base_channels or config.BASE_CHANNELS
        drop = config.DROPOUT if dropout is None else dropout
        widths = [base, base * 2, base * 4, base * 8]

        blocks, in_ch = [], in_channels
        for out_ch in widths:                    # 63 -> 31 -> 15 -> 7 -> 3
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 2, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = StressCNN()
    dummy = torch.zeros(2, 1, config.GRID_N, config.GRID_N, config.GRID_N)
    print(net)
    print("output shape:", tuple(net(dummy).shape))
    print("trainable parameters:", count_parameters(net))
