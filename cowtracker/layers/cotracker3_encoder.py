# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CoTracker3-style lightweight convolutional video encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block used by the CoTracker3 BasicEncoder."""

    def __init__(self, in_planes: int, planes: int, norm_fn: str = "instance", stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            padding=1,
            stride=stride,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == "group":
            num_groups = max(1, planes // 8)
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if stride != 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if stride != 1:
                self.norm3 = nn.BatchNorm2d(planes)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if stride != 1:
                self.norm3 = nn.InstanceNorm2d(planes)
        elif norm_fn == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            if stride != 1:
                self.norm3 = nn.Identity()
        else:
            raise ValueError("norm_fn must be one of: group, batch, instance, none")

        self.downsample = None
        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                self.norm3,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.relu(self.norm2(self.conv2(y)))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x + y)


class CoTracker3BasicEncoder(nn.Module):
    """Dense video feature encoder adapted from CoTracker3 BasicEncoder.

    Input video is expected in [0, 1]. The encoder applies CoTracker3's
    [-1, 1] normalization internally and returns channel-normalized features.
    """

    def __init__(self, input_dim: int = 3, output_dim: int = 128, stride: int = 4):
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be a positive integer.")
        if output_dim < 4:
            raise ValueError("output_dim must be >= 4.")

        self.stride = int(stride)
        self.output_dim = int(output_dim)
        self.norm_fn = "instance"
        self.in_planes = output_dim // 2
        self.norm1 = nn.InstanceNorm2d(self.in_planes)
        self.norm2 = nn.InstanceNorm2d(output_dim * 2)

        self.conv1 = nn.Conv2d(
            input_dim,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode="zeros",
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(output_dim // 2, stride=1)
        self.layer2 = self._make_layer(output_dim // 4 * 3, stride=2)
        self.layer3 = self._make_layer(output_dim, stride=2)
        self.layer4 = self._make_layer(output_dim, stride=2)

        self.conv2 = nn.Conv2d(
            output_dim * 3 + output_dim // 4,
            output_dim * 2,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.InstanceNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_layer(self, dim: int, stride: int = 1) -> nn.Sequential:
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        self.in_planes = dim
        return nn.Sequential(layer1, layer2)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must have shape [B,S,3,H,W], got {tuple(video.shape)}")

        b, s, c, h, w = video.shape
        if h % self.stride != 0 or w % self.stride != 0:
            raise ValueError(
                f"Input image size {(h, w)} must be divisible by stride={self.stride}."
            )

        x = 2.0 * video.reshape(b * s, c, h, w) - 1.0
        x = self.relu1(self.norm1(self.conv1(x)))

        a = self.layer1(x)
        b_feat = self.layer2(a)
        c_feat = self.layer3(b_feat)
        d_feat = self.layer4(c_feat)

        out_size = (h // self.stride, w // self.stride)
        pyramid = [
            F.interpolate(feat, out_size, mode="bilinear", align_corners=True)
            for feat in (a, b_feat, c_feat, d_feat)
        ]
        x = self.relu2(self.norm2(self.conv2(torch.cat(pyramid, dim=1))))
        x = self.conv3(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        return x.reshape(b, s, self.output_dim, out_size[0], out_size[1])
