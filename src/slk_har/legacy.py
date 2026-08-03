"""Adapters for the public names in the original source-only release."""

import torch
from torch import nn

from .layers import (
    DecomposedLargeKernelConv,
)
from .layers import (
    SLKBlock as MaintainedSLKBlock,
)
from .layers import (
    conv_bn as maintained_conv_bn,
)
from .models import SLKNet


def get_bn(channels):
    return nn.BatchNorm2d(channels)


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=None,
    groups=1,
    dilation=1,
    bn=True,
):
    return maintained_conv_bn(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
        batch_norm=bn,
    )


def conv_bn_relu(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=None,
    groups=1,
    dilation=1,
):
    result = conv_bn(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
        dilation=dilation,
    )
    result.add_module("nonlinear", nn.ReLU(inplace=True))
    return result


def fuse_bn(convolution, batch_norm):
    kernel = convolution.weight
    bias = (
        convolution.bias
        if convolution.bias is not None
        else torch.zeros(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    )
    scale = batch_norm.weight / torch.sqrt(batch_norm.running_var + batch_norm.eps)
    return (
        kernel * scale.reshape(-1, 1, 1, 1),
        batch_norm.bias + (bias - batch_norm.running_mean) * scale,
    )


class DecomLargeKernelConv(DecomposedLargeKernelConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        small_kernel,
        small_kernel_merged=True,
        Decom=True,
        bn=True,
    ):
        del small_kernel_merged
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            small_kernel,
            branches=3,
            decompose_kernel=False,
            batch_norm=bn,
        )
        self.Decom = Decom


class SLKBlock(MaintainedSLKBlock):
    def __init__(
        self,
        in_channels,
        dw_channels,
        block_lk_size,
        small_kernel,
        drop_path=0.1,
        Decom=True,
    ):
        super().__init__(
            in_channels,
            dw_channels,
            block_lk_size,
            small_kernel,
            drop_path=drop_path,
        )
        self.Decom = Decom


class SLK_Net(SLKNet):
    def __init__(self, train_shape, category):
        super().__init__(train_shape, category)

    @property
    def layer(self):
        return self.features

    @property
    def ada_pool(self):
        return self.pool

    @property
    def fc(self):
        return self.classifier


__all__ = [
    "DecomLargeKernelConv",
    "SLKBlock",
    "SLK_Net",
    "conv_bn",
    "conv_bn_relu",
    "fuse_bn",
    "get_bn",
]
