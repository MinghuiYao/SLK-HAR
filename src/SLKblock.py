"""Installed compatibility exports for ``SLKblock``."""

from slk_har.layers import DecomposedLargeKernelConv, StochasticDepth, get_conv2d
from slk_har.legacy import (
    DecomLargeKernelConv,
    SLKBlock,
    conv_bn,
    conv_bn_relu,
    fuse_bn,
    get_bn,
)

__all__ = [
    "DecomLargeKernelConv",
    "DecomposedLargeKernelConv",
    "SLKBlock",
    "StochasticDepth",
    "conv_bn",
    "conv_bn_relu",
    "fuse_bn",
    "get_bn",
    "get_conv2d",
]
