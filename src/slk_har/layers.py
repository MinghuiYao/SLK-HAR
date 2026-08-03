"""Sparse large-kernel building blocks."""

from typing import Union

import torch
from torch import Tensor, nn

Pair = tuple[int, int]


def _pair(value: Union[int, Pair]) -> Pair:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("Expected an integer or a pair.")
    return int(value[0]), int(value[1])


def get_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Pair],
    stride: Union[int, Pair],
    padding: Union[int, Pair],
    dilation: Union[int, Pair],
    groups: int,
    bias: bool,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Pair],
    stride: Union[int, Pair] = 1,
    padding: Union[int, Pair, None] = None,
    groups: int = 1,
    dilation: Union[int, Pair] = 1,
    batch_norm: bool = True,
) -> nn.Sequential:
    kernel_size = _pair(kernel_size)
    if padding is None:
        padding = kernel_size[0] // 2, kernel_size[1] // 2
    layers = [
        get_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            False,
        )
    ]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv_bn_relu(*args, **kwargs) -> nn.Sequential:
    result = conv_bn(*args, **kwargs)
    result.add_module("activation", nn.ReLU(inplace=True))
    return result


class StochasticDepth(nn.Module):
    """Per-sample stochastic depth without an external timm dependency."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("probability must satisfy 0 <= p < 1.")
        self.probability = probability

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep_probability = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep_probability)
        return inputs * mask / keep_probability


class DecomposedLargeKernelConv(nn.Module):
    """Parallel grouped branches representing a large-kernel parameter budget."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Pair],
        stride: Union[int, Pair],
        groups: int,
        small_kernel: Union[int, Pair, None],
        branches: int = 3,
        decompose_kernel: bool = True,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if branches <= 0:
            raise ValueError("branches must be positive.")
        target_kernel = _pair(kernel_size)
        branch_kernel = (
            (max(1, target_kernel[0] // branches), target_kernel[1])
            if decompose_kernel
            else target_kernel
        )
        if branch_kernel[0] % 2 == 0:
            branch_kernel = (branch_kernel[0] + 1, branch_kernel[1])
        self.target_kernel = target_kernel
        self.branch_kernel = branch_kernel
        self.branches = nn.ModuleList(
            conv_bn(
                in_channels,
                out_channels,
                branch_kernel,
                stride=stride,
                groups=groups,
                batch_norm=batch_norm,
            )
            for _ in range(branches)
        )
        if small_kernel is not None:
            self.small_branch = conv_bn(
                in_channels,
                out_channels,
                small_kernel,
                stride=stride,
                groups=groups,
                batch_norm=batch_norm,
            )

    def forward(self, inputs: Tensor) -> Tensor:
        output = torch.stack([branch(inputs) for branch in self.branches]).sum(dim=0)
        if hasattr(self, "small_branch"):
            output = output + self.small_branch(inputs)
        return output


class SLKBlock(nn.Module):
    """Diverse-branch grouped large-kernel block with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        large_kernel: Union[int, Pair],
        small_kernel: Union[int, Pair],
        drop_path: float = 0.1,
        branches: int = 3,
    ) -> None:
        super().__init__()
        self.channel_projection = (
            nn.Identity()
            if in_channels == out_channels
            else conv_bn_relu(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.large_kernel = DecomposedLargeKernelConv(
            out_channels,
            out_channels,
            kernel_size=large_kernel,
            stride=1,
            groups=out_channels,
            small_kernel=small_kernel,
            branches=branches,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.channel_projection(inputs)
        output = self.norm(self.large_kernel(residual))
        return residual + self.drop_path(output)


DecomLargeKernelConv = DecomposedLargeKernelConv

__all__ = [
    "DecomLargeKernelConv",
    "DecomposedLargeKernelConv",
    "SLKBlock",
    "StochasticDepth",
    "conv_bn",
    "conv_bn_relu",
    "get_conv2d",
]
