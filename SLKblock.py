import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import sys
import os


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, groups=in_channels, bias=bias)

def get_bn(channels):
    return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bn=True):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    if bn:
        result.add_module('bn', get_bn(out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation)
    result.add_module('nonlinear', nn.ReLU())
    return result

def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std

class DecomLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,small_kernel,
                 small_kernel_merged=True, Decom=True , bn=True):
        super(DecomLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.Decom = Decom

        if type(kernel_size) == tuple:
            padding = (kernel_size[0] // 2,kernel_size[1] // 2)
        elif type(kernel_size) == int:
            padding = kernel_size // 2

        self.LoRA1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size ,
                                stride=stride, padding=padding, dilation=1, groups=groups)
        self.LoRA2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=1, groups=groups)
        self.LoRA3 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, dilation=1, groups=groups)
        if small_kernel is not None:
            self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=small_kernel,
                                            stride=stride, padding=small_kernel//2 if type(small_kernel) == int else (small_kernel[0]//2,small_kernel[1]//2), groups=groups, dilation=1)

    def forward(self, inputs):
        out = self.LoRA1(inputs) + self.LoRA2(inputs) + self.LoRA3(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        return out

class SLKBlock(nn.Module):
     def __init__(self,  in_channels, dw_channels, block_lk_size, small_kernel, drop_path=0.1,Decom=True):
        super().__init__()

        self.large_kernel = DecomLargeKernelConv(in_channels=in_channels, out_channels=dw_channels,
                                                   kernel_size=(block_lk_size[0]//3,block_lk_size[1]),
                                                   stride=1, groups=in_channels, small_kernel=small_kernel,
                                                   small_kernel_merged=False, Decom=Decom)

        self.norm = nn.BatchNorm2d(dw_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.short = nn.Sequential()
        if (in_channels != dw_channels):
            self.short = nn.Sequential(
                 nn.Conv2d(in_channels,dw_channels,1),
                 nn.ReLU(),
            )

     def forward(self, x):
        out = self.large_kernel(x)
        out = self.norm(out)
        return self.short(x) + self.drop_path(out)

