"""Sparse diverse-branch large-kernel models for wearable HAR."""

from .layers import DecomposedLargeKernelConv, SLKBlock
from .models import SLKNet
from .sparsity import CosineDecay, Masking

__all__ = ["CosineDecay", "DecomposedLargeKernelConv", "Masking", "SLKBlock", "SLKNet"]
__version__ = "0.2.0"
