import torch

from slk_har.layers import DecomposedLargeKernelConv, SLKBlock
from SLKblock import DecomLargeKernelConv


def test_diverse_branches_are_independent_and_differentiable() -> None:
    layer = DecomposedLargeKernelConv(
        4,
        8,
        kernel_size=(57, 1),
        stride=1,
        groups=4,
        small_kernel=(5, 1),
    )
    assert len(layer.branches) == 3
    assert len({id(branch[0].weight) for branch in layer.branches}) == 3

    inputs = torch.randn(2, 4, 32, 7, requires_grad=True)
    outputs = layer(inputs)
    assert outputs.shape == (2, 8, 32, 7)
    outputs.mean().backward()
    assert inputs.grad is not None


def test_slk_block_residual_shape() -> None:
    block = SLKBlock(4, 8, (57, 1), (5, 1), drop_path=0.0)
    assert block(torch.randn(2, 4, 32, 7)).shape == (2, 8, 32, 7)


def test_original_decomposed_layer_signature() -> None:
    layer = DecomLargeKernelConv(
        4,
        8,
        kernel_size=(19, 1),
        stride=1,
        groups=4,
        small_kernel=(5, 1),
        small_kernel_merged=False,
        Decom=True,
        bn=False,
    )
    assert layer.branch_kernel == (19, 1)
    assert layer(torch.randn(2, 4, 32, 7)).shape == (2, 8, 32, 7)
