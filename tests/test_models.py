import pytest
import torch

from slk_har.models import SLKNet


def test_reference_model_forward_and_backward() -> None:
    model = SLKNet((1, 128, 9), 6, channels=(8, 16, 24), drop_path=0.0)
    inputs = torch.randn(2, 1, 128, 9)
    outputs = model(inputs)
    assert outputs.shape == (2, 6)
    outputs.mean().backward()
    assert model.classifier.weight.grad is not None


def test_model_rejects_modality_mismatch() -> None:
    model = SLKNet((1, 128, 9), 6, channels=(8, 16, 24), drop_path=0.0)
    with pytest.raises(ValueError, match="configured for 9 modalities"):
        model(torch.randn(2, 1, 128, 6))
