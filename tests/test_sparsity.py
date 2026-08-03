import torch
from torch import nn

from slk_har.sparsity import Masking


def test_masking_preserves_nonzero_budget_and_device() -> None:
    model = nn.Sequential(nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    masking = Masking(
        optimizer,
        prune_rate=0.2,
        density=0.5,
        update_frequency=1,
        growth_mode="gradient",
    )
    masking.add_module(model)
    counts_before = {name: int(mask.sum().item()) for name, mask in masking.masks.items()}

    loss = model(torch.randn(6, 12)).square().mean()
    loss.backward()
    masking.step()

    counts_after = {name: int(mask.sum().item()) for name, mask in masking.masks.items()}
    assert counts_after == counts_before
    for name, parameter in model.named_parameters():
        if name in masking.masks:
            assert torch.all(parameter.detach()[masking.masks[name] == 0] == 0)
