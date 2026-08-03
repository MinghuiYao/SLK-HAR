"""Small, device-safe dynamic sparsity utilities for SLK-HAR."""

import copy
import math
from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def SNIP(
    network: nn.Module,
    keep_ratio: float,
    train_dataloader: Iterable,
    device: torch.device,
    masks: Optional[dict[str, Tensor]] = None,
) -> list[float]:
    """Estimate layer sparsities from one minibatch of connection sensitivity."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must satisfy 0 < keep_ratio <= 1.")
    inputs, targets = next(iter(train_dataloader))
    model = copy.deepcopy(network).to(device)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs.to(device)), targets.to(device))
    loss.backward()

    scores = []
    names = []
    for name, weight in model.named_parameters():
        if weight.ndim < 2 or weight.grad is None:
            continue
        if masks is not None and name not in masks:
            continue
        scores.append((weight * weight.grad).abs())
        names.append(name)
    if not scores:
        raise ValueError("No prunable parameters with gradients were found.")
    all_scores = torch.cat([score.flatten() for score in scores])
    keep = max(1, round(all_scores.numel() * keep_ratio))
    threshold = torch.topk(all_scores, keep, sorted=True).values[-1]
    return [float((score < threshold).sum().item() / score.numel()) for score in scores]


class CosineDecay:
    """Cosine schedule compatible with the original pruning-rate interface."""

    def __init__(
        self,
        prune_rate: float,
        max_steps: int,
        minimum: float = 0.005,
        last_epoch: int = -1,
        init_step: int = 0,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        self._parameter = nn.Parameter(torch.zeros(1))
        self._optimizer = torch.optim.SGD([self._parameter], lr=prune_rate)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=max_steps,
            eta_min=minimum,
            last_epoch=last_epoch,
        )
        for _ in range(init_step):
            self.step()

    def step(self) -> None:
        self._scheduler.step()

    def get_dr(self, prune_rate: Optional[float] = None) -> float:
        del prune_rate
        return float(self._optimizer.param_groups[0]["lr"])


class Masking:
    """Maintain sparse masks with magnitude pruning and device-safe regrowth.

    This class keeps the original public name while providing an explicit,
    tested core. Parameters with fewer than two dimensions (biases and batch
    normalization vectors) remain dense.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        train_loader=None,
        prune_rate_decay: Optional[CosineDecay] = None,
        prune_rate: float = 0.01,
        prune_mode: str = "magnitude",
        growth_mode: str = "random",
        redistribution_mode: str = "none",
        verbose: bool = False,
        fp16: bool = False,
        density: float = 0.6,
        update_frequency: int = 30,
    ) -> None:
        if not 0 <= prune_rate < 1 or not 0 < density <= 1:
            raise ValueError("prune_rate and density must be valid probabilities.")
        if prune_mode != "magnitude":
            raise ValueError("The maintained implementation supports prune_mode='magnitude'.")
        if growth_mode not in {"random", "gradient", "momentum"}:
            raise ValueError("growth_mode must be random, gradient, or momentum.")
        if update_frequency <= 0:
            raise ValueError("update_frequency must be positive.")
        del redistribution_mode, fp16
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.prune_rate_decay = prune_rate_decay
        self.prune_rate = prune_rate
        self.prune_mode = prune_mode
        self.growth_mode = growth_mode
        self.verbose = verbose
        self.density = density
        self.update_frequency = update_frequency
        self.steps = 0
        self.modules: list[nn.Module] = []
        self.masks: dict[str, Tensor] = {}
        self.fired_masks: dict[str, Tensor] = {}
        self._parameters: dict[str, nn.Parameter] = {}

    def add_module(self, module: nn.Module, density: Optional[float] = None) -> None:
        if self.modules:
            raise ValueError("One Masking instance currently supports one module.")
        self.modules.append(module)
        target_density = self.density if density is None else density
        if not 0 < target_density <= 1:
            raise ValueError("density must satisfy 0 < density <= 1.")
        for name, parameter in module.named_parameters():
            if parameter.ndim < 2:
                continue
            mask = (torch.rand_like(parameter, dtype=torch.float32) < target_density).to(
                parameter.device
            )
            self.masks[name] = mask
            self.fired_masks[name] = mask.clone()
            self._parameters[name] = parameter
        self.apply_mask()

    @torch.no_grad()
    def apply_mask(self) -> None:
        for name, parameter in self._parameters.items():
            mask = self.masks[name].to(dtype=parameter.dtype, device=parameter.device)
            parameter.mul_(mask)
            state = self.optimizer.state.get(parameter, {})
            for value in state.values():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.mul_(mask)

    def step(self) -> None:
        self.optimizer.step()
        self.apply_mask()
        if self.prune_rate_decay is not None:
            self.prune_rate_decay.step()
            self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps += 1
        if self.steps % self.update_frequency == 0:
            self.truncate_weights()

    @torch.no_grad()
    def truncate_weights(self) -> None:
        for name, parameter in self._parameters.items():
            mask = self.masks[name].bool()
            active = mask.flatten().nonzero(as_tuple=False).flatten()
            remove = min(active.numel(), math.ceil(active.numel() * self.prune_rate))
            if remove == 0:
                continue
            active_magnitudes = parameter.detach().abs().flatten()[active]
            prune_indices = active[torch.topk(active_magnitudes, remove, largest=False).indices]
            updated = mask.flatten().clone()
            updated[prune_indices] = False

            candidates = (~updated).nonzero(as_tuple=False).flatten()
            grow = min(remove, candidates.numel())
            scores = self._growth_scores(parameter).flatten()[candidates]
            grow_indices = candidates[torch.topk(scores, grow, largest=True).indices]
            updated[grow_indices] = True
            self.masks[name] = updated.reshape_as(mask).to(dtype=torch.float32)
            self.fired_masks[name] = self.fired_masks[name].bool() | self.masks[name].bool()
        self.apply_mask()

    def _growth_scores(self, parameter: nn.Parameter) -> Tensor:
        if self.growth_mode == "gradient" and parameter.grad is not None:
            return parameter.grad.detach().abs()
        if self.growth_mode == "momentum":
            state = self.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "momentum_buffer"):
                if key in state:
                    return state[key].detach().abs()
        return torch.rand_like(parameter, dtype=torch.float32)

    def state_dict(self) -> dict[str, Tensor]:
        return {name: mask.clone() for name, mask in self.masks.items()}

    def load_state_dict(self, masks: dict[str, Tensor]) -> None:
        if set(masks) != set(self.masks):
            raise ValueError("Mask names do not match the registered module.")
        self.masks = {
            name: mask.to(device=self._parameters[name].device, dtype=torch.float32)
            for name, mask in masks.items()
        }
        self.apply_mask()

    def fired_masks_update(self) -> tuple[dict[str, float], float]:
        densities = {
            name: float(mask.float().mean().item()) for name, mask in self.fired_masks.items()
        }
        total = sum(mask.numel() for mask in self.fired_masks.values())
        fired = sum(mask.sum().item() for mask in self.fired_masks.values())
        return densities, float(fired / total) if total else 0.0


def magnitude_prune(masking, mask, weight, name):
    remove = math.ceil(masking.prune_rate * mask.sum().item())
    if remove <= 0:
        return mask
    active = mask.flatten().bool().nonzero(as_tuple=False).flatten()
    selected = active[
        torch.topk(
            weight.detach().abs().flatten()[active], min(remove, active.numel()), largest=False
        ).indices
    ]
    result = mask.flatten().clone()
    result[selected] = 0
    return result.reshape_as(mask)


def random_growth(masking, name, new_mask, total_regrowth, weight):
    del masking, name, weight
    candidates = (~new_mask.bool()).flatten().nonzero(as_tuple=False).flatten()
    count = min(total_regrowth, candidates.numel())
    if count:
        chosen = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:count]]
        new_mask.flatten()[chosen] = 1
    return new_mask


def gradient_growth(masking, name, new_mask, total_regrowth, weight):
    del masking, name
    candidates = (~new_mask.bool()).flatten().nonzero(as_tuple=False).flatten()
    if weight.grad is None:
        return random_growth(None, "", new_mask, total_regrowth, weight)
    count = min(total_regrowth, candidates.numel())
    chosen = candidates[torch.topk(weight.grad.detach().abs().flatten()[candidates], count).indices]
    new_mask.flatten()[chosen] = 1
    return new_mask


def momentum_growth(masking, name, new_mask, total_regrowth, weight):
    del name
    state = masking.optimizer.state.get(weight, {})
    scores = state.get("exp_avg", state.get("momentum_buffer"))
    if scores is None:
        return random_growth(masking, "", new_mask, total_regrowth, weight)
    candidates = (~new_mask.bool()).flatten().nonzero(as_tuple=False).flatten()
    count = min(total_regrowth, candidates.numel())
    chosen = candidates[torch.topk(scores.detach().abs().flatten()[candidates], count).indices]
    new_mask.flatten()[chosen] = 1
    return new_mask


prune_funcs = {"magnitude": magnitude_prune}
growth_funcs = {
    "random": random_growth,
    "gradient": gradient_growth,
    "momentum": momentum_growth,
}
redistribution_funcs = {"none": lambda *args, **kwargs: 0.0}

__all__ = [
    "CosineDecay",
    "Masking",
    "SNIP",
    "gradient_growth",
    "growth_funcs",
    "magnitude_prune",
    "momentum_growth",
    "prune_funcs",
    "random_growth",
    "redistribution_funcs",
]
