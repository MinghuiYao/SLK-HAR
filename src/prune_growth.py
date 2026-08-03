"""Installed compatibility exports for ``prune_growth``."""

from slk_har.sparsity import (
    gradient_growth,
    growth_funcs,
    magnitude_prune,
    momentum_growth,
    prune_funcs,
    random_growth,
    redistribution_funcs,
)

__all__ = [
    "gradient_growth",
    "growth_funcs",
    "magnitude_prune",
    "momentum_growth",
    "prune_funcs",
    "random_growth",
    "redistribution_funcs",
]
