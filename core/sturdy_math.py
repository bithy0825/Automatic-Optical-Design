from typing import Final

import torch

_EPS: Final[float] = 1e-12


def sturdy_sqrt(x: torch.Tensor, eps=_EPS) -> torch.Tensor:
    mask = x.ge(0.0)
    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    return torch.where(mask, safe.sqrt(), 0.0)


def sturdy_inv(x: torch.Tensor) -> torch.Tensor:
    mask = x.ne(0.0).logical_and(x.isfinite())
    safe = torch.where(mask, x, 1.0)
    return torch.where(mask, safe.reciprocal(), 0.0)


def sturdy_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    mask = denominator.ne(0.0).logical_and(denominator.isfinite())
    safe_denominator = torch.where(mask, denominator, 1.0)
    return torch.where(mask, numerator.div(safe_denominator), 0.0)
