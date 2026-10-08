"""数值稳健原语：NaN / ±inf 安全的平方根、倒数与除法。

三件套共同约定：非法输入（负 radicand、零或非有限分母）的输出为零且
梯度为零——死亡光线携带的坏值不以 NaN 形式污染损失，而是静默归零，
生死与代价的记账由裁决体系（:class:`~core.flow.Verdict`）负责。
"""

from typing import Final

import torch

_EPS: Final[float] = 1e-12


def sturdy_sqrt(x: torch.Tensor, eps: float = _EPS) -> torch.Tensor:
    """安全平方根：负值输出 0（零梯度）；其余先把非有限值归零、再 clamp 到
    *eps* 下界后开方，保证原点处梯度有限。"""
    mask = x.ge(0.0)
    safe = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    return torch.where(mask, safe.sqrt(), 0.0)


def sturdy_inv(x: torch.Tensor) -> torch.Tensor:
    """安全倒数：零或非有限输入输出 0（零梯度）。"""
    mask = x.ne(0.0).logical_and(x.isfinite())
    safe = torch.where(mask, x, 1.0)
    return torch.where(mask, safe.reciprocal(), 0.0)


def sturdy_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """安全除法：分母为零或非有限时输出 0（零梯度）。"""
    mask = denominator.ne(0.0).logical_and(denominator.isfinite())
    safe_denominator = torch.where(mask, denominator, 1.0)
    return torch.where(mask, numerator.div(safe_denominator), 0.0)
