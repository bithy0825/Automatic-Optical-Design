from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import torch

from core import Noun, SystemFloatScalar, TraceFlow, Verdict, sturdy_div, term
from component import Sequential
from optimization.target import Target

_CAUSE_NOUNS: tuple[tuple[Noun, Verdict.Cause], ...] = (
    (term.SAG_DOMAIN, Verdict.Cause.SAG_DOMAIN),
    (term.SOLVER_NEGATIVE, Verdict.Cause.SOLVER_NEGATIVE),
    (term.SOLVER_CONVERGENCE, Verdict.Cause.SOLVER_CONVERGENCE),
    (term.APERTURE_CLIP, Verdict.Cause.APERTURE_CLIP),
    (term.TIR, Verdict.Cause.TIR),
)

_PARAM_NOUNS: tuple[Noun, ...] = (
    term.DIAMETER,
    term.CURVATURE,
    term.KAPPA,
    term.ALPHA,
    term.THICKNESS,
)


def _cause_of(key: str) -> Verdict.Cause | None:
    """损失权重键 → 裁决死因（无匹配返回 None，交由参数名词解析）。"""
    for noun, cause in _CAUSE_NOUNS:
        if noun.match(key):
            return cause
    return None


def _noun_of(key: str) -> Noun:
    """边界配置键 → 参数名词（静态词表匹配，未知键 fail-fast）。"""
    for noun in _PARAM_NOUNS:
        if noun.match(key):
            return noun
    raise KeyError(f"unknown bounds key: {key!r}")


@dataclass(frozen=True, slots=True)
class LossWeights:
    """各损失项权重。未列出的 cause / 边界参数权重缺省为 1.0。"""

    effl: float = 1.0
    spot: float = 1.0
    toll: dict[Verdict.Cause, float] = field(default_factory=dict)
    bounds: dict[Noun, float] = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: Mapping[str, float]) -> Self:
        """从配置构造（一次性解析，``loss`` 键下的扁平映射）。

        键：``effl`` / ``spot`` / 各死因名词（``sag_domain``、
        ``solver_negative``、``solver_convergence``、``aperture_clip``、
        ``tir``）/ 各边界参数名词（``diameter``、``curvature``、``kappa``、
        ``alpha``、``thickness``）。
        """
        options = term.LOSS.resolve(options, default={})
        toll: dict[Verdict.Cause, float] = {}
        bounds: dict[Noun, float] = {}
        for key, value in options.items():
            if term.EFFL.match(key) or term.SPOT.match(key):
                continue
            cause = _cause_of(key)
            if cause is not None:
                toll[cause] = float(value)
            else:
                bounds[_noun_of(key)] = float(value)
        return cls(
            effl=term.EFFL.resolve(options, default=1.0),
            spot=term.SPOT.resolve(options, default=1.0),
            toll=toll,
            bounds=bounds,
        )


def effl_loss(
    flow: TraceFlow, target: Target, weights: LossWeights
) -> SystemFloatScalar:
    """加权相对 EFFL 误差：``w · ((f_est − f*) / f*)²``。"""
    t = flow.rays.field.tan()  # (P,F,W,N,2) tanθ
    h = flow.rays.points[..., :2]  # 传感器落点 (x, y)
    w = flow.verdict.hold.unsqueeze(-1)  # 存活权重
    f_est = sturdy_div(
        w.mul(h).mul(t).sum(dim=(1, 2, 3, 4)),
        w.mul(t.square()).sum(dim=(1, 2, 3, 4)),
    )
    return f_est.sub(target.effl).div(target.effl).square().mul(weights.effl)


def spot_loss(
    flow: TraceFlow, target: Target, weights: LossWeights
) -> SystemFloatScalar:
    """加权点列：存活光线相对理想像点的均方偏差 (mm²)。"""
    t = flow.rays.field.tan()
    h = flow.rays.points[..., :2]
    r2 = h.sub(t.mul(target.effl)).square().sum(dim=-1)  # 每光线 |残差|²
    w = flow.verdict.hold
    return sturdy_div(w.mul(r2).sum(dim=(1, 2, 3)), w.sum(dim=(1, 2, 3))).mul(
        weights.spot
    )


def toll_loss(flow: TraceFlow, weights: LossWeights) -> SystemFloatScalar:
    """加权死亡惩罚：各 cause 的 toll 按光线总数平均后按权重求和。"""
    v = flow.verdict
    total = torch.zeros(v.toll.shape[0], device=v.device, dtype=v.toll.dtype)
    for _noun, cause in _CAUSE_NOUNS:
        weight = weights.toll.get(cause, 1.0)
        if weight != 0.0:
            total = total.add(
                v.toll.mul(v.cause.eq(cause)).mean(dim=(1, 2, 3)).mul(weight)
            )
    return total


def _hinge(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return x.clamp(lo, hi).sub(x).square()


def bounds_loss(
    seq: Sequential, blocks: Sequence[Mapping[str, Any]], weights: LossWeights
) -> SystemFloatScalar:
    """加权参数边界：逐元件读各自 ``bounds``，铰链违约量按参数权重求和。"""
    total = torch.zeros(seq.population, device=seq.device, dtype=seq.dtype)
    for comp, block in zip(seq, blocks, strict=True):
        bounds = term.BOUNDS.resolve(block, default=None)
        if bounds is None:
            continue
        for key, (lo, hi) in bounds.items():
            noun = _noun_of(key)
            weight = weights.bounds.get(noun, 1.0)
            if weight == 0.0:
                continue
            v = _hinge(comp[noun], lo, hi)
            total = total.add((v.sum(dim=-1) if v.ndim > 1 else v).mul(weight))
    return total


def total_loss(
    flow: TraceFlow,
    seq: Sequential,
    target: Target,
    blocks: Sequence[Mapping[str, Any]],
    weights: LossWeights | None = None,
) -> tuple[SystemFloatScalar, dict[str, SystemFloatScalar]]:
    """总损失与四项加权分项（分项用于日志与 GA 择优）。"""
    weights = weights or LossWeights()
    parts = {
        "effl": effl_loss(flow, target, weights),
        "spot": spot_loss(flow, target, weights),
        "toll": toll_loss(flow, weights),
        "bounds": bounds_loss(seq, blocks, weights),
    }
    total = torch.zeros_like(parts["spot"])
    for p in parts.values():
        total = total.add(p)
    return total, parts
