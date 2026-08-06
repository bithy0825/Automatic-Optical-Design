from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import torch

from core import Noun, SystemFloatScalar, TraceFlow, Verdict, sturdy_div, term
from component import Sensor, Sequential
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
    blur: float = 1.0
    distortion: float = 1.0
    survival: float = 0.0
    toll: dict[Verdict.Cause, float] = field(default_factory=dict)
    bounds: dict[Noun, float] = field(default_factory=dict)

    @classmethod
    def from_options(cls, options: Mapping[str, float]) -> Self:
        """从配置构造（一次性解析，``loss`` 键下的扁平映射）。

        键：``effl`` / ``blur`` / ``distortion`` / ``survival`` / 各死因名词
        （``sag_domain``、``solver_negative``、``solver_convergence``、
        ``aperture_clip``、``tir``）/ 各边界参数名词（``diameter``、
        ``curvature``、``kappa``、``alpha``、``thickness``）。
        """
        options = term.LOSS.resolve(options, default={})
        toll: dict[Verdict.Cause, float] = {}
        bounds: dict[Noun, float] = {}
        scalars = (term.EFFL, term.BLUR, term.DISTORTION, term.SURVIVAL)
        for key, value in options.items():
            if any(noun.match(key) for noun in scalars):
                continue
            cause = _cause_of(key)
            if cause is not None:
                toll[cause] = float(value)
            else:
                bounds[_noun_of(key)] = float(value)
        return cls(
            effl=term.EFFL.resolve(options, default=1.0),
            blur=term.BLUR.resolve(options, default=1.0),
            distortion=term.DISTORTION.resolve(options, default=1.0),
            survival=term.SURVIVAL.resolve(options, default=0.0),
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


def _chief(flow: TraceFlow) -> torch.Tensor:
    """逐视场主光线：存活光线在传感器上的加权质心，``(P, F, 1, 1, 2)``。

    跨波长、跨光瞳聚合（横向色差因此留在 blur 内）；不 detach——
    梯度经质心回传（标准 RMS-about-centroid 语义）。
    """
    h = flow.rays.points[..., :2]
    w = flow.verdict.hold
    den = w.sum(dim=(2, 3), keepdim=True).unsqueeze(-1)
    return sturdy_div(w.unsqueeze(-1).mul(h).sum(dim=(2, 3), keepdim=True), den)


def blur_loss(flow: TraceFlow, seq: Sequential, weights: LossWeights) -> SystemFloatScalar:
    """模糊损失：相对本视场主光线的均方偏差 (mm²)。

    幸存光线按真实 r²（绕本视场主光线）；死亡光线每条按 **sensor 半径平方**
    （取自链上末端 Sensor 的直径）计入——死亡不再免费：全死个体
    blur = R_sensor² 直接垫底，1% 死亡约 +R²/100。链上无 Sensor 时退回
    旧语义（仅幸存者加权平均）。
    """
    h = flow.rays.points[..., :2]
    w = flow.verdict.hold
    r2 = h.sub(_chief(flow)).square().sum(dim=-1)
    dead_r2 = _dead_r2(seq)
    if dead_r2 is None:
        return sturdy_div(w.mul(r2).sum(dim=(1, 2, 3)), w.sum(dim=(1, 2, 3))).mul(
            weights.blur
        )
    r2 = torch.where(w, r2, dead_r2.view(-1, 1, 1, 1))
    return r2.mean(dim=(1, 2, 3)).mul(weights.blur)


def _dead_r2(seq: Sequential) -> SystemFloatScalar | None:
    """末端 Sensor 的半径平方（死光线的缺失误差）；链上无 Sensor 返回 None。"""
    for comp in reversed(list(seq)):
        if isinstance(comp, Sensor):
            return comp.shape.Diameter.mul(0.5).square()
    return None


def distortion_loss(
    flow: TraceFlow, target: Target, weights: LossWeights
) -> SystemFloatScalar:
    """加权畸变：逐视场主光线相对理想像点的均方偏差 (mm²)——焦距 / 畸变。

    与 :func:`blur_loss` 互补：两者等权之和精确等于"相对理想像点的点列"
    （勾股分解，交叉项为零）。
    """
    w = flow.verdict.hold
    ideal = flow.rays.field.tan().mul(target.effl)
    d2 = _chief(flow).sub(ideal).square().sum(dim=-1)
    return sturdy_div(w.mul(d2).sum(dim=(1, 2, 3)), w.sum(dim=(1, 2, 3))).mul(
        weights.distortion
    )


def survival_loss(flow: TraceFlow, weights: LossWeights) -> SystemFloatScalar:
    """加权死亡计数：每死一条光线固定代价，与死亡深度无关。

    不可微——仅在 GA 排序与 SA 接受中施加选择压力；梯度阶段由 toll 负责。
    """
    dead = flow.verdict.hold.logical_not()
    return dead.float().mean(dim=(1, 2, 3)).mul(weights.survival)


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
    """总损失与六项加权分项（分项用于日志与 GA 择优）。"""
    weights = weights or LossWeights()
    parts = {
        "effl": effl_loss(flow, target, weights),
        "blur": blur_loss(flow, seq, weights),
        "distortion": distortion_loss(flow, target, weights),
        "survival": survival_loss(flow, weights),
        "toll": toll_loss(flow, weights),
        "bounds": bounds_loss(seq, blocks, weights),
    }
    # 死光线的发散几何（NaN 命中点 / 天文数字距离）会以 0×NaN、0×inf 形式
    # 污染加权求和——逐项归为 0：该个体的该项视为无信号，排序由其余分项决定。
    parts = {k: torch.nan_to_num(v, nan=0.0) for k, v in parts.items()}
    total = torch.zeros_like(parts["blur"])
    for p in parts.values():
        total = total.add(p)
    return total, parts
