"""误差声明：可经回调注入的三类误差的纯数据规格（无副作用，不持系统引用）。

每个误差分量字段（``Spec``）接受四种写法：

* ``None``            —— 缺省，恒零（仅 :class:`Pose` 允许）；
* 数                   —— 全种群同值常量；
* ``(P,)`` 张量        —— 逐个体给定（可带 ``requires_grad``，供灵敏度反传）；
* 分布映射             —— ``{method: raw|normal|uniform, ...}``，构建回调时
  按种群采样一次（重采 = 重新 :func:`~perturb.inject.build_callback`；
  全局种子经 ``torch.manual_seed`` 控制）。

面编号约定：1 起的光学面序号（``Refractor`` / ``Stop`` / ``Sensor`` 计入，
``Gap`` 不计），与可视化布局标签 S1/S2/… 一致。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import torch

from core import parse_param, term

type Spec = float | int | torch.Tensor | Mapping[str, Any] | None


def _materialize(
    spec: Spec, population: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """误差分量 → ``(P,)`` 张量：None=0，数=常量，映射=分布采样一次，张量=原样。"""
    if spec is None:
        return torch.zeros(population, device=device, dtype=dtype)
    if isinstance(spec, torch.Tensor):
        t = spec.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.expand(population)
        if t.ndim != 1 or t.shape[0] != population:
            raise ValueError(
                f"error spec tensor must be scalar or ({population},), got shape={tuple(t.shape)}"
            )
        return t
    if isinstance(spec, (int, float)):
        return torch.full((population,), float(spec), device=device, dtype=dtype)
    if isinstance(spec, Mapping):
        # 误差分布缺省零均值（normal 的 mean 可省，其余 method 忽略此键）
        return parse_param({"spec": {"mean": 0.0, **spec}}, "spec", population).to(
            device=device, dtype=dtype
        )
    raise TypeError(f"unsupported error spec: {type(spec)}")


@dataclass(frozen=True)
class Pose:
    """位姿误差：在光学面区间 ``[first, last]`` 上叠加局部扰动 Δ = T(dx,dy,dz)·Rx(α)·Ry(β)。

    覆盖装配误差中的全部位置/姿态类：

    * ``last=None``   —— 不恢复，下游全体继承：间隔误差（``dz``）、
      整组偏心/倾斜、传感器离焦/倾斜/偏心；
    * ``last=first``  —— 单面误差：面倾斜、面偏心；只给透镜后表面即楔角；
    * 跨透镜两面       —— 元件级刚体位移（区间内的 Gap 沿扰动轴前进，语义正确）。

    平移在注入点的名义局部系内度量；倾斜转轴过（位移后的）面顶点；
    长度单位 mm，角度单位 rad。小角度下因子次序无关。

    五个分量全 ``None``（恒等扰动）会在 :func:`~perturb.inject.build_callback`
    时报错——空声明基本是配置笔误，不允许静默通过。
    """

    first: int
    last: int | None = None
    dx: Spec = None
    dy: Spec = None
    dz: Spec = None
    alpha: Spec = None  # 绕局部 x 轴
    beta: Spec = None  # 绕局部 y 轴

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        """从配置映射构造（``first`` 必填，其余缺省为恒零）。"""
        last = term.LAST.resolve(options, default=None)
        return cls(
            first=int(term.FIRST.resolve(options)),
            last=None if last is None else int(last),
            dx=term.DX.resolve(options, default=None),
            dy=term.DY.resolve(options, default=None),
            dz=term.DZ.resolve(options, default=None),
            alpha=term.ALPHA.resolve(options, default=None),
            beta=term.BETA.resolve(options, default=None),
        )


@dataclass(frozen=True)
class Scatter:
    """面散射近似：第 ``surface`` 面折射后，方向叠加各向同性角噪声（``std``，rad）。

    定性模拟高频粗糙度/污染导致的光晕；RayBundle 无振幅通道，
    非能量守恒，不能做定量杂散光分析。只作用于存活光线。
    """

    surface: int
    std: Spec

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        return cls(
            surface=int(term.SURFACE.resolve(options)),
            std=term.STD.resolve(options),
        )


@dataclass(frozen=True)
class Clip:
    """机械遮挡：光线在站点截面上的横向半径超过 ``radius`` 即判死（APERTURE_CLIP）。

    检查在每站的名义 z 平面上做（光线先直线推进到该面），是圆筒/垫圈的
    逐站采样近似——站间穿出筒壁又收回的极端路径抓不到。

    * ``at=None`` —— 每个站点都检查（圆筒形镜筒）；
    * ``at=k``    —— 只查第 k 面所在截面（垫圈/挡光环）。
    """

    radius: Spec
    at: int | None = None

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        at = term.AT.resolve(options, default=None)
        return cls(
            radius=term.RADIUS.resolve(options),
            at=None if at is None else int(at),
        )


_KINDS: tuple[tuple[Any, type], ...] = (
    (term.POSE, Pose),
    (term.SCATTER, Scatter),
    (term.CLIP, Clip),
)


def from_options(options: Mapping[str, Any]) -> Pose | Scatter | Clip:
    """按 ``type`` 键分派误差声明：``pose`` / ``scatter`` / ``clip``。"""
    kind = term.TYPE.resolve(options)
    for noun, cls in _KINDS:
        if kind in noun:
            return cls.from_options(options)
    raise ValueError(
        f"unknown perturb type: {kind!r} (available: {[n.canonical for n, _ in _KINDS]})"
    )
