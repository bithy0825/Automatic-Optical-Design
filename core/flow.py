"""追迹数据流：在元件间流动的三类数据。

* :class:`RayBundle` —— 一束光线：起点、方向与逐光线标签（光瞳坐标 / 视场 / 波长）。
* :class:`Verdict`   —— 追迹裁决：一根光线在某站点（面形域 / 求解器 / 孔径 / TIR）的生死与代价。
* :class:`TraceFlow` —— 追迹状态：光线束 + 当前位姿 + 累计裁决。

裁决契约
--------
* ``hold``  —— 存活标志。``False`` 即死亡，死亡即冻结，绝不复活。
* ``toll``  —— 非负"代价"。构造站点（:meth:`Verdict.site`）时可传有符号量：
  正 = 存活余量、负 = 死亡深度；经 :meth:`Verdict.at` 链入后一律归一为非负，
  供死亡损失反向传播。
* ``cause`` —— 死因（:class:`Verdict.Cause`），首次判死的站点钉死，后续站点不覆盖。
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Self

import torch

from core.aliases import (
    RayBoolScalar,
    RayFloat2D,
    RayFloat3D,
    RayFloatScalar,
    RayLongScalar,
)
from core.container import TensorContainer
from core.transformer import Transformer


@dataclass(slots=True, eq=False, repr=False)
class Verdict(TensorContainer):
    """追迹裁决（契约见模块 docstring）。"""

    class Cause(IntEnum):
        NONE = 0  # 无异常
        SAG_DOMAIN = 1  # 矢高定义域越界（radicand < 0）
        SOLVER_NEGATIVE = 2  # 求解器负距离（中间面 X 型打架）
        SOLVER_CONVERGENCE = 3  # 求解器不收敛（|f| > tol）
        APERTURE_CLIP = 4  # 机械孔径裁剪（r² > R²）
        TIR = 5  # 全内反射（η² sin²θᵢ > 1）

    hold: RayBoolScalar
    toll: RayFloatScalar
    cause: RayLongScalar

    @classmethod
    def alive_like(cls, ref: RayFloatScalar) -> Self:
        """全活裁决：``hold`` 全真、``toll`` 全零。"""
        return cls(
            hold=torch.ones_like(ref, dtype=torch.bool),
            toll=torch.zeros_like(ref),
            cause=torch.zeros_like(ref, dtype=torch.long),
        )

    @classmethod
    def site(cls, hold: RayBoolScalar, toll: RayFloatScalar, cause: Cause | int) -> Self:
        """构造一个站点的裁决。*toll* 可传有符号量（正 = 余量，负 = 死亡深度）。"""
        return cls(
            hold=hold,
            toll=toll,
            cause=torch.full_like(toll, int(cause), dtype=torch.long),
        )

    def at(self, other: Self) -> Self:
        """链入下一站点裁决：存活取交集；首次判死的站点钉死 toll 与 cause。"""
        hold = self.hold.logical_and(other.hold)
        need_update = self.hold.logical_and(other.hold.logical_not())
        toll = torch.where(need_update, other.toll, self.toll).abs()
        cause = torch.where(need_update, other.cause, self.cause)
        return type(self)(hold=hold, toll=toll, cause=cause)

    @property
    def device(self) -> torch.device:
        return self.hold.device

    @property
    def dtype(self) -> torch.dtype:
        return self.toll.dtype


@dataclass(slots=True, eq=False, repr=False)
class RayBundle(TensorContainer):
    """一束光线：起点、方向与逐光线标签（光瞳坐标 / 视场 / 波长）。"""

    points: RayFloat3D
    directions: RayFloat3D
    pupil: RayFloat2D
    field: RayFloat2D
    wavelength: RayFloatScalar

    @property
    def population(self) -> int:
        return self.points.shape[0]

    @property
    def num_fields(self) -> int:
        return self.points.shape[1]

    @property
    def num_wavelengths(self) -> int:
        return self.points.shape[2]


@dataclass(slots=True, eq=False, repr=False)
class TraceFlow(TensorContainer):
    """在元件间流动的追迹状态：光线束 + 当前位姿 + 累计裁决。"""

    rays: RayBundle
    transformer: Transformer
    verdict: Verdict

    def with_rays(self, rays: RayBundle) -> Self:
        """替换光线束，返回新状态。"""
        return self.replace(rays=rays)

    def with_transformer(self, transformer: Transformer) -> Self:
        """替换位姿，返回新状态。"""
        return self.replace(transformer=transformer)

    def at_verdict(self, verdict: Verdict) -> Self:
        """链入一个站点裁决（:meth:`Verdict.at` 语义），返回新状态。"""
        return self.replace(verdict=self.verdict.at(verdict))
