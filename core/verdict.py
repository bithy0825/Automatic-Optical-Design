"""追迹裁决：一根光线在某站点（shape 域 / 求解器 / 孔径 / TIR）的生死与代价。

契约
----
* ``hold``  —— 存活标志。``False`` 即死亡，死亡即冻结，绝不复活。
* ``toll``  —— 非负"代价"。构造站点（:meth:`Verdict.site`）时可传有符号
  量：正 = 存活余量、负 = 死亡深度；经 :meth:`at` 链入后一律归一为非负，
  供死亡损失反向传播。
* ``cause`` —— 死因（:class:`Verdict.Cause`），首次判死的站点钉死，后续
  站点不覆盖。
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Self

import torch

from core.aliases import RayBoolScalar, RayFloatScalar, RayLongScalar
from core.container import TensorContainer


@dataclass(slots=True, eq=False, repr=False)
class Verdict(TensorContainer):
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
    def site(
        cls, hold: RayBoolScalar, toll: RayFloatScalar, cause: Cause | int
    ) -> Self:
        """构造一个站点的裁决。*toll* 可传有符号量（正=余量，负=死亡深度）。"""
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
