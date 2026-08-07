from typing import Any, Self, override
from collections.abc import Mapping
from dataclasses import replace

from core import (
    OpticalModule,
    SystemBoolScalar,
    SystemLongScalar,
    TraceFlow,
    term,
)
from component.protocol import Component
from shape import Shape


class Stop(Component):
    """光阑：一个 Shape（通常平面 Disk），只拦截光线（不折射、不改介质、方向不变）。

    开口内（r ≤ D/2）的光线存活并原样穿过；开口外判死 ``APERTURE_CLIP``。
    求解器选项、可训练性与变异规则全部由面形配置驱动——前置光阑（与光源
    同面）需 ``solver = { allow_negative = true }``，否则 t≈0 的数值噪声
    会被误判为负距离死亡。

    Args:
        shape: 光阑面形。
    """

    kind = term.STOP

    def __init__(self, *, shape: Shape) -> None:
        super().__init__()
        self.shape = shape

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """求交 → 光线推进到光阑面，孔径裁决沿流链入（方向不变）。"""
        hit = self.shape(flow.rays.points, flow.rays.directions, flow.transformer)
        rays = replace(flow.rays, points=hit.points_global)
        return flow.with_rays(rays).at_verdict(hit.verdict)

    @override
    def mutate_(self, indices: SystemLongScalar, options: Mapping[str, Any]) -> None:
        """直径变异委托给面形（其 ``mutable`` 已含 ``DIAMETER``）。"""
        self.shape.mutate_(indices, options)

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        """面形由配置构造（同 Refractor）；``shape`` 键缺省为 ``disk``。"""
        return cls(shape=Shape.from_options(population, {"shape": "disk", **options}))

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """面形多态分派各自的 ``where``（语义同 Refractor）。"""
        OpticalModule._check_operands(mask, new, old)
        if type(new.shape) is not type(old.shape):
            raise TypeError(
                f"where: shape {type(new.shape).__name__} vs {type(old.shape).__name__}"
            )
        return cls(shape=type(new.shape).where(mask, new.shape, old.shape))

    @override
    def clone(self) -> Self:
        """深拷贝：面形克隆（求解器与 trainable 语义随之保留）。"""
        return type(self)(shape=self.shape.clone())
