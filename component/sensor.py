from typing import Any, Self, override
from collections.abc import Mapping
from dataclasses import replace

from core import OpticalModule, SystemBoolScalar, TraceFlow, term
from component.protocol import Component
from shape import Shape


class Sensor(Component):
    """像面传感器：终端元件（一个 Shape，通常平面 Disk），光线求交后落在面上止步。

    方向保持不变（传感器不改变传播，只记录落点）；
    孔径外的光线由机械孔径裁决判死。无可变异参数
    （``mutable`` 为空，GA 操作对直径的重排/填充由默认实现完成）。

    Args:
        shape: 传感面面形。
    """

    kind = term.SENSOR

    def __init__(self, *, shape: Shape) -> None:
        super().__init__()
        self.shape = shape

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """求交 → 光线落在传感面上，孔径裁决沿流链入。"""
        hit = self.shape(flow.rays.points, flow.rays.directions, flow.transformer)
        rays = replace(flow.rays, points=hit.points_global)
        return flow.with_rays(rays).at_verdict(hit.verdict)

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
