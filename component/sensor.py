from typing import Any, Self, override
from collections.abc import Mapping
from dataclasses import replace

import torch

from core import OpticalModule, SystemBoolScalar, SystemFloatScalar, TraceFlow, term, parse_param
from component.protocol import Component
from shape import Disk


class Sensor(Component):
    """像面传感器：终端元件，光线求交后落在面上止步。

    方向保持不变（传感器不改变传播，只记录落点）；
    孔径外的光线由机械孔径裁决判死。无可变异参数
    （``mutable`` 为空，GA 操作对直径的重排/填充由默认实现完成）。

    Args:
        diameter: 传感器直径 (mm)，标量或 (P,) 张量；内部包装为 :class:`Disk`。
    """

    kind = term.SENSOR

    def __init__(self, diameter: SystemFloatScalar | float) -> None:
        super().__init__()
        if isinstance(diameter, (int, float)):
            diameter = torch.tensor([float(diameter)])
        self.shape = Disk(diameter)

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """求交 → 光线落在传感面上，孔径裁决沿流链入。"""
        hit = self.shape(flow.rays.points, flow.rays.directions, flow.transformer)
        rays = replace(flow.rays, points=hit.points_global)
        return flow.with_rays(rays).at_verdict(hit.verdict)

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(parse_param(options, term.DIAMETER, population))

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择传感面直径。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(torch.where(mask, new.shape.Diameter, old.shape.Diameter))

    @override
    def clone(self) -> Self:
        """深拷贝：以当前直径重建（传感器无材料）。"""
        return type(self)(self.shape.Diameter.clone())
