from typing import Any, Self, override
from collections.abc import Mapping
from dataclasses import replace

import torch

from core import (
    OpticalModule,
    SystemBoolScalar,
    SystemFloatScalar,
    SystemLongScalar,
    TraceFlow,
    term,
    parse_param,
)
from component.protocol import Component
from shape import Disk


class Stop(Component):
    """光阑：flat 面上的圆形开口，只拦截光线（不折射、不改介质、方向不变）。

    开口内（r ≤ D/2）的光线存活并原样穿过；开口外判死 ``APERTURE_CLIP``。
    直径可训练（严格 opt-in）、可变异——分别经 :class:`Disk` 的 trainable
    映射与 ``mutable`` 词表复用既有机制。

    Args:
        diameter: 开口直径 (mm)，标量或 (P,) 张量；内部包装为 :class:`Disk`。
        trainable: 直径是否注册为可训练参数（GA / 梯度优化）。
    """

    kind = term.STOP

    def __init__(
        self,
        diameter: SystemFloatScalar | float,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(diameter, (int, float)):
            diameter = torch.tensor([float(diameter)])
        self.shape = Disk(diameter, trainable={term.DIAMETER.canonical: trainable})

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """求交 → 光线推进到光阑面，孔径裁决沿流链入（方向不变）。"""
        hit = self.shape(flow.rays.points, flow.rays.directions, flow.transformer)
        rays = replace(flow.rays, points=hit.points_global)
        return flow.with_rays(rays).at_verdict(hit.verdict)

    @override
    def mutate_(self, indices: SystemLongScalar, options: Mapping[str, Any]) -> None:
        """直径变异委托给 Disk（其 ``mutable`` 已含 ``DIAMETER``）。"""
        self.shape.mutate_(indices, options)

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        """可训练性严格 opt-in：仅当 ``train`` 映射显式点名 ``diameter`` 时
        才注册为可训练参数；未给 ``train`` 或未点名一律冻结。"""
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            trainable=term.DIAMETER.resolve(
                term.TRAIN.resolve(options, default={}), default=False
            ),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择开口直径，trainable 语义从 *new* 继承。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            diameter=torch.where(mask, new.shape.Diameter, old.shape.Diameter),
            trainable=bool(new.shape.Diameter.requires_grad),
        )

    @override
    def clone(self) -> Self:
        """深拷贝：以当前直径重建，保留 trainable 语义。"""
        return type(self)(
            diameter=self.shape.Diameter.clone(),
            trainable=bool(self.shape.Diameter.requires_grad),
        )
