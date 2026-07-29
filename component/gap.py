from typing import Any, Self, override
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from core import (
    term,
    SystemFloatScalar,
    TraceFlow,
    Transformer,
    init_param,
    parse_param,
)
from component.protocol import Component


class Gap(Component):
    """间隔：将 ``flow.transformer`` 沿局部 z 轴（光轴）推进 ``thickness``。

    光线与裁决原样传递——介质中的自由传播由下一个元件的求交完成，
    Gap 只负责把位姿原点送到下一个面。

    Args:
        thickness: 间隔厚度 (mm)，标量或 (P,) 张量。
        trainable: 是否注册为可训练参数（GA / 梯度优化）。
    """

    kind = term.GAP
    mutable = (term.THICKNESS,)

    def __init__(
        self,
        thickness: SystemFloatScalar | float,
        *,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        value: float | Sequence[float] | torch.Tensor = (
            [float(thickness)] if isinstance(thickness, (int, float)) else thickness
        )
        self.Thickness = init_param(self, term.THICKNESS, value, trainable)

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        tf = flow.transformer
        t = self.Thickness.to(device=tf.device, dtype=tf.dtype)
        step = F.pad(t.unsqueeze(-1), (2, 0))  # (P, 3) = (0, 0, t)
        return flow.with_transformer(tf.then(Transformer.translation(step)))

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        """可训练性严格 opt-in：仅当 ``train`` 映射显式点名 ``thickness`` 时
        才注册为可训练参数；未给 ``train`` 或未点名一律冻结。"""
        return cls(
            thickness=parse_param(options, term.THICKNESS, population).clamp_min(0.0),
            trainable=term.THICKNESS.resolve(
                term.TRAIN.resolve(options, default={}), default=False
            ),
        )

    @override
    def clone(self) -> Self:
        """深拷贝：以当前厚度重建，保留 trainable 语义。"""
        return type(self)(
            thickness=self.Thickness.clone(),
            trainable=bool(self.Thickness.requires_grad),
        )
