from typing import Any, Self, override
from collections.abc import Mapping

import torch

from core import (
    OpticalModule,
    SystemBoolScalar,
    SystemFloatScalar,
    term,
    parse_param,
)
from implicit import (
    NewtonSolverOptions,
    SagFunction,
    flat_sag,
)
from shape.protocol import Shape


class Disk(Shape):
    kind = term.DISK
    mutable = (term.DIAMETER,)

    def __init__(
        self,
        diameter: SystemFloatScalar,
        *,
        solver_opts: NewtonSolverOptions | Mapping[str, Any] | None = None,
        trainable: Mapping[str, bool] | None = None,
    ):
        super().__init__(diameter, solver_opts=solver_opts, trainable=trainable)

    @override
    def sag(self) -> SagFunction:
        return flat_sag()

    @override
    def clone(self) -> Self:
        return type(self)(
            diameter=self.Diameter.clone(),
            solver_opts=self._solver_opts,
            trainable=self.trainable.copy(),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择直径；求解器与 trainable 配置从 *new* 继承。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            diameter=torch.where(mask, new.Diameter, old.Diameter),
            solver_opts=new._solver_opts,
            trainable=new.trainable.copy(),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )
