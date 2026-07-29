from typing import Any, Self, override
from collections.abc import Mapping
import warnings

import torch

from core import (
    OpticalModule,
    SystemBoolScalar,
    SystemFloatScalar,
    init_param,
    term,
    parse_param,
)
from implicit import (
    NewtonSolverOptions,
    SagFunction,
    spherical_sag,
)
from shape.protocol import Shape


class Sphere(Shape):
    kind = term.SPHERE
    mutable = (term.DIAMETER, term.CURVATURE)

    def __init__(
        self,
        diameter: SystemFloatScalar,
        curvature: SystemFloatScalar,
        *,
        solver_opts: NewtonSolverOptions | Mapping[str, Any] | None = None,
        trainable: Mapping[str, bool] | None = None,
    ):
        super().__init__(diameter, solver_opts=solver_opts, trainable=trainable)

        train_C = False
        for k in self.trainable:
            if not term.CURVATURE.match(k):
                warnings.warn(
                    f"Unknown trainable key: {k}. Only 'curvature' is supported for Sphere."
                )
            else:
                train_C = self.trainable[k]

        self.Curvature = init_param(self, term.CURVATURE, curvature, train_C)

    @override
    def sag(self) -> SagFunction:
        return spherical_sag(self.Curvature)

    @override
    def clone(self) -> Self:
        return type(self)(
            diameter=self.Diameter.clone(),
            curvature=self.Curvature.clone(),
            solver_opts=self._solver_opts,
            trainable=self.trainable.copy(),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择直径与曲率；求解器与 trainable 配置从 *new* 继承。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            diameter=torch.where(mask, new.Diameter, old.Diameter),
            curvature=torch.where(mask, new.Curvature, old.Curvature),
            solver_opts=new._solver_opts,
            trainable=new.trainable.copy(),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            curvature=parse_param(options, term.CURVATURE, population),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )
