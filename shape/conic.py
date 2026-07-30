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
    conical_sag,
)
from shape.protocol import Shape


class Conic(Shape):
    kind = term.CONIC
    mutable = (term.DIAMETER, term.CURVATURE, term.KAPPA)

    def __init__(
        self,
        diameter: SystemFloatScalar,
        curvature: SystemFloatScalar,
        kappa: SystemFloatScalar,
        *,
        solver_opts: NewtonSolverOptions | Mapping[str, Any] | None = None,
        trainable: Mapping[str, bool] | None = None,
    ):
        super().__init__(diameter, solver_opts=solver_opts, trainable=trainable)

        train_C = False
        train_K = False
        for k in self.trainable:
            if (
                not term.CURVATURE.match(k)
                and not term.KAPPA.match(k)
                and not term.DIAMETER.match(k)
            ):
                warnings.warn(
                    f"Unknown trainable key: {k}. Only 'curvature', 'kappa' and 'diameter' are supported for Conic."
                )
            else:
                if term.CURVATURE.match(k):
                    train_C = self.trainable[k]
                if term.KAPPA.match(k):
                    train_K = self.trainable[k]

        self.Curvature = init_param(self, term.CURVATURE, curvature, train_C)
        self.Kappa = init_param(self, term.KAPPA, kappa, train_K)

    @override
    def sag(self) -> SagFunction:
        return conical_sag(self.Curvature, self.Kappa)

    @override
    def clone(self) -> Self:
        return type(self)(
            diameter=self.Diameter.clone(),
            curvature=self.Curvature.clone(),
            kappa=self.Kappa.clone(),
            solver_opts=self._solver_opts,
            trainable=self.trainable.copy(),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择直径、曲率与锥面常数；求解器与 trainable 配置从 *new* 继承。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            diameter=torch.where(mask, new.Diameter, old.Diameter),
            curvature=torch.where(mask, new.Curvature, old.Curvature),
            kappa=torch.where(mask, new.Kappa, old.Kappa),
            solver_opts=new._solver_opts,
            trainable=new.trainable.copy(),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            curvature=parse_param(options, term.CURVATURE, population),
            kappa=parse_param(options, term.KAPPA, population),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )
