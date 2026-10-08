"""圆锥曲面（椭球 / 抛物面 / 双曲面）。"""

from collections.abc import Mapping
from typing import Any, Self, override

import torch

from core import (
    OpticalModule,
    SystemBoolScalar,
    SystemFloatScalar,
    init_param,
    parse_param,
    term,
)
from implicit import NewtonSolverOptions, SagFunction, conical_sag
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
        flags = self._train_flags(term.CURVATURE, term.KAPPA)
        self.Curvature = init_param(self, term.CURVATURE, curvature, flags[term.CURVATURE])
        self.Kappa = init_param(self, term.KAPPA, kappa, flags[term.KAPPA])

    @override
    def sag(self) -> SagFunction:
        return conical_sag(self.Curvature, self.Kappa)

    @override
    def scale_(self, factor: float) -> None:
        """直径 ×s、曲率 ÷s；圆锥系数 k 无量纲不缩放。"""
        super().scale_(factor)
        self.Curvature.div_(factor)

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
    def where(  # pyright: ignore[reportIncompatibleMethodOverride] 与全仓库 where 语义一致：Self 收窄由 _check_operands 运行时守卫
        cls, mask: SystemBoolScalar, new: Self, old: Self
    ) -> Self:
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
