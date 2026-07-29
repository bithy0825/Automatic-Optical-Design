from typing import Any, Self, override
from collections.abc import Mapping
import warnings

from core import (
    SystemFloatScalar,
    init_param,
    term,
    fmt_param,
    parse_param,
)
from implicit import (
    NewtonSolverOptions,
    SagFunction,
    conical_sag,
)
from shape.protocol import Shape
from shape._utils import fmt_curv_pair


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
            if not term.CURVATURE.match(k) and not term.KAPPA.match(k):
                warnings.warn(
                    f"Unknown trainable key: {k}. Only 'curvature' and 'kappa' are supported for Conic."
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

    def extra_repr(self) -> str:
        return (
            f"{term.DIAMETER.canonical}={fmt_param(self.Diameter)},\n"
            f"{fmt_curv_pair(self.Curvature)},\n"
            f"{term.KAPPA.canonical}={fmt_param(self.Kappa)}"
        )

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
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            curvature=parse_param(options, term.CURVATURE, population),
            kappa=parse_param(options, term.KAPPA, population),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )
