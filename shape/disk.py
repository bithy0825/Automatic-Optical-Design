from typing import Any, Self, override
from collections.abc import Mapping

from core import (
    SystemFloatScalar,
    term,
    fmt_param,
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

    def extra_repr(self) -> str:
        return f"{term.DIAMETER.canonical}={fmt_param(self.Diameter)},\n"

    @override
    def clone(self) -> Self:
        return type(self)(
            diameter=self.Diameter.clone(),
            solver_opts=self._solver_opts,
            trainable=self.trainable.copy(),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )
