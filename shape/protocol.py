from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self, cast
from collections.abc import Mapping

from core import (
    Noun,
    OpticalModule,
    RayFloat3D,
    SystemFloatScalar,
    Transformer,
    init_param,
    term,
)
from implicit import (
    NewtonSolverOptions,
    SagFunction,
    SolverFunction,
    lift_raw,
    make_solver_options,
    solve,
)
from shape.trace import (
    ApertureFunction,
    TraceResult,
    circle_aperture,
    intersect,
)


class Shape(OpticalModule, ABC):
    kind: ClassVar[Noun]
    _REGISTRY: ClassVar[dict[Noun, type["Shape"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if kind := cls.__dict__.get("kind"):
            cls._REGISTRY[kind] = cls

    def __init__(
        self,
        diameter: SystemFloatScalar,
        *,
        solver_opts: NewtonSolverOptions | Mapping[str, Any] | None = None,
        trainable: Mapping[str, bool] | None = None,
    ):
        """统一注册机械直径（恒为可训练参数）并装配求解器。

        Args:
            diameter: 机械直径 (mm)，``(P,)`` 张量。
            solver_opts: 求解器选项实例或配置映射，缺省为 Newton 默认值。
            trainable: 其余参数的可训练标记（严格 opt-in，键经词表校验由
                子类解释）。
        """
        super().__init__()
        self.Diameter = init_param(self, term.DIAMETER, diameter, True)

        if solver_opts is None:
            solver_opts = NewtonSolverOptions()
        elif isinstance(solver_opts, Mapping):
            solver_opts = make_solver_options(**solver_opts)
        self._solver_opts = solver_opts
        self._solver_fn: SolverFunction = solve(solver_opts)

        if trainable is not None and not isinstance(trainable, Mapping):
            raise TypeError("trainable must be a mapping or None")
        self.trainable = dict(trainable or {})

    @abstractmethod
    def sag(self) -> SagFunction:
        """矢高函数（每次调用重建，确保读到当前参数）。"""

    def aperture(self) -> ApertureFunction:
        """机械孔径函数：默认圆形 ``Diameter / 2``，特殊孔径才覆盖。"""
        return circle_aperture(self.Diameter.mul(0.5))

    def forward(
        self, points: RayFloat3D, directions: RayFloat3D, transformer: Transformer
    ) -> TraceResult:
        """沿光线推进，返回命中点、法向、裁决等信息。"""
        implicit = lift_raw(self.sag())
        return intersect(
            points, directions, transformer, implicit, self._solver_fn, self.aperture()
        )

    @classmethod
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        if cls is not Shape:
            raise NotImplementedError(f"{cls.__name__} must implement from_options()")
        kind = term.SHAPE.resolve(options)
        for noun, sub in cls._REGISTRY.items():
            if kind in noun:
                return cast(Self, sub.from_options(population, options))
        raise ValueError(
            f"Unknown shape: {kind!r} "
            f"(available: {[n.canonical for n in cls._REGISTRY]})"
        )
