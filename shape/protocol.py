"""光学面形抽象基类：sag 契约、求解器装配与注册表分派。

子类声明 ``kind`` 名词即自动注册；``from_options`` 按 ``shape`` 键分派。
可训练标记严格 opt-in：``diameter`` 由基类统一消费，其余参数名词经
:meth:`Shape._train_flags` 由子类认领，未知键告警并忽略。
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Self, cast

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
from shape.trace import ApertureFunction, TraceResult, circle_aperture, intersect


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
        """统一注册机械直径并装配求解器。

        Args:
            diameter: 机械直径 (mm)，``(P,)`` 张量。
            solver_opts: 求解器选项实例或配置映射，缺省为 Newton 默认值。
            trainable: 可训练标记（严格 opt-in：仅当映射显式点名才注册为
                可训练参数，否则冻结为 buffer，由 GA 变异 + bounds 铰链演化
                约束；其余键经词表校验由子类解释）。
        """
        super().__init__()
        if trainable is not None and not isinstance(trainable, Mapping):
            raise TypeError("trainable must be a mapping or None")
        self.trainable = dict(trainable or {})

        train_D = bool(term.DIAMETER.resolve(self.trainable, default=False))
        self.Diameter = init_param(self, term.DIAMETER, diameter, train_D)

        if solver_opts is None:
            solver_opts = NewtonSolverOptions()
        elif isinstance(solver_opts, Mapping):
            solver_opts = make_solver_options(**solver_opts)
        self._solver_opts = solver_opts
        self._solver_fn: SolverFunction = solve(solver_opts)

    def _train_flags(self, *nouns: Noun) -> dict[Noun, bool]:
        """解析 ``trainable`` 映射中各参数名词的训练标志（缺省 ``False``）。

        ``diameter`` 由基类统一消费、不在 *nouns* 中列出；未知名词告警并忽略。
        """
        flags: dict[Noun, bool] = dict.fromkeys(nouns, False)
        for key, value in self.trainable.items():
            if term.DIAMETER.match(key):
                continue
            for noun in nouns:
                if noun.match(key):
                    flags[noun] = bool(value)
                    break
            else:
                supported = ["diameter", *(n.canonical.lower() for n in nouns)]
                warnings.warn(
                    f"unknown trainable key {key!r} for {type(self).__name__} "
                    f"(supported: {supported})",
                    stacklevel=2,
                )
        return flags

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
            f"unknown shape: {kind!r} (available: {[n.canonical for n in cls._REGISTRY]})"
        )
