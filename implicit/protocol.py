from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum
from typing import Protocol, Self

from core import (
    RayFloat2D,
    RayFloat3D,
    RayFloatMatrix2D,
    RayFloatMatrix3D,
    RayFloatScalar,
    Verdict,
)


@dataclass(slots=True, eq=False)
class NewtonSolverOptions:
    """求解器选项。

    Attributes:
        tol:            收敛阈值（|f| < tol 视为收敛）。
        num_iter:       迭代步数（末步带梯度展开）。
        damping:        阻尼系数 (0, 1]。
        allow_negative: 允许负 distances 根（首面合法，中间面为 X 型打架判死）。
        init_method:    初值策略。
        method:         单步格式（子类以 ``init=False`` 覆盖）。
    """

    class Method(StrEnum):
        NEWTON = "newton"
        HALLEY = "halley"

    class Init(StrEnum):
        CLOSEST = "closest"
        RANDOM = "random"
        ZERO = "zero"

    tol: float = 1e-4
    num_iter: int = 6
    damping: float = 0.95
    allow_negative: bool = False
    init_method: Init = Init.CLOSEST
    method: Method = Method.NEWTON

    def __post_init__(self) -> None:
        assert self.tol > 0, "Tolerance must be positive"
        assert self.num_iter > 0, "Number of iterations must be positive"
        assert 0 < self.damping <= 1, "Damping must be in (0, 1]"

    def update(self, **kwargs) -> Self:
        """返回一个新的选项实例，更新指定的字段。"""
        return replace(self, **kwargs)


@dataclass(slots=True, eq=False)
class HalleySolverOptions(NewtonSolverOptions):
    """三阶 Halley 求解器的选项（每步需求一个 Hessian）。"""

    method: NewtonSolverOptions.Method = field(
        default=NewtonSolverOptions.Method.HALLEY, init=False
    )


@dataclass(frozen=True, slots=True)
class FieldResult:
    """sag / 隐式函数的求值结果（梯度与 Hessian 按 order 可选）。

    访问器集中断言可用性，调用方不必逐处 ``assert is not None``。
    """

    class Order(IntEnum):
        VALUE = 0
        GRADIENT = 1
        HESSIAN = 2

    _value: RayFloatScalar
    _verdict: Verdict
    _gradient: RayFloat3D | None = None
    _hessian: RayFloatMatrix2D | RayFloatMatrix3D | None = None

    @property
    def value(self) -> RayFloatScalar:
        return self._value

    @property
    def verdict(self) -> Verdict:
        return self._verdict

    @property
    def gradient(self) -> RayFloat3D:
        assert self._gradient is not None, "Gradient is not available"
        return self._gradient

    @property
    def hessian(self) -> RayFloatMatrix2D | RayFloatMatrix3D:
        assert self._hessian is not None, "Hessian is not available"
        return self._hessian


@dataclass(frozen=True, slots=True)
class SolverResult:
    """求解器结果（distances / value / verdict 严格同点）。"""

    _distances: RayFloatScalar
    _value: RayFloatScalar
    _verdict: Verdict

    @property
    def distances(self) -> RayFloatScalar:
        return self._distances

    @property
    def value(self) -> RayFloatScalar:
        return self._value

    @property
    def verdict(self) -> Verdict:
        return self._verdict


class SagFunction(Protocol):
    """矢高函数：横向 (x, y) → 矢高 z 及其导数。"""

    def __call__(
        self, points: RayFloat2D, *, order: FieldResult.Order
    ) -> FieldResult: ...


class ImplicitFunction(Protocol):
    """3D 隐式函数：``f(x, y, z) = 0`` 等值面即曲面。"""

    def __call__(
        self, points: RayFloat3D, *, order: FieldResult.Order
    ) -> FieldResult: ...


class SolverFunction(Protocol):
    """沿光线求解与隐式曲面的交点距离。"""

    def __call__(
        self,
        points: RayFloat3D,
        directions: RayFloat3D,
        implicit: ImplicitFunction,
    ) -> SolverResult: ...
