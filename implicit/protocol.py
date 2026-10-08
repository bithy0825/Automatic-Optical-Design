"""契约层：求解器选项、求值结果与函数协议。

``FieldResult`` 的 ``gradient`` / ``hessian`` 按求值阶数可选：低于所需阶
时为 ``None``，经 :meth:`FieldResult.grad` / :meth:`FieldResult.hess`
断言访问器取用，调用方不必逐处处理 ``None``。
"""

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
    """求解器选项（构造即校验）。"""

    class Method(StrEnum):
        NEWTON = "newton"
        HALLEY = "halley"

    class Init(StrEnum):
        CLOSEST = "closest"
        RANDOM = "random"
        ZERO = "zero"

    tol: float = 1e-4  # 收敛阈值（|f| < tol 视为收敛）
    num_iter: int = 6  # 迭代步数（末步带梯度展开）
    damping: float = 0.95  # 阻尼系数 (0, 1]
    allow_negative: bool = False  # 允许负根（首面合法，中间面 X 型打架判死）
    init_method: Init = Init.CLOSEST  # 初值策略
    method: Method = Method.NEWTON  # 单步格式（子类以 init=False 覆盖）

    def __post_init__(self) -> None:
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}")
        if self.num_iter <= 0:
            raise ValueError(f"num_iter must be positive, got {self.num_iter}")
        if not 0 < self.damping <= 1:
            raise ValueError(f"damping must be in (0, 1], got {self.damping}")

    def update(self, **kwargs) -> Self:
        """返回一个更新了指定字段的新实例。"""
        return replace(self, **kwargs)


@dataclass(slots=True, eq=False)
class HalleySolverOptions(NewtonSolverOptions):
    """三阶 Halley 求解器的选项（每步需求一个 Hessian）。"""

    method: NewtonSolverOptions.Method = field(
        default=NewtonSolverOptions.Method.HALLEY, init=False
    )


@dataclass(frozen=True, slots=True)
class FieldResult:
    """sag / 隐式函数的求值结果（``gradient`` / ``hessian`` 按 order 可选）。

    ``Order.VALUE`` 仅求值；``GRADIENT`` 附一阶导；``HESSIAN`` 附二阶导。
    """

    class Order(IntEnum):
        VALUE = 0
        GRADIENT = 1
        HESSIAN = 2

    value: RayFloatScalar
    verdict: Verdict
    gradient: RayFloat3D | None = None
    hessian: RayFloatMatrix2D | RayFloatMatrix3D | None = None

    def grad(self) -> RayFloat3D:
        """一阶导（求值阶数不足时断言失败）。"""
        assert self.gradient is not None, "gradient unavailable: evaluate with order >= GRADIENT"
        return self.gradient

    def hess(self) -> RayFloatMatrix2D | RayFloatMatrix3D:
        """二阶导（求值阶数不足时断言失败）。"""
        assert self.hessian is not None, "hessian unavailable: evaluate with order >= HESSIAN"
        return self.hessian


@dataclass(frozen=True, slots=True)
class SolverResult:
    """求解器结果（``distances`` / ``value`` / ``verdict`` 严格同点）。"""

    distances: RayFloatScalar
    value: RayFloatScalar
    verdict: Verdict


class SagFunction(Protocol):
    """矢高函数：横向 (x, y) → 矢高 z 及其导数。"""

    def __call__(self, points: RayFloat2D, *, order: FieldResult.Order) -> FieldResult: ...


class ImplicitFunction(Protocol):
    """3D 隐式函数：``f(x, y, z) = 0`` 等值面即曲面。"""

    def __call__(self, points: RayFloat3D, *, order: FieldResult.Order) -> FieldResult: ...


class SolverFunction(Protocol):
    """沿光线求解与隐式曲面的交点距离。"""

    def __call__(
        self,
        points: RayFloat3D,
        directions: RayFloat3D,
        implicit: ImplicitFunction,
    ) -> SolverResult: ...
