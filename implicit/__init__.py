"""隐式曲面模块：矢高（sag）→ 隐式曲面 → 光线求交求解器。

模块组织
--------
* :mod:`implicit.protocol` —— 契约层（求解器选项、结果数据类、函数协议）。
* :mod:`implicit.sag` —— 矢高函数（球面 / 圆锥 / 非球面 / 平面）。
* :mod:`implicit.lift` —— 矢高 → 3D 隐式函数的提升。
* :mod:`implicit.guess` —— 光线-曲面交点的初值策略。
* :mod:`implicit.solver` —— 迭代求解器（Newton / Halley）。
"""

from implicit.guess import guess, init_closest, init_random, init_zero
from implicit.lift import lift_raw
from implicit.protocol import (
    FieldResult,
    HalleySolverOptions,
    ImplicitFunction,
    NewtonSolverOptions,
    SagFunction,
    SolverFunction,
    SolverResult,
)
from implicit.sag import aspheric_sag, conical_sag, flat_sag, spherical_sag
from implicit.solver import halley_step, make_solver_options, newton_step, solve

__all__ = [
    # 契约层 —— 选项
    "NewtonSolverOptions",
    "HalleySolverOptions",
    # 契约层 —— 结果
    "FieldResult",
    "SolverResult",
    # 契约层 —— 协议
    "SagFunction",
    "ImplicitFunction",
    "SolverFunction",
    # 矢高函数
    "spherical_sag",
    "conical_sag",
    "aspheric_sag",
    "flat_sag",
    # 提升
    "lift_raw",
    # 初值策略
    "init_closest",
    "init_zero",
    "init_random",
    "guess",
    # 求解器
    "newton_step",
    "halley_step",
    "solve",
    "make_solver_options",
]
