"""隐式曲面模块：矢高（sag）→ 隐式曲面 → 光线求交求解器。

模块组织
--------
* :mod:`implicit.protocol` — 契约层（求解器选项、结果数据类、函数协议）。
* :mod:`implicit.sag` — 矢高函数（球面/圆锥/非球面/平面）。
* :mod:`implicit.lift` — 矢高 → 3D 隐式函数的提升。
* :mod:`implicit.guess` — 光线‑曲面交点的初值策略。
* :mod:`implicit.solver` — 迭代求解器（Newton / Halley）。
"""

# ── 契约层 ──
from implicit.protocol import (
    FieldResult,
    HalleySolverOptions,
    ImplicitFunction,
    NewtonSolverOptions,
    SagFunction,
    SolverFunction,
    SolverResult,
)

# ── 矢高函数 ──
from implicit.sag import aspheric_sag, conical_sag, flat_sag, spherical_sag

# ── 提升函数 ──
from implicit.lift import lift_raw

# ── 初值策略 ──
from implicit.guess import guess, init_closest, init_random, init_zero

# ── 求解器 ──
from implicit.solver import halley_step, make_solver_options, newton_step, solve

__all__ = [
    # protocol — 选项
    "NewtonSolverOptions",
    "HalleySolverOptions",
    # protocol — 结果
    "FieldResult",
    "SolverResult",
    # protocol — 协议
    "SagFunction",
    "ImplicitFunction",
    "SolverFunction",
    # sag
    "spherical_sag",
    "conical_sag",
    "aspheric_sag",
    "flat_sag",
    # lift
    "lift_raw",
    # guess
    "init_closest",
    "init_zero",
    "init_random",
    "guess",
    # solver
    "newton_step",
    "halley_step",
    "solve",
    "make_solver_options",
]
