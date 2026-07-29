from collections.abc import Callable
from typing import Final

import torch

from core import RayFloat3D, RayFloatScalar, sturdy_div
from implicit.protocol import ImplicitFunction, NewtonSolverOptions

_Init = NewtonSolverOptions.Init


def init_closest(
    points: RayFloat3D,
    directions: RayFloat3D,
    *,
    implicit: ImplicitFunction,
) -> RayFloatScalar:
    """最近点初值：光线到原点的最近点参数 ``t* = −P·V / |V|²``。"""
    del implicit  # unused
    return sturdy_div(
        points.neg().mul(directions).sum(dim=-1),
        directions.square().sum(dim=-1),
    )


def init_zero(
    points: RayFloat3D,
    directions: RayFloat3D,
    *,
    implicit: ImplicitFunction,
) -> RayFloatScalar:
    """零初值：``t = 0``（光线起点处）。"""
    del implicit, directions  # unused
    return torch.zeros_like(points[..., 0])


def init_random(
    points: RayFloat3D,
    directions: RayFloat3D,
    *,
    implicit: ImplicitFunction,
) -> RayFloatScalar:
    """随机初值：``t ~ U[0, 1)``。"""
    del implicit, directions  # unused
    return torch.rand_like(points[..., 0])


_INIT_METHODS: Final[dict[_Init, Callable[..., RayFloatScalar]]] = {
    _Init.CLOSEST: init_closest,
    _Init.RANDOM: init_random,
    _Init.ZERO: init_zero,
}


def guess(
    points: RayFloat3D,
    directions: RayFloat3D,
    *,
    implicit: ImplicitFunction,
    init_method: _Init = _Init.CLOSEST,
) -> RayFloatScalar:
    if init_method not in _INIT_METHODS:
        raise ValueError(f"Unknown init method: {init_method}")
    return _INIT_METHODS[init_method](
        points=points, directions=directions, implicit=implicit
    )
