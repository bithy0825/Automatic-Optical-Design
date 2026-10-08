"""矢高 → 3D 隐式函数的提升：``f(x, y, z) = s(x, y) − z``。

z 分量只以 ``−z`` 进入，故梯度 z 分量恒为 −1，Hessian 末行/列恒零。
"""

import torch

from core import RayFloat3D, RayFloatMatrix3D, RayFloatScalar
from implicit.protocol import FieldResult, ImplicitFunction, SagFunction


def _sym3x3_sag_hessian(
    d00: RayFloatScalar, off01: RayFloatScalar, d11: RayFloatScalar
) -> RayFloatMatrix3D:
    """组装 ``[[d00, off01, 0], [off01, d11, 0], [0, 0, 0]]``（lift 结构）。"""
    zero = torch.zeros_like(d00)
    row0 = torch.stack((d00, off01, zero), dim=-1)
    row1 = torch.stack((off01, d11, zero), dim=-1)
    row2 = torch.stack((zero, zero, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def lift_raw(sag: SagFunction) -> ImplicitFunction:
    """矢高 → 3D 隐式函数的提升：``f(x, y, z) = s(x, y) − z``。"""

    def lifted(points: RayFloat3D, *, order: FieldResult.Order) -> FieldResult:
        z = points[..., 2]  # 光轴分量

        sr = sag(points[..., :2], order=order)  # 横向 xy → 矢高

        grad = None
        if order >= FieldResult.Order.GRADIENT:
            grad_x, grad_y = sr.grad().unbind(dim=-1)
            grad = torch.stack((grad_x, grad_y, torch.full_like(z, -1.0)), dim=-1)

        hess = None
        if order >= FieldResult.Order.HESSIAN:
            h = sr.hess()
            hess = _sym3x3_sag_hessian(h[..., 0, 0], h[..., 0, 1], h[..., 1, 1])

        return FieldResult(
            value=sr.value.sub(z),  # f = s(x, y) − z
            verdict=sr.verdict,
            gradient=grad,
            hessian=hess,
        )

    return lifted
