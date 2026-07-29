import torch

from core import RayFloat3D
from implicit._tensor_utils import _sym3x3_sag_hessian
from implicit.protocol import FieldResult, ImplicitFunction, SagFunction


def lift_raw(sag: SagFunction) -> ImplicitFunction:
    """矢高 → 3D 隐式函数的提升：``f(x, y, z) = s(x, y) − z``。"""

    def lifted(points: RayFloat3D, *, order: FieldResult.Order) -> FieldResult:
        z = points[..., 2]  # 光轴分量

        sr = sag(points[..., :2], order=order)  # 横向 xy → 矢高
        f_val = sr.value.sub(z)  # f = s(x, y) − z

        verdict = sr.verdict

        f_grad = None
        if order >= FieldResult.Order.GRADIENT:
            grad_x, grad_y = sr.gradient.unbind(dim=-1)
            f_grad = torch.stack((grad_x, grad_y, torch.full_like(z, -1.0)), dim=-1)

        f_hess = None
        if order >= FieldResult.Order.HESSIAN:
            g_xx = sr.hessian[..., 0, 0]
            g_xy = sr.hessian[..., 0, 1]
            g_yy = sr.hessian[..., 1, 1]
            f_hess = _sym3x3_sag_hessian(g_xx, g_xy, g_yy)

        return FieldResult(
            _value=f_val, _verdict=verdict, _gradient=f_grad, _hessian=f_hess
        )

    return lifted
