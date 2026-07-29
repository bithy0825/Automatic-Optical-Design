import torch

from core import RayFloatMatrix2D, RayFloatMatrix3D, RayFloatScalar


def _sym2x2(
    d00: RayFloatScalar,
    off01: RayFloatScalar,
    d11: RayFloatScalar,
) -> RayFloatMatrix2D:
    """组装对称 2×2 矩阵 ``[[d00, off01], [off01, d11]]``。

    Args:
        d00: 对角元 (0, 0)。
        off01: 非对角元 (0, 1) == (1, 0)。
        d11: 对角元 (1, 1)。

    Returns:
        形状 ``(P, F, W, N, 2, 2)`` 的对称矩阵。
    """
    row0 = torch.stack((d00, off01), dim=-1)
    row1 = torch.stack((off01, d11), dim=-1)
    return torch.stack((row0, row1), dim=-2)


def _sym3x3_sag_hessian(
    d00: RayFloatScalar,
    off01: RayFloatScalar,
    d11: RayFloatScalar,
) -> RayFloatMatrix3D:
    """组装对称 3×3 矩阵，其末行/列为零（lift 结构）。

    结果为 ``[[d00, off01, 0], [off01, d11, 0], [0, 0, 0]]``：光轴 z 对矢高
    无贡献（``f = s(x,y) − z`` → ``∂²f/∂z∂* ≡ 0``），故 Hessian 末行/列恒零。

    Args:
        d00: 对角元 (0, 0) —— ∂²s/∂x²。
        off01: 非对角元 (0, 1) == (1, 0) —— ∂²s/∂x∂y。
        d11: 对角元 (1, 1) —— ∂²s/∂y²。

    Returns:
        形状 ``(P, F, W, N, 3, 3)`` 的对称矩阵。
    """
    zero = torch.zeros_like(d00)
    row0 = torch.stack((d00, off01, zero), dim=-1)
    row1 = torch.stack((off01, d11, zero), dim=-1)
    row2 = torch.stack((zero, zero, zero), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def _broadcast_coeff(
    coeff: torch.Tensor,
    ray_scalar: RayFloatScalar,
) -> torch.Tensor:
    """把系统维系数 ``[P, Ncoeff]`` 广播到光线维 ``[P, 1, 1, 1, Ncoeff]``。

    在矢高评估中，每个光学系统的多项式系数需与 ``(F, W, N)`` 光线维对齐。
    本函数插入 3 个单例轴并 ``expand`` 到目标形状，同时匹配 dtype。

    Args:
        coeff: 系数张量，形状 ``(P, Ncoeff)``。
        ray_scalar: 用于推断目标设备/dtype 的逐光线标量，形状 ``(P, F, W, N)``。

    Returns:
        广播后的系数，形状 ``(P, 1, 1, 1, Ncoeff)``，dtype 与 *ray_scalar* 一致。
    """
    new_shape = (coeff.shape[0], 1, 1, 1, coeff.shape[-1])
    return (
        coeff.reshape(new_shape)
        .expand(
            ray_scalar.shape[0],
            ray_scalar.shape[1],
            ray_scalar.shape[2],
            ray_scalar.shape[3],
            coeff.shape[-1],
        )
        .to(ray_scalar.dtype)
    )
