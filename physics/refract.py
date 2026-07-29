import torch
import torch.nn.functional as F

from core import RayFloat3D, RayFloatScalar, Verdict, sturdy_div, sturdy_sqrt
from physics.protocol import InteractionResult


def refract(
    directions: RayFloat3D,
    normals: RayFloat3D,
    n1: RayFloatScalar,
    n2: RayFloatScalar,
) -> InteractionResult:
    """Snell 折射（TIR 时转为反射方向并判死）。

    Args:
        directions: ``(P, F, W, N, 3)`` 入射单位方向。
        normals:    ``(P, F, W, N, 3)`` 单位法向，指向入射介质侧
                    （故 ``-directions · normals = cos θᵢ``）。
        n1:         ``(P, F, W, N)`` 入射侧折射率。
        n2:         ``(P, F, W, N)`` 出射侧折射率。
    """

    cos_i = directions.neg().mul(normals).sum(dim=-1, keepdim=True)  # cos θᵢ
    eta = sturdy_div(n1, n2).unsqueeze(-1)  # (..., 1)

    # 共享子式 cos θᵢ·n：透射切向分量与反射方向 (V + 2cosθᵢ·n) 各用一次，
    # 免去对 reflect() 的整束重复计算。
    cos_n = cos_i.mul(normals)
    tangent = directions.add(cos_n)  # V + cos θᵢ n

    # R_perp = η(V + cosθᵢ n)；radicand = 1 − |R_perp|² = 1 − η²sin²θᵢ，
    # ≥ 0 当且仅当小于临界角。
    r_perp = eta.mul(tangent)
    radicand = r_perp.square().sum(dim=-1, keepdim=True).neg().add(1.0)

    # sturdy sqrt 在负区为零（且零梯度）：TIR 处法向分量恰不贡献，无 NaN。
    transmitted = r_perp.sub(sturdy_sqrt(radicand).mul(normals))
    reflected = tangent.add(cos_n)

    valid = radicand.ge(0.0)  # (..., 1)
    directions_out = torch.where(valid, transmitted, reflected)

    return InteractionResult(
        directions=F.normalize(directions_out, dim=-1, eps=1e-12),
        verdict=Verdict.site(
            hold=valid.squeeze(-1),
            toll=radicand.squeeze(-1),
            cause=Verdict.Cause.TIR,
        ),
    )
