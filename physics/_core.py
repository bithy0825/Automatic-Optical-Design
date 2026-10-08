"""光线与介质界面的相互作用：镜面反射与 Snell 折射。

约定
----
* 方向、法向均为单位矢量；法向指向入射介质侧（故 ``−V·n = cos θᵢ``）。
* 输出方向一律以 ``F.normalize`` 收尾，浮点漂移不破坏单位长度。
* 折射遇全内反射（TIR）转为反射方向并判死，``toll = 1 − η²sin²θᵢ``
  （正 = 存活余量，负 = 死亡深度，见 :class:`~core.flow.Verdict` 契约）。
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from core import RayFloat3D, RayFloatScalar, Verdict, sturdy_div, sturdy_sqrt
from core.container import TensorContainer


@dataclass(slots=True, eq=False, repr=False)
class InteractionResult(TensorContainer):
    """光线与表面相互作用的结果：出射方向（单位化）+ 交互裁决。"""

    directions: RayFloat3D
    verdict: Verdict


def reflect(directions: RayFloat3D, normals: RayFloat3D) -> InteractionResult:
    """镜面反射 ``R = V − 2(V·n)n``；不判死光线（裁决全活）。"""
    cosine = directions.mul(normals).sum(dim=-1, keepdim=True)  # V·n = −cos θᵢ，(..., 1)
    specular = directions.sub(cosine.mul(2.0).mul(normals))
    return InteractionResult(
        directions=F.normalize(specular, dim=-1),
        verdict=Verdict.alive_like(directions[..., 0]),
    )


def refract(
    directions: RayFloat3D,
    normals: RayFloat3D,
    n1: RayFloatScalar,
    n2: RayFloatScalar,
) -> InteractionResult:
    """Snell 折射（TIR 时转为反射方向并判死）。

    Args:
        directions: ``(P, F, W, N, 3)`` 入射单位方向。
        normals:    ``(P, F, W, N, 3)`` 单位法向，指向入射介质侧。
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
        directions=F.normalize(directions_out, dim=-1),
        verdict=Verdict.site(
            hold=valid.squeeze(-1),
            toll=radicand.squeeze(-1),
            cause=Verdict.Cause.TIR,
        ),
    )
