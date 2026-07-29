import torch.nn.functional as F

from physics.protocol import InteractionResult
from core import RayFloat3D, Verdict


def reflect(directions: RayFloat3D, normals: RayFloat3D) -> InteractionResult:
    """镜面反射：方向 ``V`` 关于法向 ``n`` 的反射。

    两个输入均须已单位化：``directions`` 为入射光线单位方向，``normals``
    为指向入射介质侧的单位法向。采用标准镜面公式 ``R = V - 2 (V·n) n``，
    并以带 eps 的归一化收尾，浮点漂移不会破坏单位长度。
    反射不判死光线（verdict 全活）。

    Args:
        directions: ``(P, F, W, N, 3)`` 入射单位方向。
        normals:    ``(P, F, W, N, 3)`` 单位法向，指向入射介质侧。
    """

    cosine = directions.mul(normals).sum(dim=-1, keepdim=True)  # cos θ_i, (..., 1)
    specular = directions.sub(cosine.mul(2.0).mul(normals))  # R = V - 2(V·n)n
    return InteractionResult(
        directions=F.normalize(specular, dim=-1, eps=1e-12),
        verdict=Verdict.alive_like(directions[..., 0]),
    )
