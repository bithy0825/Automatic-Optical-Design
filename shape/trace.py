from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from core import (
    RayFloat3D,
    RayFloatScalar,
    SystemFloatScalar,
    Transformer,
    Verdict,
    broadcast_system_to_ray,
)
from core.container import TensorContainer
from implicit import FieldResult, ImplicitFunction, SolverFunction


@dataclass(slots=True, eq=False, repr=False)
class TraceResult(TensorContainer):
    """光线与单一曲面相交的结果（``Shape.forward`` 的产物）。

    Attributes:
        distances: 光线参数 *t*（全局系），命中点 = ``points + t·directions``。
        normals: 全局系单位法向，指向入射介质侧（``normals · directions ≤ 0``）。
        points_global: 全局系下的命中点。
        verdict: 追迹裁决——solver 的 ``shape.at(negative).at(convergence)``
            再链入机械孔径裁决（几何/数值死亡优先于孔径）。
    """

    distances: RayFloatScalar
    normals: RayFloat3D
    points_global: RayFloat3D
    verdict: Verdict


class ApertureFunction(Protocol):
    """机械孔径裁决。

    在曲面**局部坐标系**下评估（sag 域为 (x,y) 平面、光轴为 z），
    返回站点 ``Verdict``：界内 ``hold=True``，越界 ``hold=False``
    （toll 有符号，正 = 界内余量，见 :class:`~core.verdict.Verdict` 契约）。
    """

    def __call__(self, points_local: RayFloat3D) -> Verdict: ...


def no_aperture() -> ApertureFunction:
    """无孔径：恒全活。"""

    def _no_aperture(points_local: RayFloat3D) -> Verdict:
        return Verdict.alive_like(points_local[..., 0])

    return _no_aperture


def circle_aperture(radius: SystemFloatScalar) -> ApertureFunction:
    """圆形机械孔径：``r² ≤ R²`` 界内，toll = ``R² − r²``。"""

    def _circle_aperture(points_local: RayFloat3D) -> Verdict:
        x, y, _ = points_local.unbind(dim=-1)
        r2 = x.square().add(y.square())
        lim = broadcast_system_to_ray(radius, r2).square()
        return Verdict.site(
            hold=r2.le(lim), toll=lim.sub(r2), cause=Verdict.Cause.APERTURE_CLIP
        )

    return _circle_aperture


def intersect(
    points: RayFloat3D,
    directions: RayFloat3D,
    transformer: Transformer,
    implicit: ImplicitFunction,
    solver_fn: SolverFunction,
    aperture_fn: ApertureFunction,
) -> TraceResult:
    """求光线 ``(points, directions)`` 与曲面的交点，装配 :class:`TraceResult`。

    曲面过 ``transformer`` 的局部原点（顶点在 z=0，光轴沿 +z）。步骤：
    全局→局部变换 → solver 求交（内部已链 ``shape.at(negative).at(convergence)``
    裁决）→ 命中点法向定向（反向入射光）→ 机械孔径裁决链入 → 全局法向归一化。

    Args:
        points, directions: 全局系光线起点与单位方向，``(P, F, W, N, 3)``。
        transformer: 面的世界位姿（刚体：旋转 + 平移，无缩放）；曲面过其局部原点。
        implicit: 提升后的 3D 隐式函数（``lift_raw(sag)``）。
        solver_fn: 求解器（``solve(options)``），返回 distances/value/verdict。
        aperture_fn: 机械孔径裁决（``circle_aperture(radius)`` 或 ``no_aperture()``）。
    """
    # 1. 全局 → 局部（曲面过 transformer 原点，在局部系下求交）。
    points_loc = transformer.transform_points(points, inverse=True)
    directions_loc = transformer.transform_vectors(directions, inverse=True)

    # 2. solver 求交：内部已链 shape.at(negative).at(convergence) 裁决。
    res = solver_fn(points_loc, directions_loc, implicit)
    distances = res.distances

    # 3. 命中点 + 局部法向（一阶隐式梯度即曲面法向）。
    hit_loc = points_loc.add(directions_loc.mul(distances.unsqueeze(-1)))
    field = implicit(hit_loc, order=FieldResult.Order.GRADIENT)
    n_loc = F.normalize(field.gradient, dim=-1, eps=1e-12)

    # 4. 定向：法向反向入射光（normals · directions ≤ 0），使其指向入射介质侧。
    dot = n_loc.mul(directions_loc).sum(dim=-1, keepdim=True)
    n_loc = torch.where(dot.gt(0.0), n_loc.neg(), n_loc)

    # 5. 命中点回到全局系。transformer 为刚体位姿（无缩放），t 在全局/局部一致，
    #    故用全局 directions 推进。
    hit_global = points.add(directions.mul(distances.unsqueeze(-1)))

    # 6. 法向回到全局系并归一化——防御性消除变换可能引入的数值误差，
    #    保证 |normal| = 1。
    normals = F.normalize(transformer.transform_vectors(n_loc), dim=-1, eps=1e-12)

    # 7. 机械孔径裁决最后链入：几何/数值死亡（solver）优先于孔径越界。
    verdict = res.verdict.at(aperture_fn(hit_loc))

    return TraceResult(
        distances=distances,
        normals=normals,
        points_global=hit_global,
        verdict=verdict,
    )
