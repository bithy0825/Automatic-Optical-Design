"""误差注入：把误差声明编译成单个 :data:`~component.sequential.FlowCallback`。

编译期做一次无扰动探测追迹，记录每个站点（每个元件之后）的名义位姿；
运行期在命中站点把 ``flow.transformer`` **绝对复位**为 ``nominal @ Δ``——
注入与恢复是同一条路径：位姿误差作用区间外的站点直接回到名义位姿，
区间嵌套天然正确，无"相对恢复"的共轭陷阱。名义探测在 ``no_grad`` 下进行；
扰动张量自身若带 ``requires_grad``，灵敏度照常反传。

回调不碰系统任何参数（无需 clone、不干扰 GA），扰动只活在当次 forward 内；
Transformer 逐个体成批，种群维即蒙特卡洛维——一次追迹 = 一整批扰动实例。
"""

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, cast

import torch
import torch.nn.functional as F

from component import FlowCallback, Refractor, Sensor, Sequential, Stop
from core import TraceFlow, Transformer, Verdict, broadcast_system_to_ray, sturdy_div
from perturb.error import Clip, Pose, Scatter, _materialize
from perturb.error import from_options as _parse_error

type ErrorLike = Pose | Scatter | Clip | Mapping[str, Any]


def _surface_indices(seq: Sequential) -> list[int]:
    """光学面号（1 起）→ 元件索引：第 n 个 Refractor/Stop/Sensor，Gap 不计。"""
    return [i for i, c in enumerate(seq) if isinstance(c, (Refractor, Stop, Sensor))]


def _nominal_poses(seq: Sequential) -> list[Transformer]:
    """无扰动探测追迹：逐站点名义位姿（dtype 钉扎：InfiniteSource 无浮点
    buffer，其初始 Transformer 取进程缺省 dtype，探测期间临时对齐 seq）。"""
    poses: list[Transformer] = []

    def _cb(_comp: object, flow: TraceFlow, _i: int) -> TraceFlow:
        poses.append(flow.transformer)
        return flow

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(seq.dtype)
    try:
        with torch.no_grad():
            seq(callback=_cb)
    finally:
        torch.set_default_dtype(prev_dtype)
    return poses


def _delta(
    spec: Pose, population: int, *, device: torch.device, dtype: torch.dtype
) -> Transformer | None:
    """Δ = T(dx,dy,dz)·Rx(α)·Ry(β) 逐个体批量构造；字段全 None（恒等）返回 None。"""

    def mat(s) -> torch.Tensor:
        return _materialize(s, population, device=device, dtype=dtype)

    delta: Transformer | None = None
    if (spec.dx, spec.dy, spec.dz) != (None, None, None):
        d = torch.stack((mat(spec.dx), mat(spec.dy), mat(spec.dz)), dim=-1)
        delta = Transformer.translation(d)
    for angle, ax_i in ((spec.alpha, 0), (spec.beta, 1)):
        if angle is None:
            continue
        axis = torch.zeros(population, 3, device=device, dtype=dtype)
        axis[:, ax_i] = 1.0
        rot = Transformer.rotation(axis, mat(angle))
        delta = rot if delta is None else delta.then(rot)
    return delta


def _scatter(flow: TraceFlow, std: torch.Tensor) -> TraceFlow:
    """折射后方向加各向同性角噪声并重归一；死光线方向冻结。

    主光线（N=0，disk 采样器保证第 0 点为光瞳中心）同样冻结：它是下游
    参考球/网格中心的定义基准，单次随机抽样的主光线抖动是参考系 artifact
    而非物理（每波长各抽一次噪声时尤其扎眼）。
    """
    rays = flow.rays
    sig = broadcast_system_to_ray(std, rays.wavelength).unsqueeze(-1)
    noise = torch.randn_like(rays.directions).mul_(sig)
    noise[..., 0, :] = 0.0
    d = F.normalize(rays.directions.add(noise), dim=-1)
    d = torch.where(flow.verdict.hold.unsqueeze(-1), d, rays.directions)
    return flow.with_rays(replace(rays, directions=d))


def _clip(flow: TraceFlow, z_plane: torch.Tensor, radius: torch.Tensor) -> Verdict:
    """把光线直线推进到站点名义 z 平面，横向半径超限判死（APERTURE_CLIP）。"""
    rays = flow.rays
    z = broadcast_system_to_ray(z_plane, rays.wavelength)
    t = sturdy_div(z - rays.points[..., 2], rays.directions[..., 2])
    p = rays.points + rays.directions * t.unsqueeze(-1)
    r = p[..., :2].square().sum(dim=-1).sqrt()
    R = broadcast_system_to_ray(radius, r)
    return Verdict.site(hold=r <= R, toll=R - r, cause=Verdict.Cause.APERTURE_CLIP)


def build_callback(seq: Sequential, errors: Iterable[ErrorLike]) -> FlowCallback:
    """把一批误差声明编译成一个 FlowCallback，供 ``seq(callback=...)`` 使用。

    Args:
        seq: 光学系统（提供 population / device / dtype 与站点结构）。
        errors: 误差声明序列（:class:`Pose` / :class:`Scatter` / :class:`Clip`
            实例或配置映射，映射按 ``type`` 键分派）。空序列返回恒等回调。
            分布映射在此刻采样一次；重采 = 重新 build。

    面号为 1 起的光学面序号（见 :mod:`perturb.error`）。多个 Pose 区间
    重叠时，同一站点上的 Δ 按声明顺序依次右乘。
    """
    blocks = cast(Iterable[ErrorLike], (errors,) if isinstance(errors, Mapping) else errors)
    errs = tuple(e if isinstance(e, (Pose, Scatter, Clip)) else _parse_error(e) for e in blocks)
    if not errs:
        return lambda _comp, flow, _i: flow

    P, device, dtype = seq.population, seq.device, seq.dtype
    surf = _surface_indices(seq)
    n_surf = len(surf)

    def _check_surface(k: int) -> int:
        if not 1 <= k <= n_surf:
            raise ValueError(f"surface {k} out of range [1, {n_surf}]")
        return surf[k - 1]

    pose_items: list[tuple[int, int | None, Transformer]] = []  # (首元件, 末元件|None, Δ)
    scatter_at: dict[int, torch.Tensor] = {}  # 站点 → 角噪声 std (P,)
    clip_at: list[tuple[int | None, torch.Tensor]] = []  # (站点|None, 半径 (P,))
    for e in errs:
        if isinstance(e, Pose):
            first_c = _check_surface(e.first)
            last_c = None if e.last is None else _check_surface(e.last)
            if last_c is not None and last_c < first_c:
                raise ValueError(f"pose interval [{e.first}, {e.last}] is empty")
            d = _delta(e, P, device=device, dtype=dtype)
            if d is not None:
                pose_items.append((first_c, last_c, d))
        elif isinstance(e, Scatter):
            scatter_at[_check_surface(e.surface)] = _materialize(
                e.std, P, device=device, dtype=dtype
            )
        else:
            at = None if e.at is None else _check_surface(e.at)
            clip_at.append((at, _materialize(e.radius, P, device=device, dtype=dtype)))

    # 站点编译：站点 i（元件 i 之后）的位姿 = nominal[i] @ ∏{活跃 Δ}；
    # 活跃集与上一站不同才写入——区间结束即自动回到名义位姿。
    stations: dict[int, Transformer] = {}
    zs: list[torch.Tensor] = []
    if pose_items or clip_at:
        nominals = _nominal_poses(seq)
        zs = [t.forward[:, 2, 3] for t in nominals]
        active: tuple[int, ...] = ()
        for i in range(len(seq)):
            now = tuple(
                k
                for k, (f, last, _d) in enumerate(pose_items)
                if f - 1 <= i and (last is None or i < last)
            )
            if now != active:
                if now:
                    d = pose_items[now[0]][2]
                    for k in now[1:]:
                        d = d.then(pose_items[k][2])
                    stations[i] = nominals[i].then(d)
                else:
                    stations[i] = nominals[i]  # 区间结束 → 复位名义位姿
                active = now

    def _cb(_comp: object, flow: TraceFlow, i: int) -> TraceFlow:
        if i in stations:
            flow = flow.with_transformer(stations[i])
        std = scatter_at.get(i)
        if std is not None:
            flow = _scatter(flow, std)
        for at, radius in clip_at:
            if at is None or at == i:
                flow = flow.at_verdict(_clip(flow, zs[i], radius))
        return flow

    return _cb
