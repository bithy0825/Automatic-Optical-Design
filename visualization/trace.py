"""2D 视图追迹：光路图（x-z 截面）与点列图（传感器落点）的数据提取。

约定：光路图在 x-z 截面（光轴 +z 向右，横向 x 向上——fov 配置惯例为 x 向
变化），坐标一律 mm；点列为传感器局部 (x, y)。本层只产出张量数据，
JSON 打包在 :mod:`visualization.server`。
"""

from dataclasses import dataclass
from typing import Final

import torch

from component import InfiniteSource, Refractor, Sensor, Sequential, Stop
from core import TraceFlow, Transformer, sturdy_div
from implicit import FieldResult
from sampling import SampleOptions
from shape import Shape

N_PROFILE_PTS: Final = 200


@dataclass(slots=True)
class LayoutData:
    """光路图全种群数据（torch，未切片）。"""

    labels: list[str]  # 面标签 ["S1", ..., "Sensor"]，长度 S
    kinds: list[str]  # 面种类 ["sphere", ..., "sensor"]
    regions: list[list[str]]  # (S+1) × (P,)：regions[j] = 面 j-1 下游介质名
    profiles: torch.Tensor  # (P, S, N_PROFILE_PTS, 2)，(x, z) 全局，域外 NaN
    rims: torch.Tensor  # (P, S, 2, 2)，[下边缘, 上边缘]
    paths: torch.Tensor  # (P, F, W, N, S+1, 2)，(x, z)；第 0 步为发射面
    holds: torch.Tensor  # (P, F, W, N, S+1) bool
    effl: torch.Tensor  # (P,) 存活光线最小二乘估计焦距
    fields_deg: torch.Tensor  # (F, 2)
    wavelengths_nm: torch.Tensor  # (W,)


@dataclass(slots=True)
class SpotData:
    """点列图全种群数据（torch，未切片）。"""

    spots: torch.Tensor  # (P, F, W, N, 2) 传感器局部 xy；N 第 0 点为主光线
    holds: torch.Tensor  # (P, F, W, N) bool
    fields_deg: torch.Tensor  # (F, 2)
    wavelengths_nm: torch.Tensor  # (W,)


def viz_clone(seq: Sequential, pupil_cfg: SampleOptions) -> Sequential:
    """以 *pupil_cfg* 替换光源光瞳采样，克隆整条链（不碰原系统）。"""
    src = seq[0]
    if not isinstance(src, InfiniteSource):
        raise TypeError(f"first component must be an InfiniteSource, got {type(src).__name__}")
    viz_src = InfiniteSource(
        epd=src.epd,
        field_x=src.field_x,
        field_y=src.field_y,
        wavelength=src.wavelengths,
        population=src.population,
        pupil_cfg=pupil_cfg,
        field_cfg=src.field_cfg,
        wavel_cfg=src.wavel_cfg,
        transmitted=src.transmitted.clone(),
    )
    viz = Sequential(viz_src, *(comp.clone() for comp in seq[1:]))
    viz.rebind()
    return viz


def _profiles(shape: Shape, transformer: Transformer) -> tuple[torch.Tensor, torch.Tensor]:
    """逐种群面的 x-z 截面 profile 与边缘点。

    Returns:
        prof: ``(P, N_PROFILE_PTS, 2)`` —— (x, z) 全局坐标，域外 NaN。
        rims: ``(P, 2, 2)`` —— [下边缘点, 上边缘点]（首/尾有效采样点）。
    """
    D = shape.Diameter  # (P,)
    P, device, dtype = D.shape[0], D.device, D.dtype
    u = torch.linspace(-0.5, 0.5, N_PROFILE_PTS, device=device, dtype=dtype)
    xs = u.unsqueeze(0) * D.unsqueeze(1)  # (P, NP)
    pts2d = torch.stack((xs, torch.zeros_like(xs)), dim=-1).view(P, 1, 1, N_PROFILE_PTS, 2)
    res = shape.sag()(pts2d, order=FieldResult.Order.VALUE)
    z = res.value[:, 0, 0]  # (P, NP)
    ok = res.verdict.hold[:, 0, 0]  # (P, NP)
    local = torch.stack((xs, torch.zeros_like(xs), z), dim=-1)
    glob = transformer.transform_points(local)  # (P, NP, 3)
    prof = glob[..., [0, 2]].masked_fill(~ok.unsqueeze(-1), float("nan"))
    idx = torch.arange(N_PROFILE_PTS, device=device)
    first = torch.where(ok, idx, N_PROFILE_PTS).amin(dim=1).clamp(max=N_PROFILE_PTS - 1)
    last = torch.where(ok, idx, -1).amax(dim=1).clamp(min=0)
    pop = torch.arange(P, device=device)
    rims = torch.stack((prof[pop, first], prof[pop, last]), dim=1)  # (P, 2, 2)
    rims = rims.masked_fill((~ok.any(dim=1)).view(P, 1, 1), float("nan"))
    return prof, rims


def trace_layout(seq: Sequential, n_rays: int) -> LayoutData:
    """全种群光路图追迹：扇形光瞳 (n_rays, 1)，记录逐面交点、profile、材料链。"""
    if n_rays < 2:
        raise ValueError(f"n_rays must be >= 2, got {n_rays}")
    P = seq[0].population
    viz = viz_clone(seq, SampleOptions(method="uniform", region="rect", count=(n_rays, 1)))

    step_pts: list[torch.Tensor] = []
    step_hold: list[torch.Tensor] = []
    profiles: list[torch.Tensor] = []
    rims: list[torch.Tensor] = []
    labels: list[str] = []
    kinds: list[str] = []
    regions: list[list[str]] = [seq[0].transmitted.names()]

    def _cb(comp, flow: TraceFlow, _i: int) -> TraceFlow:
        if isinstance(comp, (InfiniteSource, Refractor, Stop, Sensor)):
            step_pts.append(flow.rays.points.detach())
            step_hold.append(flow.verdict.hold.detach())
        if isinstance(comp, (Refractor, Stop, Sensor)):
            prof, rim = _profiles(comp.shape, flow.transformer)
            profiles.append(prof)
            rims.append(rim)
            if isinstance(comp, Refractor):
                labels.append(f"S{len(labels) + 1}")
                kinds.append(str(comp.shape.kind.canonical))
                regions.append(comp.transmitted.names())
            elif isinstance(comp, Stop):
                labels.append("Stop")
                kinds.append("stop")
                regions.append(regions[-1])  # 光阑不改介质：下游 = 上游
            else:
                labels.append("Sensor")
                kinds.append("sensor")
                regions.append([""] * P)
        return flow

    flow = _trace_no_grad(viz, _cb)

    paths = torch.stack(step_pts, dim=-2)[..., [0, 2]]  # (P,F,W,N,S+1,2) 取 (x,z)
    holds = torch.stack(step_hold, dim=-1)
    # EFFL：存活光线"像高 ~ 视场角正切"的最小二乘斜率（与 effl_loss 同一估计量）
    t = flow.rays.field.tan()
    h = flow.rays.points[..., :2]
    w = flow.verdict.hold.unsqueeze(-1)
    effl = sturdy_div(
        w.mul(h).mul(t).sum(dim=(1, 2, 3, 4)),
        w.mul(t.square()).sum(dim=(1, 2, 3, 4)),
    )
    return LayoutData(
        labels=labels,
        kinds=kinds,
        regions=regions,
        profiles=torch.stack(profiles, dim=1),
        rims=torch.stack(rims, dim=1),
        paths=paths,
        holds=holds,
        effl=effl,
        fields_deg=torch.rad2deg(flow.rays.field[0, :, 0, 0, :]),
        wavelengths_nm=flow.rays.wavelength[0, 0, :, 0],
    )


def trace_spot(seq: Sequential, density: int, sampling: str = "uniform") -> SpotData:
    """全种群点列追迹：disk 光瞳，采样第 0 点 = 光瞳中心（主光线）。

    Args:
        seq: 光学系统。
        density: 密度档位。uniform → disk ``(density, 2·density)``；
            fibonacci → 等点数 disk（N = 2·d² − 2·d + 1，两种采样点数一致）。
        sampling: ``"uniform"``（同心环）或 ``"fibonacci"``（黄金角螺旋）。
    """
    if density < 2:
        raise ValueError(f"density must be >= 2, got {density}")
    if sampling == "uniform":
        pupil = SampleOptions(method="uniform", region="disk", count=(density, 2 * density))
    elif sampling == "fibonacci":
        pupil = SampleOptions(
            method="fibonacci", region="disk", count=2 * density * density - 2 * density + 1
        )
    else:
        raise ValueError(f"unknown sampling {sampling!r}")
    viz = viz_clone(seq, pupil)

    captured: dict[str, torch.Tensor] = {}

    def _cb(comp, flow: TraceFlow, _i: int) -> TraceFlow:
        if isinstance(comp, Sensor):
            local = flow.transformer.transform_points(flow.rays.points, inverse=True)
            captured["spots"] = local[..., :2].detach()
        return flow

    flow = _trace_no_grad(viz, _cb)
    if "spots" not in captured:
        raise RuntimeError("no Sensor in the system")

    return SpotData(
        spots=captured["spots"],
        holds=flow.verdict.hold.detach(),
        fields_deg=torch.rad2deg(flow.rays.field[0, :, 0, 0, :]),
        wavelengths_nm=flow.rays.wavelength[0, 0, :, 0],
    )


def probe_illumination(seq: Sequential) -> tuple[torch.Tensor, torch.Tensor]:
    """光源单发探针：返回 (fields_deg ``(F,2)``, wavelengths_nm ``(W,)``)。"""
    with torch.no_grad():
        flow = seq[0].forward()
    return (
        torch.rad2deg(flow.rays.field[0, :, 0, 0, :]),
        flow.rays.wavelength[0, 0, :, 0],
    )


def _trace_no_grad(viz: Sequential, cb) -> TraceFlow:
    """无梯度追迹；光源初始位姿按进程缺省 dtype 创建，期间临时切到链 dtype。"""
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(viz.dtype)
    try:
        with torch.no_grad():
            return viz(callback=cb)
    finally:
        torch.set_default_dtype(prev_dtype)
