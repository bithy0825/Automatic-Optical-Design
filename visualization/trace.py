"""可视化追迹管线:按视图采样参数对全种群批量追迹,结果缓存后按 pop 切片。

两种视图:
* 布局 —— 扇形光瞳 (n_rays, 1),记录逐面交点路径、面 profile、材料链。
* 点列 —— disk 光瞳 (density, 2*density),记录传感器局部坐标(第 0 点为主光线)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch

from component import InfiniteSource, Refractor, Sensor, Sequential
from component.protocol import Component
from core import TraceFlow, Transformer
from implicit.protocol import FieldResult
from sampling import SampleOptions
from shape import Shape
from visualization.protocol import pack

N_PROFILE_PTS: Final = 200


# ═══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class LayoutData:
    """全种群布局数据(缓存单元,已 numpy 化)。"""

    labels: list[str]            # 面标签 ["S1", ..., "Sensor"],长度 S
    kinds: list[str]             # 面种类 ["Sphere", ..., "sensor"]
    regions: list[list[str]]     # (S+1) × (P,):regions[j] = 面 j-1 下游介质名
    profiles: np.ndarray         # (P, S, N_PROFILE_PTS, 2) f32,(x, z) 全局,域外 NaN
    rims: np.ndarray             # (P, S, 2, 2) f32,[下边缘, 上边缘]
    paths: np.ndarray            # (P, F, W, N, S+1, 3) f32,第 0 步为发射面
    holds: np.ndarray            # (P, F, W, N, S+1) u8
    fields_deg: np.ndarray       # (F, 2) f32
    wavelengths_nm: np.ndarray   # (W,) f32

    @property
    def population(self) -> int:
        return int(self.paths.shape[0])


@dataclass(slots=True)
class SpotData:
    """全种群点列数据(缓存单元,已 numpy 化)。"""

    spots: np.ndarray            # (P, F, W, N, 2) f32,传感器局部 xy;N 第 0 点为主光线
    holds: np.ndarray            # (P, F, W, N) u8
    fields_deg: np.ndarray       # (F, 2) f32
    wavelengths_nm: np.ndarray   # (W,) f32

    @property
    def population(self) -> int:
        return int(self.spots.shape[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 构建与追迹
# ═══════════════════════════════════════════════════════════════════════════════


def _viz_seq(seq: Sequential, pupil_cfg: SampleOptions) -> Sequential:
    """以 *pupil_cfg* 替换光源光瞳采样,克隆整条链(不碰原系统)。"""
    src: InfiniteSource = seq[0]
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
    """逐种群面的 xz 截面 profile 与边缘点。

    Returns:
        prof: (P, N_PROFILE_PTS, 2) — (x, z) 全局坐标,域外 NaN。
        rims: (P, 2, 2) — [下边缘点, 上边缘点](首/尾有效采样点)。
    """
    D = shape.Diameter  # (P,)
    P, device, dtype = D.shape[0], D.device, D.dtype
    u = torch.linspace(-0.5, 0.5, N_PROFILE_PTS, device=device, dtype=dtype)
    xs = u.unsqueeze(0) * D.unsqueeze(1)  # (P, NP)
    pts2d = torch.stack((xs, torch.zeros_like(xs)), dim=-1).view(P, 1, 1, N_PROFILE_PTS, 2)
    res = shape.sag()(pts2d, order=FieldResult.Order.VALUE)
    z = res.value[:, 0, 0]          # (P, NP)
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
    """全种群布局追迹:扇形光瞳 (n_rays, 1),记录逐面交点、profile、材料链。"""
    if n_rays < 2:
        raise ValueError(f"n_rays must be >= 2, got {n_rays}")
    P = seq[0].population
    viz = _viz_seq(seq, SampleOptions(method="uniform", region="rect", count=(n_rays, 1)))

    step_pts: list[torch.Tensor] = []
    step_hold: list[torch.Tensor] = []
    profiles: list[torch.Tensor] = []
    rims: list[torch.Tensor] = []
    labels: list[str] = []
    kinds: list[str] = []
    regions: list[list[str]] = [seq[0].transmitted.names()]

    def _cb(comp: Component, flow: TraceFlow, _i: int) -> TraceFlow:
        if isinstance(comp, (InfiniteSource, Refractor, Sensor)):
            step_pts.append(flow.rays.points.detach())
            step_hold.append(flow.verdict.hold.detach())
        if isinstance(comp, (Refractor, Sensor)):
            prof, rim = _profiles(comp.shape, flow.transformer)
            profiles.append(prof)
            rims.append(rim)
            if isinstance(comp, Refractor):
                labels.append(f"S{len(labels) + 1}")
                kinds.append(str(comp.shape.kind.canonical))
                regions.append(comp.transmitted.names())
            else:
                labels.append("Sensor")
                kinds.append("sensor")
                regions.append([""] * P)
        return flow

    with torch.no_grad():
        flow = viz.forward(callback=_cb)

    paths = torch.stack(step_pts, dim=-2).cpu().numpy().astype(np.float32)
    holds = torch.stack(step_hold, dim=-1).cpu().numpy().astype(np.uint8)
    profs = torch.stack(profiles, dim=1).cpu().numpy().astype(np.float32)
    rim = torch.stack(rims, dim=1).cpu().numpy().astype(np.float32)
    fields_deg = torch.rad2deg(flow.rays.field[0, :, 0, 0, :]).cpu().numpy().astype(np.float32)
    wls = flow.rays.wavelength[0, 0, :, 0].cpu().numpy().astype(np.float32)
    return LayoutData(
        labels=labels, kinds=kinds, regions=regions,
        profiles=profs, rims=rim, paths=paths, holds=holds,
        fields_deg=fields_deg, wavelengths_nm=wls,
    )


def probe_illumination(seq: Sequential) -> tuple[np.ndarray, np.ndarray]:
    """光源单发探针:返回 (fields_deg (F,2) f32, wavelengths_nm (W,) f32)。"""
    with torch.no_grad():
        flow = seq[0].forward()
    fields = torch.rad2deg(flow.rays.field[0, :, 0, 0, :]).cpu().numpy().astype(np.float32)
    wls = flow.rays.wavelength[0, 0, :, 0].cpu().numpy().astype(np.float32)
    return fields, wls


def trace_spot(seq: Sequential, density: int, sampling: str = "uniform") -> SpotData:
    """全种群点列追迹:disk 光瞳,采样第 0 点 = 光瞳中心(主光线)。

    Args:
        seq: 光学系统。
        density: 密度档位。uniform → disk (density, 2*density);
            fibonacci → 等点数 disk(N = 2·d² − 2·d + 1,两种采样点数一致)。
        sampling: ``"uniform"``(同心环)或 ``"fibonacci"``(黄金角螺旋)。
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
    viz = _viz_seq(seq, pupil)

    captured: dict[str, torch.Tensor] = {}

    def _cb(comp: Component, flow: TraceFlow, _i: int) -> TraceFlow:
        if isinstance(comp, Sensor):
            local = flow.transformer.transform_points(flow.rays.points, inverse=True)
            captured["spots"] = local[..., :2].detach()
        return flow

    with torch.no_grad():
        flow = viz.forward(callback=_cb)
    if "spots" not in captured:
        raise RuntimeError("trace_spot: 系统中没有 Sensor 元件")

    spots = captured["spots"].cpu().numpy().astype(np.float32)
    holds = flow.verdict.hold.detach().cpu().numpy().astype(np.uint8)
    fields_deg = torch.rad2deg(flow.rays.field[0, :, 0, 0, :]).cpu().numpy().astype(np.float32)
    wls = flow.rays.wavelength[0, 0, :, 0].cpu().numpy().astype(np.float32)
    return SpotData(spots=spots, holds=holds, fields_deg=fields_deg, wavelengths_nm=wls)


class PopOutOfRange(IndexError):
    """pop 越界(服务端映射为 400)。"""


def _check_pop(pop: int, population: int) -> None:
    if not 0 <= pop < population:
        raise PopOutOfRange(f"pop {pop} out of range [0, {population})")


class TraceCache:
    """按视图参数缓存全种群追迹;打包时按 pop 切片。"""

    def __init__(self, seq: Sequential) -> None:
        self._seq = seq
        self._layout: dict[int, LayoutData] = {}
        self._spot: dict[tuple[int, str], SpotData] = {}

    def layout(self, n_rays: int) -> LayoutData:
        if n_rays not in self._layout:
            self._layout[n_rays] = trace_layout(self._seq, n_rays)
        return self._layout[n_rays]

    def spot(self, density: int, sampling: str = "uniform") -> SpotData:
        key = (density, sampling)
        if key not in self._spot:
            self._spot[key] = trace_spot(self._seq, density, sampling)
        return self._spot[key]

    def layout_packet(self, pop: int, n_rays: int) -> bytes:
        data = self.layout(n_rays)
        _check_pop(pop, data.population)
        meta = {
            "labels": data.labels,
            "kinds": data.kinds,
            "regions": [r[pop] for r in data.regions],
            "fields_deg": data.fields_deg.tolist(),
            "wavelengths_nm": data.wavelengths_nm.tolist(),
        }
        return pack(meta, {
            "profiles": data.profiles[pop],
            "rims": data.rims[pop],
            "paths": data.paths[pop],
            "holds": data.holds[pop],
        })

    def spot_packet(self, pop: int, density: int, sampling: str = "uniform") -> bytes:
        data = self.spot(density, sampling)
        _check_pop(pop, data.population)
        meta = {
            "chief_index": 0,
            "fields_deg": data.fields_deg.tolist(),
            "wavelengths_nm": data.wavelengths_nm.tolist(),
        }
        return pack(meta, {"spots": data.spots[pop], "holds": data.holds[pop]})
