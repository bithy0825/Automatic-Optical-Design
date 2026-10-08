"""相机传感器仿真 PSF：密集波长 Kirchhoff 积分 + 光谱响应加权合成 RGB。

与 :mod:`analysis.psf` 同一物理内核（出瞳参考球 + 精确距离 Kirchhoff 核 +
倾斜因子），但自包含重实现，接口面向"一个结构的完整光谱像质"：

* 任意密集波长网格（不限于系统配置的少数设计波长）；
* 可选追迹回调（:mod:`perturb` 误差注入直接可用）；
* 逐波长 PSF 经传感器光谱响应（camspec 曲线，如 Nikon D200）加权合成 RGB
  通道——横向色差以亚像素平移保留：各单色 PSF 先双线性重采样到逐视场的
  公共参考中心（总响应加权均值），再加权求和。

约定：等能光源（平谱）、镜头透过率理想；白平衡 = 各通道按响应和归一
（Σ=1），即"平谱白 → 三通道等能量"。全程 float64、无梯度，评估在克隆链
上进行（材料单例随 ``.to`` 迁移的既定行为同 analysis/psf.py）。

camspec CSV 格式：``#`` 开头的注释头 + 数据行 ``波长nm,R,G,B``
（响应为任意单位的相对灵敏度）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from component import InfiniteSource, Refractor, Sensor, Sequential
from component.sequential import FlowCallback
from core import TraceFlow, Transformer
from sampling import SampleOptions

_NM_TO_MM: float = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 传感器光谱响应
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SensorResponse:
    """相机光谱响应：``wavelengths`` (W,) nm 升序；``rgb`` (W,3) 相对灵敏度。"""

    wavelengths: Tensor
    rgb: Tensor

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        """读取 camspec CSV（``波长,R,G,B``，``#`` 注释头）。"""
        data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
        if data.shape[1] != 4:
            raise ValueError(
                f"Expected 4 columns (wavelength,R,G,B), got {data.shape[1]}"
            )
        wl = torch.from_numpy(data[:, 0]).double()
        rgb = torch.from_numpy(data[:, 1:4]).double()
        order = torch.argsort(wl)
        return cls(wavelengths=wl[order], rgb=rgb[order])

    def at(self, wavelengths: Tensor) -> Tensor:
        """线性插值到给定波长 (W',) → (W',3)；域外取 0。dtype/device 跟随输入。"""
        query = wavelengths.detach().cpu().numpy()
        base_w = self.wavelengths.numpy()
        cols = [
            np.interp(query, base_w, self.rgb[:, c].numpy(), left=0.0, right=0.0)
            for c in range(3)
        ]
        return torch.from_numpy(np.stack(cols, axis=-1)).to(wavelengths)


# ═══════════════════════════════════════════════════════════════════════════════
# 追迹与参考球（物理内核，与 analysis/psf.py 同源；附加 callback 支持）
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _Trace:
    """追迹快照：末折射面状态 + 传感器数据 + 逐光线光程。"""

    points: Tensor  # (P,F,W,N,3) 末折射面命中点(全局)
    dirs: Tensor  # (P,F,W,N,3) 末折射面出射方向
    opl: Tensor  # (P,F,W,N) 快照处累积光程 mm
    n_img: Tensor  # (P,F,W,N) 像方折射率(逐光线波长)
    sensor_pts: Tensor  # (P,F,W,N,3) 传感器命中点(全局)
    sensor_tf: Transformer  # 传感器局部坐标系
    alive: Tensor  # (P,F,W,N) 传感器处最终存活
    wavelengths: Tensor  # (W,) nm


def _eval_seq(
    seq: Sequential, pupil: SampleOptions | None, wavelengths: tuple[float, ...] | None
) -> Sequential:
    """评估链：（可选）密集波长覆盖 + disk 光瞳覆盖克隆 + 整体 float64。

    波长覆盖时 ``wavel_cfg`` 重建为等数 uniform line，采样值与控制点一一
    对齐（``_interp_wavelength`` 的精确路径）；否则沿用原光源采样配置。
    """
    src = seq[0]
    if not isinstance(src, InfiniteSource):
        raise ValueError(f"首元件必须是 InfiniteSource, got {type(src).__name__}")
    cfg = (
        pupil
        if pupil is not None
        else SampleOptions(method="fibonacci", region="disk", count=20001)
    )
    if cfg.region != "disk" or cfg.method not in ("uniform", "fibonacci"):
        raise ValueError(
            "PSF 仅支持 uniform/fibonacci 的 disk 光瞳采样(第 0 点为光瞳中心),"
            f" got method={cfg.method!r} region={cfg.region!r}"
        )
    if wavelengths is None:
        wl, wavel_cfg = src.wavelengths, src.wavel_cfg
    else:
        wl = tuple(float(w) for w in wavelengths)
        wavel_cfg = SampleOptions(method="uniform", region="line", count=len(wl))
    eval_src = InfiniteSource(
        epd=src.epd,
        field_x=src.field_x,
        field_y=src.field_y,
        wavelength=wl,
        population=src.population,
        pupil_cfg=cfg,
        field_cfg=src.field_cfg,
        wavel_cfg=wavel_cfg,
        transmitted=src.transmitted.clone(),
    )
    out = Sequential(eval_src, *(comp.clone() for comp in seq[1:]))
    out.rebind()
    return out.to(torch.float64)


def _trace(seq: Sequential, extra: FlowCallback | None) -> _Trace:
    """追迹评估链：callback 逐段累积 OPL（几何段长 × 当前介质折射率）。

    段介质 = 上游最近 Source/Refractor 的出射材料（折射面处先按旧介质
    累积到达段、再更新介质）；Gap 不移动光线，Stop 在同介质内推进。
    *extra* 为误差注入等附加回调（先于 OPL 记账执行；记账只读当前状态）。

    InfiniteSource 无浮点 buffer，初始 Transformer 取进程缺省 dtype；
    追迹期间临时切换缺省为 float64（退出还原）。
    """
    if not any(isinstance(c, Refractor) for c in seq):
        raise ValueError("PSF 需要至少一个折射面(Refractor)")
    if not isinstance(seq[-1], Sensor):
        raise ValueError("末元件必须是 Sensor")

    prev: list[Tensor | None] = [None]
    opl: list[Tensor | None] = [None]
    n_cur: list[Tensor | None] = [None]
    snap: dict[str, Tensor] = {}
    term: dict[str, object] = {}

    def _cb(comp, flow: TraceFlow, _i: int) -> TraceFlow:
        pts = flow.rays.points
        if isinstance(comp, InfiniteSource):
            # 发射面:物方折射率 + 无穷远倾斜平面波相位参考(主光线为零)
            n_cur[0] = comp.transmitted(flow.rays.wavelength)
            opl[0] = (pts * flow.rays.directions).sum(dim=-1) * n_cur[0]
        else:
            # 到达段:几何长度 × 上游介质折射率(逐光线波长)
            assert opl[0] is not None and prev[0] is not None and n_cur[0] is not None
            opl[0] = opl[0] + (pts - prev[0]).norm(dim=-1) * n_cur[0]
        prev[0] = pts
        if isinstance(comp, Refractor):
            n_cur[0] = comp.transmitted(flow.rays.wavelength)
            snap.update(
                points=pts, dirs=flow.rays.directions, opl=opl[0].clone(), n=n_cur[0]
            )
        if isinstance(comp, Sensor):
            term.update(
                points=pts,
                tf=flow.transformer,
                hold=flow.verdict.hold,
                wl=flow.rays.wavelength[0, 0, :, 0],
            )
        return flow

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.no_grad():
            seq(callback=_cb if extra is None else (extra, _cb))
    finally:
        torch.set_default_dtype(prev_dtype)
    if not term:
        raise RuntimeError("系统中没有 Sensor 元件")
    return _Trace(
        points=snap["points"],
        dirs=snap["dirs"],
        opl=snap["opl"],
        n_img=snap["n"],
        sensor_pts=cast(Tensor, term["points"]),
        sensor_tf=cast(Transformer, term["tf"]),
        alive=cast(Tensor, term["hold"]),
        wavelengths=cast(Tensor, term["wl"]),
    )


@dataclass(slots=True)
class _Sphere:
    """参考球：球面点、内法向、球面 OPD、有效掩码、球心与半径。"""

    points: Tensor  # (P,F,W,N,3) 球面交点(全局)
    normals: Tensor  # (P,F,W,N,3) 指向球心的单位内法向
    opd: Tensor  # (P,F,W,N) 球面光程差 mm(主光线为零点,无效 nan)
    valid: Tensor  # (P,F,W,N) 参与积分掩码
    centers_g: Tensor  # (P,F,W,3) 球心 = 主光线像点(全局)
    radius: Tensor  # (P,F,W) 球半径 mm


def _reference_sphere(tr: _Trace) -> _Sphere:
    """以主光线像点为球心、出瞳估计距离为半径，光线从末折射面解析反投影。

    出瞳 z = 末面后主光线与光轴的最近交点；病态（轴上视场）回退同 (p,w)
    其他视场中位数，全部病态再回退主光线末面点。精确距离核下 R 的小误差
    只影响采样经济性，不影响核的正确性。
    """
    chief_alive = tr.alive[..., 0]  # (P,F,W)
    w = tr.alive.unsqueeze(-1).to(tr.sensor_pts.dtype)
    centroid = (tr.sensor_pts * w).sum(dim=-2) / w.sum(dim=-2).clamp_min(1.0)
    # 主光线死亡 → 存活质心兜底(仅影响中心定义;活塞相位不影响强度)
    centers_g = torch.where(
        chief_alive.unsqueeze(-1), tr.sensor_pts[..., 0, :], centroid
    )

    c0, d0 = tr.points[..., 0, :], tr.dirs[..., 0, :]  # (P,F,W,3) 主光线
    dxy2 = d0[..., :2].square().sum(dim=-1)
    ok = dxy2 > 1e-12
    t_star = -(c0[..., :2] * d0[..., :2]).sum(dim=-1) / dxy2.clamp_min(1e-300)
    z_cross = c0[..., 2] + t_star * d0[..., 2]
    z_sel = torch.where(ok, z_cross, torch.full_like(z_cross, float("nan")))
    med = z_sel.nanmedian(dim=1, keepdim=True).values  # (P,1,W) 视场维中位数
    z_ep = torch.where(ok, z_cross, med.expand_as(z_cross))
    z_ep = torch.where(z_ep.isnan(), c0[..., 2], z_ep)  # 全病态:主光线末面点
    ep = torch.stack((torch.zeros_like(z_ep), torch.zeros_like(z_ep), z_ep), dim=-1)
    radius = (centers_g - ep).norm(dim=-1).clamp_min(1e-9)

    # 光线-球解析求交:|X + t·d − C|² = R²,取前方根(朝球心会聚的一侧)
    m = tr.points - centers_g.unsqueeze(-2)  # (P,F,W,N,3)
    b = (tr.dirs * m).sum(dim=-1)  # (P,F,W,N)
    disc = b.square() - m.square().sum(dim=-1) + radius.unsqueeze(-1).square()
    hit = disc > 0
    t = -b - disc.clamp_min(0).sqrt()
    points = tr.points + t.unsqueeze(-1) * tr.dirs
    opl = tr.opl + t * tr.n_img
    opd = opl - opl[..., :1]  # 主光线(N=0)为零点
    normals = (centers_g.unsqueeze(-2) - points) / radius.unsqueeze(-1).unsqueeze(-1)
    valid = tr.alive & hit
    opd = torch.where(valid, opd, torch.full_like(opd, float("nan")))
    return _Sphere(points, normals, opd, valid, centers_g, radius)


# ═══════════════════════════════════════════════════════════════════════════════
# 单色 PSF
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class MonoPsf:
    """逐波长单色 PSF（每张 Σ=1）与诊断。所有张量 float64，位于 seq.device。"""

    psf: Tensor  # (P,F,W,H,H);psf[...,i,j] ↔ (x_i, y_j)
    delta: float  # 像面采样间隔 mm/px
    centers: Tensor  # (P,F,W,2) 网格中心(传感器局部 xy, mm)
    wavelengths: Tensor  # (W,) nm
    na: Tensor  # (P,F,W) 像方数值孔径
    alive: Tensor  # (P,F,W,N) 参与积分的光线掩码
    warnings: list[str]


def mono_psf(
    seq: Sequential,
    image_sampling: int,
    *,
    wavelengths: tuple[float, ...] | None = None,
    image_delta: float | None = None,
    pupil: SampleOptions | None = None,
    chunk: int = 512,
    callback: FlowCallback | None = None,
) -> MonoPsf:
    """计算系统全部 (P,F,W) 的 Kirchhoff 单色 PSF（网格以主光线像点为中心）。

    Args:
        seq:            光学系统（不被修改；内部克隆评估并转 float64）。
        image_sampling: PSF 网格边长 H（H×H）。
        wavelengths:    波长网格覆盖 (nm)；None → 系统配置的采样。
        image_delta:    像面采样间隔 mm/px；None → 自动 λ_min/(4·NA_max)。
        pupil:          光瞳采样覆盖；仅 uniform/fibonacci disk；缺省 fibonacci 20001。
        chunk:          积分光线分块大小（控制单步内存 ~H²·chunk·24B）。
        callback:       附加追迹回调（如 perturb 误差注入），在评估克隆链上执行；
            其预计算张量须为 float64（先 ``seq.to(torch.float64)`` 再 build）。
    """
    if image_sampling < 8:
        raise ValueError(f"image_sampling must be >= 8, got {image_sampling}")
    if image_delta is not None and image_delta <= 0:
        raise ValueError(f"image_delta must be positive, got {image_delta}")
    ev = _eval_seq(seq, pupil, wavelengths)
    tr = _trace(ev, callback)
    sp = _reference_sphere(tr)
    P, F_, W, _N = tr.points.shape[:4]
    device, dtype = tr.points.device, tr.points.dtype
    H = image_sampling
    warnings: list[str] = []

    # ---- 网格中心(传感器局部坐标) ----
    centers = tr.sensor_tf.transform_points(sp.centers_g, inverse=True)[..., :2]

    # ---- 像方 NA:存活光线相对主光线方向的最大半角正弦 ----
    d0 = tr.dirs[..., 0:1, :]  # (P,F,W,1,3)
    cosang = (tr.dirs * d0).sum(dim=-1).clamp(-1.0, 1.0)
    sinang = cosang.square().neg().add(1.0).clamp_min(0).sqrt()
    na = torch.where(sp.valid, sinang, torch.zeros_like(sinang)).amax(dim=-1)
    na_max = float(na.max())
    if na_max <= 0:
        raise RuntimeError("没有存活光线,无法计算 PSF")

    lam_mm = tr.wavelengths.to(dtype) * _NM_TO_MM  # (W,)
    delta_nyq = float(lam_mm.min()) / (4.0 * na_max)
    if image_delta is None:
        delta = delta_nyq
    else:
        delta = image_delta
        if image_delta > delta_nyq:
            warnings.append(
                f"image_delta={image_delta:.4g} mm 超过 Nyquist 建议 "
                f"{delta_nyq:.4g} mm(λ_min/(4·NA)),PSF 欠采样"
            )

    # ---- 光瞳采样充分性(heuristic):N_alive ≥ (2·OPD_ptv/λ)² ----
    pos = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("-inf")))
    neg = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("inf")))
    ptv = pos.amax(dim=-1) - neg.amin(dim=-1)  # (P,F,W)
    waves = (ptv / lam_mm).max()
    n_alive = int(sp.valid.sum(dim=-1).min())
    need = int((2.0 * float(waves)) ** 2)
    if torch.isfinite(waves) and n_alive < need:
        warnings.append(
            f"光瞳采样可能不足:存活 {n_alive} 根 < 估计需求 {need} 根"
            f"(OPD 峰谷 {float(waves):.1f}λ);建议提高光瞳密度"
        )

    # ---- 采样网格(传感器局部 → 全局) ----
    g = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2) * delta
    qx, qy = torch.meshgrid(g, g, indexing="ij")  # (H,H);i↔x, j↔y
    q_loc = torch.stack((qx, qy, torch.zeros_like(qx)), dim=-1)  # (H,H,3)
    c3 = torch.cat((centers, torch.zeros_like(centers[..., :1])), dim=-1)
    q_loc = q_loc.view(1, 1, 1, H, H, 3) + c3.view(P, F_, W, 1, 1, 3)
    Q = tr.sensor_tf.transform_points(q_loc)  # (P,F,W,H,H,3) 全局

    # ---- Kirchhoff 积分:U(Q) = Σ_r K·exp(i·k0·(OPD + n_img·|r|)) ----
    k0 = 2.0 * torch.pi / lam_mm  # (W,)
    psf = torch.zeros(P, F_, W, H, H, device=device, dtype=dtype)
    for p in range(P):
        for f in range(F_):
            for w in range(W):
                valid = sp.valid[p, f, w]
                n_v = int(valid.sum())
                if n_v == 0:
                    warnings.append(f"pop={p} field={f} λidx={w}: 无存活光线")
                    continue
                S = sp.points[p, f, w][valid]  # (Nv,3)
                nrm = sp.normals[p, f, w][valid]
                opd = sp.opd[p, f, w][valid]  # (Nv,)
                dirs = tr.dirs[p, f, w][valid]
                n_img = tr.n_img[p, f, w][valid]
                q = Q[p, f, w]  # (H,H,3)
                amp = torch.zeros(H, H, device=device, dtype=torch.complex128)
                for s in range(0, n_v, chunk):
                    Sc = S[s : s + chunk]  # (Nc,3)
                    r = q.unsqueeze(-2) - Sc  # (H,H,Nc,3)
                    rho = r.norm(dim=-1).clamp_min(1e-300)  # (H,H,Nc)
                    phase = (opd[s : s + chunk] + n_img[s : s + chunk] * rho) * k0[w]
                    nrmc = nrm[s : s + chunk]  # (Nc,3)
                    cos_i = (dirs[s : s + chunk] * nrmc).sum(dim=-1)  # (Nc,)
                    cos_d = (r / rho.unsqueeze(-1) * nrmc).sum(dim=-1)  # (H,H,Nc)
                    k_obl = 0.5 * (cos_i + cos_d)  # Kirchhoff 倾斜因子
                    amp = amp + (
                        k_obl * torch.exp(1j * phase.to(torch.complex128))
                    ).sum(dim=-1)
                psf[p, f, w] = amp.abs().square()

    # ---- 归一(每张 Σ=1)与边缘截断诊断 ----
    total = psf.sum(dim=(-2, -1), keepdim=True)
    psf = torch.where(total > 0, psf / total.clamp_min(1e-300), psf)
    edge = torch.zeros(H, H, dtype=torch.bool, device=device)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    edge_frac = psf.mul(edge).sum(dim=(-2, -1)).max()
    if float(edge_frac) > 0.02:
        warnings.append(
            f"PSF 边缘能量占比 {float(edge_frac):.1%} > 2%,网格可能过小(截断)"
        )

    return MonoPsf(
        psf=psf,
        delta=delta,
        centers=centers,
        wavelengths=tr.wavelengths,
        na=na,
        alive=sp.valid,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 相机 RGB 合成
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class CameraPsf:
    """RGB 通道 PSF（白平衡后逐通道 Σ=1）与诊断。float64。"""

    rgb: Tensor  # (P,F,3,H,H);[...,i,j] ↔ (x_i, y_j)
    delta: float  # 像面采样间隔 mm/px
    centers: Tensor  # (P,F,2) 公共参考中心(传感器局部 xy, mm)
    response: Tensor  # (W,3) 实际波长网格上的响应
    wavelengths: Tensor  # (W,) nm
    warnings: list[str]


def _shift_integrate(
    psf: Tensor, centers: Tensor, c_ref: Tensor, delta: float, weights: Tensor
) -> Tensor:
    """逐波长 PSF（各自以 centers 为中心）→ RGB（公共中心 c_ref）。

    强度(c_ref + q) = psf_w(q + c_ref − c_w)：逐波长双线性重采样
    （亚像素平移，域外补零），再按响应加权求和。
    grid_sample 约定：grid 末维 (x, y)，x ↔ 输入末维（j，物理 y），
    y ↔ 输入倒数第二维（i，物理 x），align_corners 像素中心归一化。
    """
    P, F_, W, H, _ = psf.shape
    base = torch.arange(H, dtype=psf.dtype, device=psf.device)
    ii = base.view(H, 1).expand(H, H)  # i 索引(物理 x 向)
    jj = base.view(1, H).expand(H, H)  # j 索引(物理 y 向)
    out = torch.zeros(P, F_, 3, H, H, dtype=psf.dtype, device=psf.device)
    for p in range(P):
        for f in range(F_):
            off = (c_ref[p, f] - centers[p, f]) / delta  # (W,2) 像素偏移 (x, y)
            grid_y = 2.0 * (ii + off[:, 0].view(W, 1, 1)) / (H - 1) - 1.0
            grid_x = 2.0 * (jj + off[:, 1].view(W, 1, 1)) / (H - 1) - 1.0
            grid = torch.stack((grid_x, grid_y), dim=-1)  # (W,H,H,2)
            shifted = F.grid_sample(
                psf[p, f].unsqueeze(1),
                grid,  # (W,1,H,H)
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out[p, f] = torch.einsum("wij,wc->cij", shifted.squeeze(1), weights)
    return out


def camera_psf(
    seq: Sequential,
    response: SensorResponse,
    image_sampling: int,
    *,
    image_delta: float | None = None,
    pupil: SampleOptions | None = None,
    chunk: int = 512,
    callback: FlowCallback | None = None,
) -> CameraPsf:
    """模拟相机传感器：响应曲线波长网格上的单色 PSF → 加权合成 RGB 通道。

    公共参考中心 = 逐视场按总响应（R+G+B）加权的单色中心均值；横向色差
    表现为通道间的相对位移（亚像素重采样保留）。白平衡：逐通道除以响应
    和，平谱白 → 三通道等能量（各 Σ=1）。``image_delta`` 默认自动
    λ_min/(4·NA_max)，需要与既有图件像素对齐时可显式给定。
    """
    mono = mono_psf(
        seq,
        image_sampling,
        wavelengths=tuple(response.wavelengths.tolist()),
        image_delta=image_delta,
        pupil=pupil,
        chunk=chunk,
        callback=callback,
    )
    S = response.at(mono.wavelengths)  # (W,3)，dtype/device 跟随
    wsum = S.sum(dim=-1)  # (W,) 总响应
    c_ref = (mono.centers * wsum.view(1, 1, -1, 1)).sum(dim=2) / wsum.sum()  # (P,F,2)
    rgb = _shift_integrate(mono.psf, mono.centers, c_ref, mono.delta, S)
    rgb = rgb / S.sum(dim=0).clamp_min(1e-300).view(1, 1, 3, 1, 1)  # 白平衡
    return CameraPsf(
        rgb=rgb,
        delta=mono.delta,
        centers=c_ref,
        response=S,
        wavelengths=mono.wavelengths,
        warnings=mono.warnings,
    )
