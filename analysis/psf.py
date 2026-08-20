"""Kirchhoff 波前 PSF:出瞳参考球 + 精确距离核 + 倾斜因子。

物理链(全程 float64、无梯度):
1. 评估克隆链(disk 光瞳采样覆盖)追迹,callback 逐段累积光程 OPL;
2. 在最后一个折射面后快照光线状态,解析反投影到以主光线像点为球心的参考球;
3. 从参考球面向传感器采样网格做 Kirchhoff 积分:
   相位 φ = k0·(OPL − OPL_chief + n_img·|r|),r 为球面点指向像元的精确矢量;
   倾斜因子 K = (cos_i + cos_d)/2(cos 均相对指向球心的内法向)。

约定:
* 主光线 = N 维第 0 条(disk 采样器构造保证第 0 点为光瞳中心);
* 网格中心 = 主光线(死则存活光线质心)在传感器上的落点,传感器局部坐标;
* 波长 nm、长度 mm;恒定活塞相位不影响 |U|²,故主光线死亡只影响中心定义。

注意:本模块会把评估克隆链 `.to(torch.float64)`,共享材料单例随之迁移
(既定行为);单进程单链用法下自洽,不要在同进程混用其他 dtype 的链。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from component import InfiniteSource, Refractor, Sensor, Sequential
from core import TraceFlow, Transformer
from sampling import SampleOptions

_NM_TO_MM: float = 1e-6


@dataclass(slots=True)
class _Trace:
    """追迹快照:末折射面状态 + 传感器数据 + 逐光线光程。"""

    points: Tensor        # (P,F,W,N,3) 末折射面命中点(全局)
    dirs: Tensor          # (P,F,W,N,3) 末折射面出射方向
    opl: Tensor           # (P,F,W,N) 快照处累积光程 mm
    n_img: Tensor         # (P,F,W,N) 像方折射率(逐光线波长)
    sensor_pts: Tensor    # (P,F,W,N,3) 传感器命中点(全局)
    sensor_tf: Transformer  # 传感器局部坐标系
    alive: Tensor         # (P,F,W,N) 传感器处最终存活
    wavelengths: Tensor   # (W,) nm


def _eval_seq(seq: Sequential, pupil: SampleOptions | None) -> Sequential:
    """评估链:disk 光瞳覆盖克隆 + 整体 float64。

    缺省 fibonacci disk 20001 点(等面积采样,每根光线权重相等)。
    rect/random 采样不含光瞳中心光线(主光线无定义),直接报错。
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
    eval_src = InfiniteSource(
        epd=src.epd,
        field_x=src.field_x,
        field_y=src.field_y,
        wavelength=src.wavelengths,
        population=src.population,
        pupil_cfg=cfg,
        field_cfg=src.field_cfg,
        wavel_cfg=src.wavel_cfg,
        transmitted=src.transmitted.clone(),
    )
    out = Sequential(eval_src, *(comp.clone() for comp in seq[1:]))
    out.rebind()
    return out.to(torch.float64)


def _trace_eval(seq: Sequential) -> _Trace:
    """追迹评估链:callback 逐段累积 OPL(几何段长 × 当前介质折射率)。

    段介质 = 上游最近 Source/Refractor 的出射材料(折射面处先按旧介质
    累积到达段、再更新介质,顺序即物理);Gap 不移动光线(Δ=0 自动跳过),
    Stop 在同介质内推进,均被 ‖Δp‖ 通式覆盖。末折射面状态逐面覆盖快照,
    循环结束即最后一面。

    InfiniteSource 无浮点 buffer,初始 Transformer 取进程缺省 dtype;
    故追迹期间临时切换缺省为 float64(同 train.py 惯例,退出即还原),
    保证位姿链与光线全程 f64。
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
                points=pts,
                dirs=flow.rays.directions,
                opl=opl[0].clone(),
                n=n_cur[0],
            )
        if isinstance(comp, Sensor):
            term.update(
                points=pts,
                tf=flow.transformer,
                hold=flow.verdict.hold,
                wl=flow.rays.wavelength[0, 0, :, 0],
            )
        return flow

    # 光源的初始位姿按进程缺省 dtype 创建,追迹期间临时切 f64(退出还原)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with torch.no_grad():
            seq(callback=_cb)
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
    """参考球:球面点、内法向、球面 OPD、有效掩码、球心与半径。"""

    points: Tensor      # (P,F,W,N,3) 球面交点(全局)
    normals: Tensor     # (P,F,W,N,3) 指向球心的单位内法向
    opd: Tensor         # (P,F,W,N) 球面光程差 mm(主光线为零点,无效 nan)
    valid: Tensor       # (P,F,W,N) 参与积分掩码
    centers_g: Tensor   # (P,F,W,3) 球心 = 主光线像点(全局)
    radius: Tensor      # (P,F,W) 球半径 mm


def _reference_sphere(tr: _Trace) -> _Sphere:
    """以主光线像点为球心、出瞳估计距离为半径,光线从末折射面解析反投影。

    出瞳 z = 末面后主光线与光轴的最近交点;病态(轴上视场)回退同 (p,w)
    其他视场中位数,全部病态再回退主光线末面点。精确距离核下 R 的小误差
    只影响采样经济性,不影响核的正确性。
    """
    chief_alive = tr.alive[..., 0]                       # (P,F,W)
    w = tr.alive.unsqueeze(-1).to(tr.sensor_pts.dtype)
    centroid = (tr.sensor_pts * w).sum(dim=-2) / w.sum(dim=-2).clamp_min(1.0)
    # 主光线死亡 → 存活质心兜底(仅影响中心定义;活塞相位不影响强度)
    centers_g = torch.where(chief_alive.unsqueeze(-1), tr.sensor_pts[..., 0, :], centroid)

    c0, d0 = tr.points[..., 0, :], tr.dirs[..., 0, :]    # (P,F,W,3) 主光线
    dxy2 = d0[..., :2].square().sum(dim=-1)
    ok = dxy2 > 1e-12
    t_star = -(c0[..., :2] * d0[..., :2]).sum(dim=-1) / dxy2.clamp_min(1e-300)
    z_cross = c0[..., 2] + t_star * d0[..., 2]
    z_sel = torch.where(ok, z_cross, torch.full_like(z_cross, float("nan")))
    med = z_sel.nanmedian(dim=1, keepdim=True).values    # (P,1,W) 视场维中位数
    z_ep = torch.where(ok, z_cross, med.expand_as(z_cross))
    z_ep = torch.where(z_ep.isnan(), c0[..., 2], z_ep)   # 全病态:主光线末面点
    ep = torch.stack(
        (torch.zeros_like(z_ep), torch.zeros_like(z_ep), z_ep), dim=-1
    )                                                    # (P,F,W,3) 出瞳中心(光轴上)
    radius = (centers_g - ep).norm(dim=-1).clamp_min(1e-9)

    # 光线-球解析求交:|X + t·d − C|² = R²,取前方根(朝球心会聚的一侧)
    m = tr.points - centers_g.unsqueeze(-2)              # (P,F,W,N,3)
    b = (tr.dirs * m).sum(dim=-1)                        # (P,F,W,N)
    disc = b.square() - m.square().sum(dim=-1) + radius.unsqueeze(-1).square()
    hit = disc > 0
    t = -b - disc.clamp_min(0).sqrt()
    points = tr.points + t.unsqueeze(-1) * tr.dirs
    opl = tr.opl + t * tr.n_img
    opd = opl - opl[..., :1]                             # 主光线(N=0)为零点
    normals = (centers_g.unsqueeze(-2) - points) / radius.unsqueeze(-1).unsqueeze(-1)
    valid = tr.alive & hit
    opd = torch.where(valid, opd, torch.full_like(opd, float("nan")))
    return _Sphere(points, normals, opd, valid, centers_g, radius)


@dataclass(slots=True)
class PsfResult:
    """逐波长单色 PSF 与诊断。所有张量 float64,位于 seq.device。"""

    psf: Tensor          # (P,F,W,H,H) 每张能量归一(Σ=1);psf[...,i,j] ↔ (x_i, y_j)
    delta: float         # 像面采样间隔 mm/px(自动建议时为实际使用值)
    centers: Tensor      # (P,F,W,2) 网格中心(传感器局部 xy, mm)
    opd: Tensor          # (P,F,W,N) 参考球光程差 mm(主光线为零点,无效光线 nan)
    alive: Tensor        # (P,F,W,N) 参与积分的光线掩码
    na: Tensor           # (P,F,W) 像方数值孔径(相对主光线最大半角正弦)
    warnings: list[str]  # 采样充分性诊断


def psf_kirchhoff(
    seq: Sequential,
    image_sampling: int,
    image_delta: float | None = None,
    *,
    pupil: SampleOptions | None = None,
    chunk: int = 512,
) -> PsfResult:
    """计算系统全部 (P,F,W) 的 Kirchhoff PSF(逐波长单色,网格以主光线像点为中心)。

    Args:
        seq:            光学系统(不被修改;内部克隆评估)。
        image_sampling: PSF 网格边长 H(H×H)。
        image_delta:    像面采样间隔 mm/px;None → 自动 λ_min/(4·NA_max)。
        pupil:          光瞳采样覆盖;仅 uniform/fibonacci disk;缺省 fibonacci 20001。
        chunk:          积分光线分块大小(控制单步内存 ~H²·chunk·24B)。
    """
    if image_sampling < 8:
        raise ValueError(f"image_sampling must be >= 8, got {image_sampling}")
    if image_delta is not None and image_delta <= 0:
        raise ValueError(f"image_delta must be positive, got {image_delta}")
    ev = _eval_seq(seq, pupil)
    tr = _trace_eval(ev)
    sp = _reference_sphere(tr)
    P, F, W, _N = tr.points.shape[:4]
    device, dtype = tr.points.device, tr.points.dtype
    H = image_sampling
    warnings: list[str] = []

    # ---- 网格中心(传感器局部坐标) ----
    centers = tr.sensor_tf.transform_points(sp.centers_g, inverse=True)[..., :2]

    # ---- 像方 NA:存活光线相对主光线方向的最大半角正弦 ----
    d0 = tr.dirs[..., 0:1, :]                                   # (P,F,W,1,3)
    cosang = (tr.dirs * d0).sum(dim=-1).clamp(-1.0, 1.0)
    sinang = cosang.square().neg().add(1.0).clamp_min(0).sqrt()
    na = torch.where(sp.valid, sinang, torch.zeros_like(sinang)).amax(dim=-1)
    na_max = float(na.max())
    if na_max <= 0:
        raise RuntimeError("没有存活光线,无法计算 PSF")

    lam_mm = tr.wavelengths.to(dtype) * _NM_TO_MM               # (W,)
    lam_min = float(lam_mm.min())
    delta_nyq = lam_min / (4.0 * na_max)
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
    # torch 2.11 无 nanmax/nanmin:无效光线以 ∓inf 填充后用 amax/amin,语义相同
    pos = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("-inf")))
    neg = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("inf")))
    ptv = pos.amax(dim=-1) - neg.amin(dim=-1)           # (P,F,W)
    waves = (ptv / lam_mm).max()                        # 最大波前峰谷(波长数)
    n_alive = int(sp.valid.sum(dim=-1).min())
    need = int((2.0 * float(waves)) ** 2)
    if torch.isfinite(waves) and n_alive < need:
        warnings.append(
            f"光瞳采样可能不足:存活 {n_alive} 根 < 估计需求 {need} 根"
            f"(OPD 峰谷 {float(waves):.1f}λ);建议提高光瞳密度"
        )

    # ---- 采样网格(传感器局部 → 全局) ----
    g = (torch.arange(H, device=device, dtype=dtype) - (H - 1) / 2) * delta
    qx, qy = torch.meshgrid(g, g, indexing="ij")        # (H,H);i↔x, j↔y
    q_loc = torch.stack((qx, qy, torch.zeros_like(qx)), dim=-1)         # (H,H,3)
    c3 = torch.cat((centers, torch.zeros_like(centers[..., :1])), dim=-1)
    q_loc = q_loc.view(1, 1, 1, H, H, 3) + c3.view(P, F, W, 1, 1, 3)
    Q = tr.sensor_tf.transform_points(q_loc)            # (P,F,W,H,H,3) 全局

    # ---- Kirchhoff 积分:U(Q) = Σ_r K·exp(i·k0·(OPD + n_img·|r|)) ----
    k0 = 2.0 * torch.pi / lam_mm                        # (W,)
    psf = torch.zeros(P, F, W, H, H, device=device, dtype=dtype)
    for p in range(P):
        for f in range(F):
            for w in range(W):
                valid = sp.valid[p, f, w]
                n_v = int(valid.sum())
                if n_v == 0:
                    warnings.append(f"pop={p} field={f} λidx={w}: 无存活光线")
                    continue
                S = sp.points[p, f, w][valid]           # (Nv,3)
                nrm = sp.normals[p, f, w][valid]
                opd = sp.opd[p, f, w][valid]            # (Nv,)
                dirs = tr.dirs[p, f, w][valid]
                n_img = tr.n_img[p, f, w][valid]
                q = Q[p, f, w]                          # (H,H,3)
                amp = torch.zeros(H, H, device=device, dtype=torch.complex128)
                for s in range(0, n_v, chunk):
                    Sc = S[s : s + chunk]                       # (Nc,3)
                    r = q.unsqueeze(-2) - Sc                    # (H,H,Nc,3)
                    rho = r.norm(dim=-1).clamp_min(1e-300)      # (H,H,Nc)
                    phase = (opd[s : s + chunk] + n_img[s : s + chunk] * rho) * k0[w]
                    nrmc = nrm[s : s + chunk]                   # (Nc,3)
                    cos_i = (dirs[s : s + chunk] * nrmc).sum(dim=-1)            # (Nc,)
                    cos_d = (r / rho.unsqueeze(-1) * nrmc).sum(dim=-1)          # (H,H,Nc)
                    k_obl = 0.5 * (cos_i + cos_d)               # Kirchhoff 倾斜因子
                    amp = amp + (k_obl * torch.exp(1j * phase.to(torch.complex128))).sum(dim=-1)
                psf[p, f, w] = amp.abs().square()

    # ---- 归一(每张 Σ=1)与边缘截断诊断 ----
    total = psf.sum(dim=(-2, -1), keepdim=True)
    psf = torch.where(total > 0, psf / total.clamp_min(1e-300), psf)
    edge = torch.zeros(H, H, dtype=torch.bool, device=device)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    edge_frac = psf.mul(edge).sum(dim=(-2, -1)).max()   # psf 已归一,即边缘能量占比
    if float(edge_frac) > 0.02:
        warnings.append(
            f"PSF 边缘能量占比 {float(edge_frac):.1%} > 2%,网格可能过小(截断)"
        )

    return PsfResult(
        psf=psf,
        delta=delta,
        centers=centers,
        opd=sp.opd,
        alive=sp.valid,
        na=na,
        warnings=warnings,
    )
