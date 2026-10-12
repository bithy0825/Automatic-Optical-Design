"""Kirchhoff 波前内核：追迹 + OPL 记账 + 参考球 + 向量化积分器。

物理链（全程 float64、无梯度）：

1. 评估克隆链（disk 光瞳采样覆盖、可选密集波长覆盖）追迹，callback 逐段
   累积光程 OPL；
2. 末折射面后快照光线状态，解析反投影到以主光线像点为球心的参考球；
3. 从参考球面向显式绝对网格做 Kirchhoff 积分：

   相位 φ = k0·(OPD − OPD_chief + n_img·|r|)，r 为球面点指向像元的精确矢量；
   倾斜因子 K = (cos_i + cos_d)/2（cos 均相对指向球心的内法向）；
   振幅含精确 1/|r| 因子（以 R/|r| 写入，R 为参考球半径，raw 标度不变）。

约定
----
* 主光线 = N 维第 0 条（disk 采样器构造保证第 0 点为光瞳中心）；
* 波长 nm、长度 mm；恒定活塞相位不影响 |U|²，故主光线死亡只影响中心定义；
* 评估在克隆链上进行并把链 ``.to(torch.float64)``，共享材料单例随之迁移
  （既定行为）；单进程单链用法下自洽，不要在同进程混用其他 dtype 的链；
* 积分对全部 N 根光线求和（无效光线的贡献精确为 0），按 (F,W) 批量、
  按光线数分块，块大小由 ``chunk_elems`` 元素预算推出。

诚实的近似清单：标量 Kirchhoff 近似（边界项略去）；等振幅光瞳（无切趾，
RayBundle 无振幅通道）；参考球半径的出瞳估计只影响采样经济性，不影响核
正确性（精确距离核 + 精确 1/ρ 振幅）；分块求和仅改变加法顺序（float64
非结合误差 ~1e-15）。
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from component import FlowCallback, InfiniteSource, Refractor, Sensor, Sequential
from core import TraceFlow, Transformer
from sampling import SampleOptions

_NM_TO_MM: float = 1e-6

# 缺省光瞳采样：等面积 fibonacci disk（每根光线权重相等，第 0 点为光瞳中心）
_DEFAULT_PUPIL = SampleOptions(method="fibonacci", region="disk", count=20001)


@dataclass(slots=True)
class TraceSnapshot:
    """追迹快照：末折射面状态 + 传感器数据 + 逐光线光程。"""

    points: Tensor  # (P,F,W,N,3) 末折射面命中点（全局）
    dirs: Tensor  # (P,F,W,N,3) 末折射面出射方向
    opl: Tensor  # (P,F,W,N) 快照处累积光程 mm
    n_img: Tensor  # (P,F,W,N) 像方折射率（逐光线波长）
    sensor_pts: Tensor  # (P,F,W,N,3) 传感器命中点（全局）
    sensor_tf: Transformer  # 传感器局部坐标系
    alive: Tensor  # (P,F,W,N) 传感器处最终存活
    wavelengths: Tensor  # (W,) nm


def eval_chain(
    seq: Sequential,
    pupil: SampleOptions | None,
    wavelengths: tuple[float, ...] | None,
    field: tuple[float, float] | None = None,
) -> Sequential:
    """评估链：disk 光瞳覆盖 + （可选）密集波长覆盖克隆 + 整体 float64。

    缺省 fibonacci disk 20001 点（等面积采样，每根光线权重相等）。
    rect/random 采样不含光瞳中心光线（主光线无定义），直接报错。
    波长覆盖时 ``wavel_cfg`` 重建为等数 uniform line，采样值与控制点一一对齐。
    视场覆盖时 ``field=(fx, fy)``（deg）单点取代原视场网格（``field_cfg``
    退化为 1 点），用于任意工况角的分析；None 保持原视场配置。
    """
    src = seq[0]
    if not isinstance(src, InfiniteSource):
        raise TypeError(f"first component must be an InfiniteSource, got {type(src).__name__}")
    cfg = pupil if pupil is not None else _DEFAULT_PUPIL
    if cfg.region != "disk" or cfg.method not in ("uniform", "fibonacci"):
        raise ValueError(
            "PSF requires uniform/fibonacci disk pupil sampling "
            f"(point 0 is the pupil center), got method={cfg.method!r} region={cfg.region!r}"
        )
    if wavelengths is None:
        wl, wavel_cfg = src.wavelengths, src.wavel_cfg
    else:
        wl = tuple(float(w) for w in wavelengths)
        wavel_cfg = SampleOptions(method="uniform", region="line", count=len(wl))
    if field is None:
        field_x, field_y, field_cfg = src.field_x, src.field_y, src.field_cfg
    else:
        field_x = field_y = None  # 由下方单点覆盖
        field_x, field_y = float(field[0]), float(field[1])
        field_cfg = SampleOptions(method="uniform", region="rect", count=(1, 1))
    eval_src = InfiniteSource(
        epd=src.epd,
        field_x=field_x,
        field_y=field_y,
        wavelength=wl,
        population=src.population,
        pupil_cfg=cfg,
        field_cfg=field_cfg,
        wavel_cfg=wavel_cfg,
        transmitted=src.transmitted.clone(),
    )
    out = Sequential(eval_src, *(comp.clone() for comp in seq[1:]))
    out.rebind()
    return out.to(torch.float64)


def trace_with_opl(
    seq: Sequential, extra: FlowCallback | Iterable[FlowCallback] | None
) -> TraceSnapshot:
    """追迹评估链：callback 逐段累积 OPL（几何段长 × 当前介质折射率）。

    段介质 = 上游最近 Source/Refractor 的出射材料（折射面处先按旧介质
    累积到达段、再更新介质）；Gap 不移动光线，Stop 在同介质内推进。
    *extra* 为误差注入等附加回调（单个或可迭代；先于 OPL 记账执行，
    记账只读当前状态）。

    InfiniteSource 无浮点 buffer，初始 Transformer 取进程缺省 dtype；
    追迹期间临时切换缺省为 float64（退出还原）。
    """
    if not any(isinstance(c, Refractor) for c in seq):
        raise ValueError("PSF requires at least one refracting surface (Refractor)")
    if not isinstance(seq[-1], Sensor):
        raise TypeError(f"last component must be a Sensor, got {type(seq[-1]).__name__}")

    prev: list[Tensor | None] = [None]
    opl: list[Tensor | None] = [None]
    n_cur: list[Tensor | None] = [None]
    snap: dict[str, Tensor] = {}
    terminal: dict[str, object] = {}

    def _cb(comp, flow: TraceFlow, _i: int) -> TraceFlow:
        pts = flow.rays.points
        if isinstance(comp, InfiniteSource):
            # 发射面：物方折射率 + 无穷远倾斜平面波相位参考（主光线为零）
            n_cur[0] = comp.transmitted(flow.rays.wavelength)
            opl[0] = (pts * flow.rays.directions).sum(dim=-1) * n_cur[0]
        else:
            # 到达段：几何长度 × 上游介质折射率（逐光线波长）
            assert opl[0] is not None and prev[0] is not None and n_cur[0] is not None
            opl[0] = opl[0] + (pts - prev[0]).norm(dim=-1) * n_cur[0]
        prev[0] = pts
        if isinstance(comp, Refractor):
            n_cur[0] = comp.transmitted(flow.rays.wavelength)
            snap.update(points=pts, dirs=flow.rays.directions, opl=opl[0].clone(), n=n_cur[0])
        if isinstance(comp, Sensor):
            terminal.update(
                points=pts,
                tf=flow.transformer,
                hold=flow.verdict.hold,
                wl=flow.rays.wavelength[0, 0, :, 0],
            )
        return flow

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    if extra is None:
        callbacks: FlowCallback | tuple[FlowCallback, ...] = _cb
    elif callable(extra):
        callbacks = (extra, _cb)
    else:
        callbacks = (*extra, _cb)
    try:
        with torch.no_grad():
            seq(callback=callbacks)
    finally:
        torch.set_default_dtype(prev_dtype)
    if not terminal:
        raise RuntimeError("no Sensor in the system")
    return TraceSnapshot(
        points=snap["points"],
        dirs=snap["dirs"],
        opl=snap["opl"],
        n_img=snap["n"],
        sensor_pts=cast(Tensor, terminal["points"]),
        sensor_tf=cast(Transformer, terminal["tf"]),
        alive=cast(Tensor, terminal["hold"]),
        wavelengths=cast(Tensor, terminal["wl"]),
    )


@dataclass(slots=True)
class ReferenceSphere:
    """参考球：球面点、内法向、球面 OPD、有效掩码、球心与半径。"""

    points: Tensor  # (P,F,W,N,3) 球面交点（全局）
    normals: Tensor  # (P,F,W,N,3) 指向球心的单位内法向
    opd: Tensor  # (P,F,W,N) 球面光程差 mm（主光线为零点，无效 nan）
    valid: Tensor  # (P,F,W,N) 参与积分掩码
    centers_g: Tensor  # (P,F,W,3) 球心 = 主光线像点（全局）
    radius: Tensor  # (P,F,W) 球半径 mm


def reference_sphere(tr: TraceSnapshot) -> ReferenceSphere:
    """以主光线像点为球心、出瞳估计距离为半径，光线从末折射面解析反投影。

    出瞳 z = 末面后主光线与光轴的最近交点；病态（轴上视场）回退同 (p,w)
    其他视场中位数，全部病态再回退主光线末面点。精确距离核下 R 的小误差
    只影响采样经济性，不影响核的正确性。
    """
    chief_alive = tr.alive[..., 0]  # (P,F,W)
    w = tr.alive.unsqueeze(-1).to(tr.sensor_pts.dtype)
    centroid = (tr.sensor_pts * w).sum(dim=-2) / w.sum(dim=-2).clamp_min(1.0)
    # 主光线死亡 → 存活质心兜底（仅影响中心定义；活塞相位不影响强度）
    centers_g = torch.where(chief_alive.unsqueeze(-1), tr.sensor_pts[..., 0, :], centroid)

    c0, d0 = tr.points[..., 0, :], tr.dirs[..., 0, :]  # (P,F,W,3) 主光线
    dxy2 = d0[..., :2].square().sum(dim=-1)
    ok = dxy2 > 1e-12
    t_star = -(c0[..., :2] * d0[..., :2]).sum(dim=-1) / dxy2.clamp_min(1e-300)
    z_cross = c0[..., 2] + t_star * d0[..., 2]
    z_sel = torch.where(ok, z_cross, torch.full_like(z_cross, float("nan")))
    med = z_sel.nanmedian(dim=1, keepdim=True).values  # (P,1,W) 视场维中位数
    z_ep = torch.where(ok, z_cross, med.expand_as(z_cross))
    z_ep = torch.where(z_ep.isnan(), c0[..., 2], z_ep)  # 全病态：主光线末面点
    ep = torch.stack((torch.zeros_like(z_ep), torch.zeros_like(z_ep), z_ep), dim=-1)
    radius = (centers_g - ep).norm(dim=-1).clamp_min(1e-9)

    # 光线-球解析求交：|X + t·d − C|² = R²，取前方根（朝球心会聚的一侧）
    m = tr.points - centers_g.unsqueeze(-2)  # (P,F,W,N,3)
    b = (tr.dirs * m).sum(dim=-1)  # (P,F,W,N)
    disc = b.square() - m.square().sum(dim=-1) + radius.unsqueeze(-1).square()
    hit = disc > 0
    t = -b - disc.clamp_min(0).sqrt()
    points = tr.points + t.unsqueeze(-1) * tr.dirs
    opl = tr.opl + t * tr.n_img
    opd = opl - opl[..., :1]  # 主光线（N=0）为零点
    normals = (centers_g.unsqueeze(-2) - points) / radius.unsqueeze(-1).unsqueeze(-1)
    valid = tr.alive & hit
    opd = torch.where(valid, opd, torch.full_like(opd, float("nan")))
    return ReferenceSphere(points, normals, opd, valid, centers_g, radius)


def integrate(
    sp: ReferenceSphere,
    dirs: Tensor,
    n_img: Tensor,
    Q: Tensor,
    k0: Tensor,
    chunk_elems: int,
) -> Tensor:
    """Kirchhoff 积分：``U(Q) = Σ_r K·(R/|r|)·exp(i·k0·(OPD + n_img·|r|))``。

    振幅含精确 1/|r| 因子，以 R/|r| 形式写入（R = 参考球半径，保持 raw
    标度约定不变：窗口尺度 R/|r| ≈ 1，远场/大窗口极限更精确）。

    Args:
        sp: 参考球状态（``(P,F,W,N,...)``）。
        dirs: 末折射面出射方向 ``(P,F,W,N,3)``。
        n_img: 像方折射率 ``(P,F,W,N)``。
        Q: 采样点全局坐标 ``(P,F,H,H,3)``——逐 (P,F) 一个网格，波长共享。
        k0: 逐波数 ``(W,)``（2π/λ，mm⁻¹）。
        chunk_elems: 单步张量元素预算（控制内存峰值）。

    Returns:
        未归一化的 ``|U|²``，形状 ``(P,F,W,H,H)``。无效光线贡献精确的 0。

    实现要点（性能重写，数学不变）：

    * 无效光线在逐光线小张量上消毒（几何量→有限值、倾斜权重→0），
      贡献精确为 0，与原双重 where 屏蔽语义一致；
    * ``|r|² = |Q|² − 2·Q·S + |S|²`` 展开式 + einsum（bmm 单核），
      全程不物化 ``(FW,H,H,C,3)`` 的 r 向量；
    * ``exp(iφ)`` 拆为 cos/sin 实数对累加（避免 complex128 大临时量）。
    """
    P, F, W, N = sp.points.shape[:4]
    H = Q.shape[2]
    device = sp.points.device
    psf = torch.zeros(P, F, W, H, H, device=device, dtype=torch.float64)
    FW = F * W
    chunk = max(1, chunk_elems // (FW * H * H))
    k0fw = k0.repeat(F)  # (FW,)：fw = f·W + w，与 reshape(FW,…) 同序
    for p in range(P):
        valid = sp.valid[p].reshape(FW, N)
        v3 = valid.unsqueeze(-1)
        S = torch.where(v3, sp.points[p].reshape(FW, N, 3), 0.0)
        nrm = torch.where(v3, sp.normals[p].reshape(FW, N, 3), 0.0)
        d = torch.where(v3, dirs[p].reshape(FW, N, 3), 0.0)
        opd = torch.where(valid, sp.opd[p].reshape(FW, N), 0.0) * k0fw.unsqueeze(-1)
        n_im = torch.where(valid, n_img[p].reshape(FW, N), 1.0) * k0fw.unsqueeze(-1)
        # 倾斜因子权重 w = valid·R/2（R = 参考球半径，1/ρ 振幅的保标度参考）
        w_ray = torch.where(valid, 0.5 * sp.radius[p].reshape(FW, 1), 0.0)
        ss = (S * S).sum(-1)  # (FW,N)
        sn = (S * nrm).sum(-1)  # (FW,N)
        cos_i = (d * nrm).sum(-1)  # (FW,N)
        Qf = Q[p].unsqueeze(1).expand(F, W, H, H, 3).reshape(FW, H * H, 3)
        qq = (Qf * Qf).sum(-1)  # (FW,HH)

        amp_re = torch.zeros(FW, H * H, device=device, dtype=torch.float64)
        amp_im = torch.zeros_like(amp_re)
        for s in range(0, N, chunk):
            sl = slice(s, s + chunk)
            qs = torch.einsum("fhc,fgc->fhg", Qf, S[:, sl])  # (FW,HH,C)
            rr = (qq.unsqueeze(-1) + ss[:, None, sl] - 2.0 * qs).clamp_min(1e-300)
            rho = rr.sqrt()
            # 相位 k0·(OPD + n_img·ρ)：k0 已预乘进逐光线项，单次融合乘加
            phase = torch.addcmul(opd[:, None, sl], n_im[:, None, sl], rho)
            # k_obl = ½(cos_i + cos_d)·(R/ρ) = (cos_i·ρ + (Q·n̂−S·n̂))·w/ρ²，
            # w_ray 已含 ½R 与有效掩码（rr≥1e-300；死亡光线 w=0 → 贡献精确 0）
            dot = torch.einsum("fhc,fgc->fhg", Qf, nrm[:, sl]) - sn[:, None, sl]
            k_obl = torch.addcmul(dot, cos_i[:, None, sl], rho) * w_ray[:, None, sl] / rr
            amp_re += torch.einsum("fhc,fhc->fh", k_obl, torch.cos(phase))
            amp_im += torch.einsum("fhc,fhc->fh", k_obl, torch.sin(phase))
        psf[p] = (amp_re.square() + amp_im.square()).view(F, W, H, H)
    return psf
