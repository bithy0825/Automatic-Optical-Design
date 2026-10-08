"""PSF 主管线：显式网格上的 Kirchhoff 直评（无重采样插值）。

网格逐 (P,F) 共享于全部波长——横向色差体现为各波长 PSF 在绝对网格上的
真实位移，而非后处理平移。能量记账两分：``raw`` 为吞吐可见强度
（``|U|²/N_emit²``），``psf`` 为逐张 Σ=1 的显示约定。
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor

from analysis._kernel import (
    _NM_TO_MM,
    ReferenceSphere,
    eval_chain,
    integrate,
    reference_sphere,
    trace_with_opl,
)
from analysis.grid import GridSpec
from component import FlowCallback, Sequential
from sampling import SampleOptions


class Issue(IntEnum):
    """结构化诊断（``PsfResult.issues`` 的元素类型；``(p, f, w)`` 定位）。"""

    UNDERSAMPLED_PUPIL = 1  # 光瞳采样不足：存活光线数 < (2·OPD_ptv/λ)²
    UNDERSAMPLED_IMAGE = 2  # 像面欠采样：delta > λ_min/(4·NA_max)（全局一次）
    EDGE_TRUNCATION = 3  # PSF 边缘能量占比 > 2%（网格可能过小）
    NO_ALIVE_RAYS = 4  # 该 (p, f, w) 无存活光线


@dataclass(frozen=True, slots=True)
class PsfResult:
    """逐波长单色 PSF 与诊断。所有张量 float64，位于 ``seq.device``。

    ``issues`` 元素为 ``(issue, p, f, w)``；``UNDERSAMPLED_IMAGE`` 为全局
    诊断，定位记 ``(-1, -1, -1)``。
    """

    psf: Tensor  # (P,F,W,H,H) 每张 Σ=1；psf[...,i,j] ↔ (x_i, y_j)
    raw: Tensor  # (P,F,W,H,H) 吞吐可见强度（|U|²/N_emit²）
    delta: float  # 像素节距 mm/px（实际使用值）
    centers: Tensor  # (P,F,2) 网格中心（传感器局部 xy, mm）
    wavelengths: Tensor  # (W,) nm
    na: Tensor  # (P,F,W) 像方数值孔径（相对主光线最大半角正弦）
    throughput: Tensor  # (P,F,W) 参与积分的光线占比（valid/发射数）
    opd: Tensor  # (P,F,W,N) 参考球光程差 mm（主光线为零点，无效 nan）
    alive: Tensor  # (P,F,W,N) 参与积分的光线掩码
    issues: list[tuple[Issue, int, int, int]]


def _check_sampling(
    sp: ReferenceSphere, lam_mm: Tensor
) -> tuple[Tensor, list[tuple[Issue, int, int, int]]]:
    """像方 NA 与光瞳采样充分性诊断：存活数 < (2·OPD_ptv/λ)² 则逐格记录。"""
    issues: list[tuple[Issue, int, int, int]] = []
    pos = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("-inf")))
    neg = torch.where(sp.valid, sp.opd, torch.full_like(sp.opd, float("inf")))
    ptv = pos.amax(dim=-1) - neg.amin(dim=-1)  # (P,F,W)
    waves = ptv / lam_mm.view(1, 1, -1)  # 波前峰谷（波长数）
    n_alive = sp.valid.sum(dim=-1)
    insufficient = torch.isfinite(waves) & (n_alive < (2.0 * waves) ** 2)
    for p, f, w in insufficient.nonzero().tolist():
        issues.append((Issue.UNDERSAMPLED_PUPIL, p, f, w))
    for p, f, w in (n_alive == 0).nonzero().tolist():
        issues.append((Issue.NO_ALIVE_RAYS, p, f, w))
    return n_alive, issues


def psf(
    seq: Sequential,
    grid: GridSpec,
    *,
    pupil: SampleOptions | None = None,
    wavelengths: tuple[float, ...] | None = None,
    pixel_integrate: int = 1,
    chunk_elems: int = 4_000_000,
    callback: FlowCallback | None = None,
) -> PsfResult:
    """计算系统全部 (P,F,W) 的 Kirchhoff 单色 PSF（网格逐视场共享于全部波长）。

    Args:
        seq: 光学系统（不被修改；内部克隆评估并转 float64）。
        grid: 像面采样网格（大小 / 节距 / 中心，见 :class:`GridSpec`）。
        pupil: 光瞳采样覆盖；仅 uniform/fibonacci disk；缺省 fibonacci 20001。
        wavelengths: 波长网格覆盖 (nm)；None → 系统配置的采样。
        pixel_integrate: 像素面积分的亚像素密度（1 = 像素中心点采样）。
        chunk_elems: 积分单步张量元素预算（控制内存峰值）。
        callback: 附加追迹回调（如 perturb 误差注入），在评估克隆链上执行；
            其预计算张量须为 float64（先 ``seq.to(torch.float64)`` 再 build）。
    """
    if pixel_integrate < 1:
        raise ValueError(f"pixel_integrate must be >= 1, got {pixel_integrate}")
    ev = eval_chain(seq, pupil, wavelengths)
    tr = trace_with_opl(ev, callback)
    sp = reference_sphere(tr)
    P, F, W, _N = tr.points.shape[:4]
    device = tr.points.device
    H = grid.size
    issues: list[tuple[Issue, int, int, int]] = []

    # ── 像方 NA 与采样节距 ──
    d0 = tr.dirs[..., 0:1, :]  # (P,F,W,1,3)
    cosang = (tr.dirs * d0).sum(dim=-1).clamp(-1.0, 1.0)
    sinang = cosang.square().neg().add(1.0).clamp_min(0).sqrt()
    na = torch.where(sp.valid, sinang, torch.zeros_like(sinang)).amax(dim=-1)
    na_max = float(na.max())
    if na_max <= 0:
        raise RuntimeError("no alive rays; cannot compute PSF")

    lam_mm = tr.wavelengths * _NM_TO_MM  # (W,)
    delta_nyq = float(lam_mm.min()) / (4.0 * na_max)
    if grid.delta is None:
        delta = delta_nyq
    else:
        delta = grid.delta
        if delta > delta_nyq:
            issues.append((Issue.UNDERSAMPLED_IMAGE, -1, -1, -1))

    # ── 网格中心（逐 (P,F) 波长平均；主光线死亡 → 存活质心兜底，内核已定） ──
    centers_g = sp.centers_g.mean(dim=2)  # (P,F,3)
    centers = tr.sensor_tf.transform_points(centers_g, inverse=True)[..., :2]
    if grid.center is not None:
        centers = (
            torch.tensor(grid.center, dtype=centers.dtype, device=device)
            .view(1, 1, 2)
            .expand(P, F, 2)
        )

    # ── 光瞳采样充分性与全死诊断 ──
    n_alive, diag = _check_sampling(sp, lam_mm)
    issues.extend(diag)

    # ── 采样网格（传感器局部 → 全局）；像素积分则细分后汇聚 ──
    n_sub = pixel_integrate
    Hf = H * n_sub
    g = (torch.arange(Hf, device=device, dtype=torch.float64) - (Hf - 1) / 2) * (delta / n_sub)
    qx, qy = torch.meshgrid(g, g, indexing="ij")  # (Hf,Hf)；i↔x, j↔y
    q_loc = torch.stack((qx, qy, torch.zeros_like(qx)), dim=-1)  # (Hf,Hf,3)
    c3 = torch.cat((centers, torch.zeros_like(centers[..., :1])), dim=-1)
    q_loc = q_loc.view(1, 1, Hf, Hf, 3) + c3.view(P, F, 1, 1, 3)
    Q = tr.sensor_tf.transform_points(q_loc)  # (P,F,Hf,Hf,3) 全局

    # ── Kirchhoff 直评：U(Q) = Σ_r K·exp(i·k0·(OPD + n_img·|r|)) ──
    k0 = 2.0 * torch.pi / lam_mm  # (W,)
    raw = integrate(sp, tr.dirs, tr.n_img, Q, k0, chunk_elems)
    raw = raw / float(sp.points.shape[3]) ** 2  # 归一到发射数：峰值 ≈ 吞吐²
    if n_sub > 1:
        raw = raw.view(P, F, W, H, n_sub, H, n_sub).mean(dim=(4, 6))

    # ── 归一化（每张 Σ=1，显示约定）与边缘截断诊断 ──
    total = raw.sum(dim=(-2, -1), keepdim=True)
    psf_n = torch.where(total > 0, raw / total.clamp_min(1e-300), raw)
    edge = torch.zeros(H, H, dtype=torch.bool, device=device)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    edge_frac = psf_n.mul(edge).sum(dim=(-2, -1))  # 已归一，即边缘能量占比
    for p, f, w in (edge_frac > 0.02).nonzero().tolist():
        issues.append((Issue.EDGE_TRUNCATION, p, f, w))

    return PsfResult(
        psf=psf_n,
        raw=raw,
        delta=delta,
        centers=centers,
        wavelengths=tr.wavelengths,
        na=na,
        throughput=n_alive.to(torch.float64) / sp.points.shape[3],
        opd=sp.opd,
        alive=sp.valid,
        issues=issues,
    )
