"""派生像质指标：MTF / 波前误差 / Strehl 比（同一内核输出的纯后处理）。

全部指标消费 :class:`~analysis.psf.PsfReport`，不重复追迹。
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from analysis._kernel import _NM_TO_MM
from analysis.psf import PsfReport


@dataclass(frozen=True, slots=True)
class MtfResult:
    """调制传递函数（OTF 归一化模，fftshift 后零频居中）。"""

    mtf: Tensor  # (P,F,W,H,H)
    freqs: Tensor  # (H,) cycles/mm（fftshift 对齐）


def mtf(res: PsfReport) -> MtfResult:
    """MTF：``|FFT(psf)|``（psf 已 Σ=1 归一，故 OTF(0)=1 约定自动满足）。"""
    otf = torch.fft.fft2(res.psf)
    m = torch.fft.fftshift(otf.abs(), dim=(-2, -1))
    freqs = torch.fft.fftshift(torch.fft.fftfreq(res.psf.shape[-1], d=res.delta))
    return MtfResult(mtf=m, freqs=freqs)


@dataclass(frozen=True, slots=True)
class WavefrontResult:
    """波前误差统计（存活光线 OPD，mm）。全死 (p,f,w) 记 nan（诚实）。"""

    rms: Tensor  # (P,F,W)
    ptv: Tensor  # (P,F,W)


def wavefront(res: PsfReport) -> WavefrontResult:
    """波前误差：存活光线 OPD 的 RMS 与峰谷值（等面积采样 → 等权）。"""
    opd, alive = res.opd, res.alive
    n = alive.sum(dim=-1)
    safe = torch.where(alive, opd, torch.zeros_like(opd))
    mean = safe.sum(dim=-1) / n.clamp_min(1)
    centered = torch.where(alive, safe - mean.unsqueeze(-1), torch.zeros_like(opd))
    var = centered.square().sum(dim=-1) / n.clamp_min(1)
    rms = var.sqrt()
    pos = torch.where(alive, opd, torch.full_like(opd, float("-inf")))
    neg = torch.where(alive, opd, torch.full_like(opd, float("inf")))
    ptv = pos.amax(dim=-1) - neg.amin(dim=-1)
    dead = n == 0
    rms = torch.where(dead, torch.full_like(rms, float("nan")), rms)
    ptv = torch.where(dead, torch.full_like(ptv, float("nan")), ptv)
    return WavefrontResult(rms=rms, ptv=ptv)


def strehl(res: PsfReport) -> Tensor:
    """Marechal 近似 Strehl 比：``exp(−(2πσ/λ)²)``，σ 为 OPD RMS。(P,F,W)"""
    sigma = wavefront(res).rms
    lam_mm = res.wavelengths * _NM_TO_MM  # (W,)
    return torch.exp(-((2.0 * math.pi * sigma / lam_mm.view(1, 1, -1)) ** 2))
