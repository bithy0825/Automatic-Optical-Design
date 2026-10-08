"""相机传感器仿真：光谱响应加权的多波长直评合成 RGB（无重采样插值）。

同一绝对网格逐波长直评（见 :func:`analysis.psf.psf`），横向色差表现为
通道间的真实相对位移。等能光源（平谱）、镜头透过率理想；白平衡 = 各
通道按实际能量归一（Σ=1）。

camspec CSV 格式：``#`` 开头的注释头 + 数据行 ``波长nm,R,G,B``
（响应为任意单位的相对灵敏度）。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from torch import Tensor

from analysis.grid import GridSpec
from analysis.psf import Issue, psf
from component import FlowCallback, Sequential
from sampling import SampleOptions


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
            raise ValueError(f"expected 4 columns (wavelength,R,G,B), got {data.shape[1]}")
        wl = torch.from_numpy(data[:, 0]).double()
        rgb = torch.from_numpy(data[:, 1:4]).double()
        order = torch.argsort(wl)
        return cls(wavelengths=wl[order], rgb=rgb[order])

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        """读取预处理的相机数据（``data/cameras/*.pt``，峰值已归一）。

        随机选一相机（数据集采样）：``cls.from_file(random.choice(list(dir.glob('*.pt'))))``。
        """
        d = torch.load(path, weights_only=True, map_location="cpu")
        return cls(wavelengths=d["wavelengths"], rgb=d["rgb"])

    def at(self, wavelengths: Tensor) -> Tensor:
        """线性插值到给定波长 (W',) → (W',3)；域外取 0。dtype/device 跟随输入。"""
        base = self.wavelengths.to(wavelengths).contiguous()
        rgb = self.rgb.to(wavelengths)
        idx = torch.searchsorted(base, wavelengths.contiguous()).clamp(1, base.shape[0] - 1)
        lo, hi = base[idx - 1], base[idx]
        t = ((wavelengths - lo) / (hi - lo).clamp_min(1e-300)).clamp(0.0, 1.0)
        inside = (wavelengths >= base[0]) & (wavelengths <= base[-1])
        interp = rgb[idx - 1] + (rgb[idx] - rgb[idx - 1]) * t.unsqueeze(-1)
        return interp * inside.unsqueeze(-1)


@dataclass(frozen=True, slots=True)
class CameraPsf:
    """RGB 通道 PSF（白平衡后逐通道 Σ=1）与诊断。float64。"""

    rgb: Tensor  # (P,F,3,H,H)；[...,i,j] ↔ (x_i, y_j)
    raw_rgb: Tensor  # (P,F,3,H,H) 响应加权的吞吐诚实强度（未归一化）
    delta: float  # 像素节距 mm/px
    centers: Tensor  # (P,F,2) 网格中心（传感器局部 xy, mm）
    response: Tensor  # (W,3) 实际波长网格上的响应
    wavelengths: Tensor  # (W,) nm
    throughput: Tensor  # (P,F,W) 参与积分的光线占比
    issues: list[tuple[Issue, int, int, int]]


def camera_psf(
    seq: Sequential,
    response: SensorResponse,
    grid: GridSpec,
    *,
    pupil: SampleOptions | None = None,
    pixel_integrate: int = 1,
    chunk_elems: int = 4_000_000,
    callback: FlowCallback | None = None,
) -> CameraPsf:
    """模拟相机传感器：响应曲线波长网格上的直评单色 PSF → 加权合成 RGB。

    逐波长在同一绝对网格上直评再按响应加权（``I_c = Σ_w S_w·|U_w|²``），
    不经过任何重采样。白平衡：逐通道除以实际总能量（平谱白 → 三通道
    等能量，各 Σ=1）。
    """
    mono = psf(
        seq,
        grid,
        pupil=pupil,
        wavelengths=tuple(response.wavelengths.tolist()),
        pixel_integrate=pixel_integrate,
        chunk_elems=chunk_elems,
        callback=callback,
    )
    S = response.at(mono.wavelengths)  # (W,3)，dtype/device 跟随
    raw_rgb = torch.einsum("pfwij,wc->pfcij", mono.raw, S)
    rgb = raw_rgb / raw_rgb.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
    return CameraPsf(
        rgb=rgb,
        raw_rgb=raw_rgb,
        delta=mono.delta,
        centers=mono.centers,
        response=S,
        wavelengths=mono.wavelengths,
        throughput=mono.throughput,
        issues=mono.issues,
    )
