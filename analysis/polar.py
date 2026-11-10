"""旋转对称系统的极坐标 raw PSF 提取与任意方位角合成。

只计算基准方位角 ``phi=0`` 的径向视场。像面点由最终奇数 Cartesian
网格的唯一半径和等间隔角向节点组成，因此渲染回同一网格时不需要径向
插值；旋转换到角向 Fourier 系数上完成，不做 Cartesian 图像重采样。

精度策略：追迹、参考球、坐标、逐光线常量和 FFT/相位合成使用 float64；
最昂贵的逐像面点积分在 CUDA 上默认 float32，在其他设备上默认 float64。
可用 ``PolarPsfSpec.dtype`` 显式覆盖。raw 不做逐样本归一化或能量门控。
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from analysis._kernel import _NM_TO_MM, integrate_points
from analysis.psf import (
    _DEFAULT_PUPIL,
    Issue,
    PsfSpec,
    _check_sampling,
    _coverage,
    _probe,
)
from component import FlowCallback, Sequential
from sampling import SampleOptions


@dataclass(frozen=True, slots=True)
class PolarPsfSpec:
    """单结构、单径向视场的极坐标 PSF 规格。

    ``field=(theta,0)`` 中 theta 必须非负，单位 deg。``size`` 和
    ``angular_samples`` 使用奇数，分别保证中心像素和无 Nyquist 角向频率。
    ``dtype=None`` 时 CUDA 积分为 float32，其他设备为 float64；追迹和
    坐标等小张量始终为 float64。
    """

    structure: int = 0
    field: tuple[float, float] = (0.0, 0.0)
    wavelengths: tuple[float, ...] | None = None
    pupil: SampleOptions = _DEFAULT_PUPIL
    size: int = 63
    delta: float | None = None
    angular_samples: int = 257
    dtype: torch.dtype | None = None
    chunk_elems: int = 64_000_000
    callback: FlowCallback | None = None

    def __post_init__(self) -> None:
        if self.size < 9 or self.size % 2 == 0:
            raise ValueError(f"polar PSF size must be odd and >= 9, got {self.size}")
        if self.angular_samples < 9 or self.angular_samples % 2 == 0:
            raise ValueError(
                f"angular_samples must be odd and >= 9, got {self.angular_samples}"
            )
        if self.delta is not None and (
            not math.isfinite(self.delta) or self.delta <= 0
        ):
            raise ValueError(f"delta must be positive and finite, got {self.delta}")
        if self.dtype not in (None, torch.float32, torch.float64):
            raise ValueError(f"dtype must be None, float32 or float64, got {self.dtype}")
        if self.chunk_elems < 1:
            raise ValueError(f"chunk_elems must be >= 1, got {self.chunk_elems}")
        if (
            len(self.field) != 2
            or not all(math.isfinite(value) for value in self.field)
            or self.field[0] < 0
            or self.field[1] != 0
        ):
            raise ValueError(
                f"polar base field must be (theta, 0) with theta >= 0, got {self.field}"
            )


@dataclass(slots=True)
class PolarPsfReport:
    """极坐标直接积分结果及完整角向 Fourier 表示。

    ``raw`` 为 ``(W,Nr,Q)``，标度与 :attr:`PsfReport.raw` 相同，均为
    ``|U|²/N_emit²``。``coefficients`` 是完整 FFT/Q，未截断角向频率。
    ``radius_index`` 和 ``alpha`` 描述 ``(size,size)`` 的 ``(x,y)`` 网格。
    """

    raw: Tensor  # (W,Nr,Q)，积分 dtype
    coefficients: Tensor  # (W,Nr,Q)，complex128
    radii: Tensor  # (Nr,) mm，float64
    beta: Tensor  # (Q,) rad，float64
    radius_index: Tensor  # (H,H)，int64
    alpha: Tensor  # (H,H) rad，float64
    size: int
    delta: float
    structure: int
    field: tuple[float, float]
    centers: Tensor  # (2,) 传感器局部 mm，float64
    wavelengths: Tensor  # (W,) nm
    na: Tensor  # (W,)
    throughput: Tensor  # (W,)
    coverage: Tensor  # (W,)，由 phi=0 Cartesian raw 合成值计算
    integration_dtype: torch.dtype
    opd: Tensor  # (W,N) mm，float64
    alive: Tensor  # (W,N)
    issues: list[tuple[Issue, int, int, int]]


def field_from_polar(theta_deg: float, phi_deg: float) -> tuple[float, float]:
    """物理径向视场角/方位角 → InfiniteSource 的 ``(field_x,field_y)``。

    旧光源方向为 ``(tan(fx),tan(fy),1)``；而极坐标目标方向为
    ``sin(theta)·(cos(phi),sin(phi))`` 加 ``cos(theta)`` 的 z 分量。
    """

    if not 0 <= theta_deg < 90 or not math.isfinite(phi_deg):
        raise ValueError(f"require 0 <= theta < 90 and finite phi, got {theta_deg},{phi_deg}")
    phi = math.radians(phi_deg)
    tangent = math.tan(math.radians(theta_deg))
    return (
        math.degrees(math.atan(tangent * math.cos(phi))),
        math.degrees(math.atan(tangent * math.sin(phi))),
    )


def _integration_dtype(device: torch.device, requested: torch.dtype | None) -> torch.dtype:
    """大积分默认 CUDA f32 / 其他设备 f64；小张量不受此选择影响。"""

    if requested is not None:
        return requested
    return torch.float32 if device.type == "cuda" else torch.float64


def _grid_coordinates(size: int, delta: float, device: torch.device):
    """奇数方形网格 → 整数唯一半径、半径索引及极角。"""

    half = size // 2
    indices = torch.arange(-half, half + 1, device=device, dtype=torch.int64)
    ix, iy = torch.meshgrid(indices, indices, indexing="ij")
    radius2 = ix.square() + iy.square()
    unique2, radius_index = torch.unique(radius2, sorted=True, return_inverse=True)
    radii = unique2.to(torch.float64).sqrt() * delta
    alpha = torch.atan2(iy.to(torch.float64), ix.to(torch.float64))
    return radii, radius_index, alpha


@torch.no_grad()
def polar_coefficients(raw: Tensor) -> Tensor:
    """从 ``(W,Nr,Q)`` polar raw 重建完整 complex128 角向系数。

    磁盘只需保存 raw；这里的 FFT 是确定性的线性变换，不会引入新的
    光学近似。Q 必须为奇数，与 :class:`PolarPsfSpec` 的约定一致。
    """

    if raw.ndim != 3 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"polar raw must have shape (W,Nr,Q), got {tuple(raw.shape)}")
    if not raw.is_floating_point() or raw.is_complex():
        raise TypeError(f"polar raw must be a real floating tensor, got {raw.dtype}")
    nq = raw.shape[-1]
    if nq < 9 or nq % 2 == 0:
        raise ValueError(f"polar angular sample count must be odd and >=9, got {nq}")
    if not torch.isfinite(raw).all():
        raise ValueError("polar raw contains NaN/Inf")
    return torch.fft.fft(raw.to(torch.float64), dim=-1) / nq


@torch.no_grad()
def psf_polar(seq: Sequential, spec: PolarPsfSpec | None = None) -> PolarPsfReport:
    """直接计算 ``phi=0`` 基准视场的多波长极坐标 raw PSF。"""

    spec = spec if spec is not None else PolarPsfSpec()
    if not 0 <= spec.structure < seq.population:
        raise IndexError(
            f"structure must be in [0,{seq.population}), got {spec.structure}"
        )
    device = seq.device
    dtype = _integration_dtype(device, spec.dtype)

    # 复用经过验证的 f64 追迹/参考球流程；旧 PsfSpec 仅作公共参数载体。
    probe_spec = PsfSpec(
        structures=spec.structure,
        field=spec.field,
        wavelengths=spec.wavelengths,
        pupil=spec.pupil,
        size=spec.size,
        delta=spec.delta,
        energy_threshold=0.0,
        dtype=dtype,
        chunk_elems=spec.chunk_elems,
        callback=spec.callback,
    )
    pb = _probe(seq, spec.structure, probe_spec)
    na_max = float(pb.na.max())
    if not math.isfinite(na_max) or na_max <= 0:
        raise RuntimeError("no alive rays; cannot compute polar PSF")
    lam_mm = pb.tr.wavelengths * _NM_TO_MM
    delta_nyq = float(lam_mm.min()) / (4.0 * na_max)
    delta = delta_nyq if spec.delta is None else spec.delta

    radii, radius_index, alpha = _grid_coordinates(spec.size, delta, device)
    beta = (
        torch.arange(spec.angular_samples, device=device, dtype=torch.float64)
        * (2.0 * torch.pi / spec.angular_samples)
    )
    nr, nq = len(radii), spec.angular_samples

    # 与 _image 完全一致：所有波长共享逐视场参考中心和传感器坐标变换。
    centers_g = pb.sp.centers_g.mean(dim=2)
    centers = pb.tr.sensor_tf.transform_points(centers_g, inverse=True)[..., :2]
    qx = radii[:, None] * beta.cos()[None, :]
    qy = radii[:, None] * beta.sin()[None, :]
    q_local = torch.stack((qx, qy, torch.zeros_like(qx)), dim=-1)
    c3 = torch.cat((centers, torch.zeros_like(centers[..., :1])), dim=-1)
    q_local = q_local.view(1, 1, nr, nq, 3) + c3.view(1, 1, 1, 1, 3)
    points = pb.tr.sensor_tf.transform_points(q_local).reshape(1, 1, nr * nq, 3)

    k0 = 2.0 * torch.pi / (pb.tr.wavelengths * _NM_TO_MM)
    values = integrate_points(
        pb.sp,
        pb.tr.dirs,
        pb.tr.n_img,
        points,
        k0,
        spec.chunk_elems,
        dtype=dtype,
    )
    n_emit = pb.tr.points.shape[3]
    raw = values[0, 0].view(-1, nr, nq) / float(n_emit) ** 2
    if not torch.isfinite(raw).all():
        raise RuntimeError("polar raw PSF contains NaN/Inf")

    # 完整角向变换；旋转和 Cartesian 合成统一使用 f64/complex128。
    coefficients = polar_coefficients(raw)
    report = PolarPsfReport(
        raw=raw,
        coefficients=coefficients,
        radii=radii,
        beta=beta,
        radius_index=radius_index,
        alpha=alpha,
        size=spec.size,
        delta=delta,
        structure=spec.structure,
        field=spec.field,
        centers=centers[0, 0],
        wavelengths=pb.tr.wavelengths,
        na=pb.na[0, 0],
        throughput=pb.sp.valid.sum(dim=-1)[0, 0].to(torch.float64) / n_emit,
        coverage=torch.empty(0, device=device),  # 在下方由 phi=0 raw 合成填充
        integration_dtype=dtype,
        opd=pb.sp.opd[0, 0],
        alive=pb.sp.valid[0, 0],
        issues=[],
    )
    cart_raw = render_polar_psf(report, 0.0)
    coverage = _coverage(cart_raw[None, None], delta, pb.epd, pb.tr, pb.sp)[0, 0]
    _n_alive, issues = _check_sampling(pb.sp, lam_mm)
    if delta > delta_nyq:
        issues.append((Issue.UNDERSAMPLED_IMAGE, -1, -1, -1))
    report.coverage = coverage
    report.issues = issues
    return report


def _render_polar_coefficients(
    coefficients: Tensor,
    center_raw: Tensor,
    radius_index: Tensor,
    alpha: Tensor,
    phi_deg: float | Sequence[float],
) -> Tensor:
    """完整角向系数 → 任意方位角 Cartesian raw 的共享实现。"""

    if radius_index.ndim != 2 or radius_index.shape != alpha.shape:
        raise ValueError("radius_index and alpha must have the same (H,H) shape")
    device = coefficients.device
    phis = torch.as_tensor(phi_deg, dtype=torch.float64, device=device)
    scalar = phis.ndim == 0
    if scalar:
        phis = phis[None]
    if phis.ndim != 1 or phis.numel() == 0 or not torch.isfinite(phis).all():
        raise ValueError("phi_deg must be a finite scalar or non-empty 1D sequence")

    nq = coefficients.shape[-1]
    frequency = torch.arange(nq, device=device, dtype=torch.float64)
    frequency[frequency > nq // 2] -= nq
    outputs = []
    for phi in torch.deg2rad(phis):
        angle = (alpha - phi).unsqueeze(-1) * frequency
        basis = torch.exp(torch.complex(torch.zeros_like(angle), angle))
        values = (coefficients[:, radius_index] * basis.unsqueeze(0)).sum(dim=-1).real
        # r=0 的所有角向节点是同一物理点，直接保留其 raw 值。
        values[:, radius_index == 0] = center_raw.double().unsqueeze(-1)
        outputs.append(values)
    result = torch.stack(outputs)
    return result[0] if scalar else result


@torch.no_grad()
def render_polar_raw(
    polar_raw: Tensor,
    radius_index: Tensor,
    alpha: Tensor,
    phi_deg: float | Sequence[float],
) -> Tensor:
    """直接从紧凑保存的 polar raw 合成任意方位角 Cartesian raw。

    ``polar_raw`` 为 ``(W,Nr,Q)``；FFT 系数只在内存中临时重建为
    complex128，不需要写入磁盘。返回值不归一化、不截断。
    """

    return _render_polar_coefficients(
        polar_coefficients(polar_raw),
        polar_raw[:, 0, 0],
        radius_index,
        alpha,
        phi_deg,
    )


@torch.no_grad()
def render_polar_psf(
    report: PolarPsfReport, phi_deg: float | Sequence[float]
) -> Tensor:
    """合成任意视场方位角的 raw Cartesian PSF，不归一化、不截断。

    标量 ``phi_deg`` 返回 ``(W,H,H)``；一维序列返回 ``(A,W,H,H)``。
    所有输出点使用提取时的相同半径节点，仅做完整 Fourier 相位合成。
    """

    return _render_polar_coefficients(
        report.coefficients,
        report.raw[:, 0, 0],
        report.radius_index,
        report.alpha,
        phi_deg,
    )
