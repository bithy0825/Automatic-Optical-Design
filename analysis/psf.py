"""PSF 主管线：显式网格上的 Kirchhoff 直评（无重采样插值）。

设计文档见 ``docs/psf-redesign.md``。网格逐 (P,F) 共享于全部波长——横向
色差体现为各波长 PSF 在绝对网格上的真实位移，而非后处理平移。能量记账
两分：``raw`` 为吞吐可见强度（``|U|²/N_emit²``），``psf`` 为逐张 Σ=1 的
显示约定。

处理模型
--------
逐结构独立处理：``structures`` 选中的每个个体先切片为 P=1 链
（:meth:`Sequential.select`），再经评估链（视场单点覆盖、光瞳/波长覆盖、
float64）追迹 + 直评。逐结构而非批量的原因：光源 ``epd`` 是种群共享标量，
Zemax 式缩放（:meth:`Sequential.scale`）要求逐个体独立缩放。

能量覆盖率（Parseval 归一化）
-----------------------------
连续 Parseval：``∫|U_true|² dQ = Σ_r w_r²·ΔA``（w_r 光瞳振幅权重，
``ΔA = A_pupil/N_emit`` 单光线代表面积），而 ``U_true = (ΔA/λR)·U_code``、
``raw = |U_code|²/N_emit²``，故::

    coverage = Σ_window raw · delta² · N_emit · A_pupil / ((λ·f_eff/n_img)² · Σ_valid w_r²)

w_r 取入射倾斜因子 cos_i（出射侧 cos_d 随 Q 变化无法入账；NA 不大时
cos_i ≈ 1，误差在几个百分点内）。Fourier 尺度用**光瞳映射有效焦距**
``f_eff = (epd/2)/NA``（逐 (P,F,W) 由边缘光线方向实测，对应 sine 映射
``px = f_eff·n·ŝ``）——参考球半径是顶点-像点距离，与 f_eff 可差数个
百分点，直接决定归一化常数。离散求和代替积分要求 delta 满足 Nyquist
（自动 delta 保证；用户覆盖时由 ``UNDERSAMPLED_IMAGE`` 告警）。

有限光线数 N 的相位混叠底噪（raw ≈ 1/N_emit / px，远场相位去相关后的
随机游走残差）使超大窗口的覆盖率被高估；grow 模式以 ``max_size`` 上限
缓解，阈值判定在常规窗口（64–256）内是可靠的。

阈值补救
--------
``energy_threshold = τ > 0`` 时按**全波长最差覆盖率** ``min_w coverage ≥ τ``
判定（物理系统只能缩一次），两种 remedy：

* ``"scale"``：Zemax 式整系统缩放（焦距随之变化），log 域线性二分
  ``[scale_bounds]``，收敛带 ``[τ, τ+search_tol]``；衍射极限饱和导致
  不可达时记 ``THRESHOLD_UNREACHABLE`` 并返回最缩小的尽力结果；
* ``"grow"``：系统不动，保 delta 扩窗口（``size`` 语义变为最小尺寸），
  整数二分最小满足尺寸，上限 ``max_size``。

误差注入
--------
``errors``（perturb 误差声明）在每个评估链上重新编译——缩放迭代改变名义
位姿，重编译保证注入基准正确；误差量纲为绝对单位（mm / rad），不随缩放
变化（加工误差的物理诚实解释）。逐结构 P=1 评估，每次调用是误差分布的
单次实现；蒙特卡洛公差分析请逐结构多次调用。``callback`` 为底层原始回调
（其预计算张量须为 float64）。
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import Tensor

from analysis._kernel import (
    _NM_TO_MM,
    ReferenceSphere,
    TraceSnapshot,
    eval_chain,
    integrate,
    reference_sphere,
    trace_with_opl,
)
from component import FlowCallback, InfiniteSource, Sequential
from perturb import Clip, Pose, Scatter, build_callback
from sampling import SampleOptions

# 缺省光瞳采样：等面积 fibonacci disk（每根光线权重相等，第 0 点为光瞳中心）
_DEFAULT_PUPIL = SampleOptions(method="fibonacci", region="disk", count=20001)

type ErrorDecl = Pose | Scatter | Clip | Mapping[str, Any]
"""perturb 误差声明（实例或配置映射），见 :mod:`perturb`。"""


class Issue(IntEnum):
    """结构化诊断（``PsfReport.issues`` 的元素类型；``(p, f, w)`` 定位）。

    ``p`` 为 ``PsfReport.structures`` 内的位置（非原链索引）；定位无关的
    维度记 -1：``UNDERSAMPLED_IMAGE`` 为全局 ``(-1,-1,-1)``，
    ``THRESHOLD_UNREACHABLE`` 为结构级 ``(p,-1,-1)``。
    """

    UNDERSAMPLED_PUPIL = 1  # 光瞳采样不足：存活光线数 < (2·OPD_ptv/λ)²
    UNDERSAMPLED_IMAGE = 2  # 像面欠采样：delta > λ_min/(4·NA_max)（全局一次）
    EDGE_TRUNCATION = 3  # PSF 边缘能量占比 > 2%（网格可能过小）
    NO_ALIVE_RAYS = 4  # 该 (p, f, w) 无存活光线
    THRESHOLD_UNREACHABLE = 5  # 阈值物理不可达（衍射极限饱和 / 窗口上限）


@dataclass(frozen=True, slots=True)
class PsfSpec:
    """PSF 计算规格：全部旋钮的单一事实源（缺省即常用工况）。

    * ``structures``：原链 P 维个体选择，索引或索引序列（缺省第 0 个）。
    * ``field``：视场角 (deg) 单点 ``(fx, fy)``，缺省轴上；任意角，不受
      训练视场网格限制。
    * ``wavelengths``：波长 (nm)，可多个；``None`` → 系统内嵌波长与其采样
      （与训练一致；内嵌缺失兜底 555nm）。
    * ``pupil``：仅 uniform/fibonacci disk（第 0 点须为瞳心）。
    * ``size`` / ``delta``：最终 PSF 尺寸与像素节距 (mm/px)；``delta=None``
      → Nyquist ``λ_min/(4·NA_max)``。grow 模式下 ``size`` 为最小尺寸。
    * ``energy_threshold``：窗口能量覆盖率阈值 τ ∈ [0,1)，0 = 不判定不补救。
    * ``errors`` / ``callback``：误差注入（见模块 docstring）。
    """

    structures: int | Sequence[int] = 0
    field: tuple[float, float] = (0.0, 0.0)
    wavelengths: tuple[float, ...] | None = None
    pupil: SampleOptions = _DEFAULT_PUPIL
    size: int = 64
    delta: float | None = None
    energy_threshold: float = 0.0
    remedy: Literal["scale", "grow"] = "scale"
    scale_bounds: tuple[float, float] = (1e-3, 1.0)
    search_tol: float = 0.01
    search_max_iter: int = 40
    max_size: int = 1024
    pixel_integrate: int = 1
    chunk_elems: int = 4_000_000
    errors: Iterable[ErrorDecl] | None = None
    callback: FlowCallback | None = None

    def __post_init__(self) -> None:
        if self.size < 8:
            raise ValueError(f"size must be >= 8, got {self.size}")
        if self.max_size < self.size:
            raise ValueError(f"max_size must be >= size ({self.size}), got {self.max_size}")
        if self.delta is not None and self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")
        if not 0.0 <= self.energy_threshold < 1.0:
            raise ValueError(f"energy_threshold must be in [0, 1), got {self.energy_threshold}")
        if self.remedy not in ("scale", "grow"):
            raise ValueError(f"remedy must be 'scale' or 'grow', got {self.remedy!r}")
        lo, hi = self.scale_bounds
        if not 0.0 < lo < hi:
            raise ValueError(f"scale_bounds must satisfy 0 < lo < hi, got {self.scale_bounds}")
        if self.search_tol <= 0:
            raise ValueError(f"search_tol must be positive, got {self.search_tol}")
        if self.search_max_iter < 1:
            raise ValueError(f"search_max_iter must be >= 1, got {self.search_max_iter}")
        if self.pixel_integrate < 1:
            raise ValueError(f"pixel_integrate must be >= 1, got {self.pixel_integrate}")
        if self.chunk_elems < 1:
            raise ValueError(f"chunk_elems must be >= 1, got {self.chunk_elems}")
        if self.pupil.region != "disk" or self.pupil.method not in ("uniform", "fibonacci"):
            raise ValueError(
                "pupil requires uniform/fibonacci disk sampling (point 0 is the pupil "
                f"center), got method={self.pupil.method!r} region={self.pupil.region!r}"
            )
        if self.wavelengths is not None and (
            len(self.wavelengths) == 0 or min(self.wavelengths) <= 0
        ):
            raise ValueError(f"wavelengths must be non-empty and positive, got {self.wavelengths}")
        if len(self.field) != 2:
            raise ValueError(f"field must be a (fx, fy) pair in degrees, got {self.field!r}")


@dataclass(frozen=True, slots=True)
class PsfReport:
    """逐波长单色 PSF、覆盖率与缩放记录、诊断。所有张量 float64。

    ``psf`` / ``raw`` 形状 ``(P',F,W,H,H)``：``P'`` 对应 ``structures``
    顺序，``F`` 为视场数（spec.field 单点 → F=1），``psf[...,i,j] ↔ (x_i, y_j)``。
    """

    psf: Tensor  # (P',F,W,H,H) 每张 Σ=1
    raw: Tensor  # (P',F,W,H,H) 吞吐可见强度（|U|²/N_emit²）
    delta: float  # 像素节距 mm/px（实际使用值）
    size: int  # 最终边长 H（grow 模式统一为各结构最大值）
    centers: Tensor  # (P',F,2) 网格中心（传感器局部 xy, mm）
    wavelengths: Tensor  # (W,) nm
    na: Tensor  # (P',F,W) 像方数值孔径（相对主光线最大半角正弦）
    throughput: Tensor  # (P',F,W) 参与积分的光线占比（valid/发射数）
    coverage: Tensor  # (P',F,W) 窗口能量覆盖率（Parseval 归一化，见模块 docstring）
    scale: Tensor  # (P',) 实际缩放因子（无缩放为 1）
    structures: tuple[int, ...]  # 原链 P 维索引（P' 维的顺序释义）
    opd: Tensor  # (P',F,W,N) 参考球光程差 mm（主光线为零点，无效 nan）
    alive: Tensor  # (P',F,W,N) 参与积分的光线掩码
    issues: list[tuple[Issue, int, int, int]]


# ── 内部状态 ──


@dataclass(slots=True)
class _Probe:
    """逐结构探测：P=1 基础链 + s=1 评估链与追迹（阈值搜索的出发点）。"""

    index: int  # 原链 P 维索引
    sub: Sequential  # P=1 基础链（缩放由此克隆）
    ev: Sequential  # s=1 评估链
    tr: TraceSnapshot
    sp: ReferenceSphere
    epd: float  # 评估链入瞳直径（s=1）
    na: Tensor  # (1,F,W)


@dataclass(slots=True)
class _Image:
    """单次网格积分的结果。"""

    psf: Tensor  # (1,F,W,H,H)
    raw: Tensor  # (1,F,W,H,H)
    coverage: Tensor  # (1,F,W)
    centers: Tensor  # (1,F,2)
    issues: list[tuple[Issue, int, int, int]]  # p 局部（恒 0），由调用方改映射


@dataclass(slots=True)
class _Eval:
    """一个搜索终态：评估链 + 追迹 + 图像 + 实际缩放与尺寸。"""

    ev: Sequential
    tr: TraceSnapshot
    sp: ReferenceSphere
    img: _Image
    scale: float
    size: int
    issues: list[tuple[Issue, int, int, int]]


# ── 图像形成与覆盖率 ──


def _na(tr: TraceSnapshot, sp: ReferenceSphere) -> Tensor:
    """像方 NA：存活光线相对主光线的最大半角正弦，(P,F,W)。"""
    d0 = tr.dirs[..., 0:1, :]  # (P,F,W,1,3)
    cosang = (tr.dirs * d0).sum(dim=-1).clamp(-1.0, 1.0)
    sinang = cosang.square().neg().add(1.0).clamp_min(0).sqrt()
    return torch.where(sp.valid, sinang, torch.zeros_like(sinang)).amax(dim=-1)


def _coverage(
    raw: Tensor, delta: float, epd: float, tr: TraceSnapshot, sp: ReferenceSphere
) -> Tensor:
    """窗口能量覆盖率 (P,F,W)（Parseval 归一化，见模块 docstring）。

    Fourier 尺度为逐 (P,F,W) 的光瞳映射有效焦距 ``f_eff = (epd/2)/NA``
    （由存活光线最大半角正弦实测）；na ≈ 0（全死）时覆盖率为 0。
    """
    na = _na(tr, sp)  # (P,F,W)
    cos_i = (tr.dirs * sp.normals).sum(dim=-1).clamp_min(0.0)  # (P,F,W,N)
    w2 = torch.where(sp.valid, cos_i.square(), torch.zeros_like(cos_i)).sum(dim=-1)
    n_emit = sp.points.shape[3]
    a_half = 0.5 * epd
    a_pupil = torch.pi * a_half**2
    lam_mm = tr.wavelengths * _NM_TO_MM  # (W,)
    n_img = tr.n_img[..., 0]  # (P,F,W)：逐格恒定（波长维已分离）
    f_eff = a_half / na.clamp_min(1e-300)  # (P,F,W)
    lam_f2 = (lam_mm.view(1, 1, -1) * f_eff / n_img).square()  # (P,F,W)
    denom = lam_f2 * w2
    num = raw.sum(dim=(-2, -1)) * (delta**2 * n_emit * a_pupil)
    ok = (na > 0) & (w2 > 0)
    return torch.where(ok, num / denom.clamp_min(1e-300), torch.zeros_like(num))


def _image(
    tr: TraceSnapshot,
    sp: ReferenceSphere,
    *,
    epd: float,
    delta: float,
    size: int,
    pixel_integrate: int,
    chunk_elems: int,
) -> _Image:
    """单次直评：网格 Kirchhoff 积分 + Σ=1 归一化 + 覆盖率 + 边缘诊断。"""
    P, F, W, N = tr.points.shape[:4]
    device = tr.points.device
    H = size
    issues: list[tuple[Issue, int, int, int]] = []

    # ── 网格中心（逐 (P,F) 波长平均；主光线死亡 → 存活质心兜底，内核已定） ──
    centers_g = sp.centers_g.mean(dim=2)  # (P,F,3)
    centers = tr.sensor_tf.transform_points(centers_g, inverse=True)[..., :2]

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
    k0 = 2.0 * torch.pi / (tr.wavelengths * _NM_TO_MM)  # (W,)
    raw = integrate(sp, tr.dirs, tr.n_img, Q, k0, chunk_elems)
    raw = raw / float(N) ** 2  # 归一到发射数：峰值 ≈ 吞吐²
    if n_sub > 1:
        raw = raw.view(P, F, W, H, n_sub, H, n_sub).mean(dim=(4, 6))

    # ── 覆盖率（Parseval 归一化） ──
    coverage = _coverage(raw, delta, epd, tr, sp)

    # ── 归一化（每张 Σ=1，显示约定）与边缘截断诊断 ──
    total = raw.sum(dim=(-2, -1), keepdim=True)
    psf_n = torch.where(total > 0, raw / total.clamp_min(1e-300), raw)
    edge = torch.zeros(H, H, dtype=torch.bool, device=device)
    edge[:2, :] = edge[-2:, :] = edge[:, :2] = edge[:, -2:] = True
    edge_frac = psf_n.mul(edge).sum(dim=(-2, -1))  # 已归一，即边缘能量占比
    for p, f, w in (edge_frac > 0.02).nonzero().tolist():
        issues.append((Issue.EDGE_TRUNCATION, p, f, w))

    return _Image(psf=psf_n, raw=raw, coverage=coverage, centers=centers, issues=issues)


def _check_sampling(
    sp: ReferenceSphere, lam_mm: Tensor
) -> tuple[Tensor, list[tuple[Issue, int, int, int]]]:
    """光瞳采样充分性诊断：存活数 < (2·OPD_ptv/λ)² 则逐格记录。"""
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


# ── 回调装配与探测 ──


def _extra(ev: Sequential, spec: PsfSpec) -> tuple[FlowCallback, ...] | None:
    """评估链上的附加回调：用户 callback 原样注入；errors 逐链重编译。

    errors 经 :func:`perturb.build_callback` 在当前评估链上编译——float64
    自动满足，名义位姿随缩放迭代自动跟随当前链。
    """
    cbs: list[FlowCallback] = []
    if spec.callback is not None:
        cbs.append(spec.callback)
    if spec.errors:
        cbs.append(build_callback(ev, spec.errors))
    return tuple(cbs) if cbs else None


def _wavelengths(sub: Sequential, spec: PsfSpec) -> tuple[float, ...] | None:
    """波长缺省链：spec 显式 > 系统内嵌（None 交由评估链沿用其采样）> 555nm。"""
    if spec.wavelengths is not None:
        return spec.wavelengths
    src = sub[0]
    if isinstance(src, InfiniteSource) and src.wavelengths:
        return None  # 沿用内嵌波长与其采样配置（与训练一致）
    return (555.0,)


def _probe(seq: Sequential, idx: int, spec: PsfSpec) -> _Probe:
    """单结构探测：切片 P=1 → 评估链 → 追迹（含误差注入）→ 参考球。"""
    sub = seq.select(idx)
    ev = eval_chain(sub, spec.pupil, _wavelengths(sub, spec), spec.field)
    tr = trace_with_opl(ev, _extra(ev, spec))
    sp = reference_sphere(tr)
    return _Probe(
        index=idx,
        sub=sub,
        ev=ev,
        tr=tr,
        sp=sp,
        epd=cast(InfiniteSource, ev[0]).epd,
        na=_na(tr, sp),
    )


# ── 阈值搜索 ──


def _worst(img: _Image) -> float:
    """全 (F,W) 最差覆盖率（物理系统只能缩/扩一次，按最差判定）。"""
    return float(img.coverage.min())


def _eval_at(pb: _Probe, spec: PsfSpec, s: float, delta: float) -> _Eval:
    """scale 模式单点评估：缩放 → 评估链 → 追迹 → 积分（s=1 复用探测）。"""
    if s == 1.0:
        ev, tr, sp, epd = pb.ev, pb.tr, pb.sp, pb.epd
    else:
        scaled = pb.sub.scale(s)
        ev = eval_chain(scaled, spec.pupil, _wavelengths(scaled, spec), spec.field)
        tr = trace_with_opl(ev, _extra(ev, spec))
        sp = reference_sphere(tr)
        epd = cast(InfiniteSource, ev[0]).epd
    img = _image(
        tr,
        sp,
        epd=epd,
        delta=delta,
        size=spec.size,
        pixel_integrate=spec.pixel_integrate,
        chunk_elems=spec.chunk_elems,
    )
    return _Eval(ev=ev, tr=tr, sp=sp, img=img, scale=s, size=spec.size, issues=[])


def _grow_at(pb: _Probe, spec: PsfSpec, delta: float, size: int) -> _Eval:
    """grow 模式单点评估：复用探测追迹，仅重铺网格再积分。"""
    img = _image(
        pb.tr,
        pb.sp,
        epd=pb.epd,
        delta=delta,
        size=size,
        pixel_integrate=spec.pixel_integrate,
        chunk_elems=spec.chunk_elems,
    )
    return _Eval(ev=pb.ev, tr=pb.tr, sp=pb.sp, img=img, scale=1.0, size=size, issues=[])


def _search_scale(pb: _Probe, spec: PsfSpec, delta: float, tau: float, p: int) -> _Eval:
    """缩放二分：不变式 coverage(lo) ≥ τ > coverage(hi)，收敛带 [τ, τ+tol]。"""
    first = _eval_at(pb, spec, 1.0, delta)
    if _worst(first.img) >= tau:
        return first
    lo, hi = spec.scale_bounds
    b = min(1.0, hi)
    b_state = first if b == 1.0 else _eval_at(pb, spec, b, delta)
    if _worst(b_state.img) >= tau:
        return b_state  # 上界已满足（hi < 1 的罕见配置）
    a_state = _eval_at(pb, spec, lo, delta)
    if _worst(a_state.img) < tau:
        # 衍射极限饱和：最大缩小仍不达标 → 尽力结果 + 诊断
        a_state.issues.append((Issue.THRESHOLD_UNREACHABLE, p, -1, -1))
        return a_state
    a = lo
    for _ in range(spec.search_max_iter):
        mid = 0.5 * (a + b)
        st = _eval_at(pb, spec, mid, delta)
        w = _worst(st.img)
        if tau <= w <= tau + spec.search_tol:
            return st
        if w >= tau:
            a, a_state = mid, st
        else:
            b = mid
    return a_state  # 迭代耗尽：返回最接近 τ 的可行解


def _search_grow(pb: _Probe, spec: PsfSpec, delta: float, tau: float, p: int) -> _Eval:
    """扩窗搜索：保 delta 扩 size，整数二分最小满足尺寸（上限 max_size）。"""
    first = _grow_at(pb, spec, delta, spec.size)
    if _worst(first.img) >= tau:
        return first
    lo = spec.size
    hi_state: _Eval | None = None
    h = spec.size * 2
    while True:
        h = min(h, spec.max_size)
        st = _grow_at(pb, spec, delta, h)
        if _worst(st.img) >= tau:
            hi_state = st
            break
        if h >= spec.max_size:
            st.issues.append((Issue.THRESHOLD_UNREACHABLE, p, -1, -1))
            return st  # 窗口上限仍不达标 → 尽力结果 + 诊断
        lo, h = h, h * 2
    # (lo, best] 整数二分：best 恒为满足 τ 的最小已知尺寸
    best = hi_state
    while best.size - lo > 1:
        mid = (lo + best.size) // 2
        st = _grow_at(pb, spec, delta, mid)
        w = _worst(st.img)
        if tau <= w <= tau + spec.search_tol:
            return st
        if w >= tau:
            best = st
        else:
            lo = mid
    return best


# ── 公开入口 ──


def psf(seq: Sequential, spec: PsfSpec | None = None) -> PsfReport:
    """计算选中结构在指定工况下的 Kirchhoff 单色 PSF（逐波长，网格共享）。

    Args:
        seq: 光学系统（不被修改；逐结构切片克隆评估并转 float64）。
        spec: 计算规格（缺省 :class:`PsfSpec` 全部默认值：第 0 个结构、
            轴上视场、内嵌波长、fibonacci 20001、64×64、无阈值）。
    """
    spec = spec if spec is not None else PsfSpec()
    indices = [spec.structures] if isinstance(spec.structures, int) else list(spec.structures)
    if not indices:
        raise ValueError("structures must be non-empty")
    P = seq.population
    for i in indices:
        if not 0 <= i < P:
            raise IndexError(f"structures index out of range [0, {P}): {i}")

    # ── 逐结构探测（s=1 追迹；确定全局 delta 需要全部 NA） ──
    probes = [_probe(seq, i, spec) for i in indices]
    na = torch.cat([pb.na for pb in probes], dim=0)  # (P',F,W)
    na_max = float(na.max())
    if na_max <= 0:
        raise RuntimeError("no alive rays; cannot compute PSF")

    lam_mm = probes[0].tr.wavelengths * _NM_TO_MM  # (W,)，各结构共享
    delta_nyq = float(lam_mm.min()) / (4.0 * na_max)
    issues: list[tuple[Issue, int, int, int]] = []
    if spec.delta is None:
        delta = delta_nyq
    else:
        delta = spec.delta
        if delta > delta_nyq:
            issues.append((Issue.UNDERSAMPLED_IMAGE, -1, -1, -1))

    # ── 逐结构阈值搜索（或无阈值单次评估） ──
    tau = spec.energy_threshold
    finals: list[_Eval] = []
    for p, pb in enumerate(probes):
        if tau > 0:
            st = (
                _search_scale(pb, spec, delta, tau, p)
                if spec.remedy == "scale"
                else _search_grow(pb, spec, delta, tau, p)
            )
        else:
            st = _eval_at(pb, spec, 1.0, delta)
        finals.append(st)

    # ── 统一 H（grow 模式各结构尺寸可能不同）：不足者复用追迹重积分 ──
    h_final = max(st.size for st in finals)
    for st in finals:
        if st.size < h_final:
            st.img = _image(
                st.tr,
                st.sp,
                epd=cast(InfiniteSource, st.ev[0]).epd,
                delta=delta,
                size=h_final,
                pixel_integrate=spec.pixel_integrate,
                chunk_elems=spec.chunk_elems,
            )
            st.size = h_final

    # ── 汇总 ──
    n_emit = probes[0].sp.points.shape[3]
    na_out, throughput_out = [], []
    for p, st in enumerate(finals):
        na_out.append(_na(st.tr, st.sp))
        n_alive, diag = _check_sampling(st.sp, lam_mm)
        throughput_out.append(n_alive.to(torch.float64) / n_emit)
        issues.extend((iss, p, f, w) for iss, _p, f, w in diag)  # 局部 p=0 → 结构位 p
        issues.extend(st.issues)
        issues.extend((iss, p, f, w) for iss, _p, f, w in st.img.issues)

    device = probes[0].tr.points.device
    return PsfReport(
        psf=torch.cat([st.img.psf for st in finals], dim=0),
        raw=torch.cat([st.img.raw for st in finals], dim=0),
        delta=delta,
        size=h_final,
        centers=torch.cat([st.img.centers for st in finals], dim=0),
        wavelengths=probes[0].tr.wavelengths,
        na=torch.cat(na_out, dim=0),
        throughput=torch.cat(throughput_out, dim=0),
        coverage=torch.cat([st.img.coverage for st in finals], dim=0),
        scale=torch.tensor([st.scale for st in finals], dtype=torch.float64, device=device),
        structures=tuple(indices),
        opd=torch.cat([st.sp.opd for st in finals], dim=0),
        alive=torch.cat([st.sp.valid for st in finals], dim=0),
        issues=issues,
    )


def psf_from_checkpoint(path: str | Path, spec: PsfSpec | None = None) -> PsfReport:
    """一站式入口：训练检查点 ``.pth`` → :func:`psf`。

    检查点内嵌完整配置（含波长、视场、EPD），经
    :func:`optimization.utils.load` 重建系统并注入训练后状态。
    """
    from optimization.utils import load  # 延迟导入：保持 analysis 轻依赖

    seq, _target = load(path)
    return psf(seq, spec)
