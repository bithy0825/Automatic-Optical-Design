"""PSF 数据集生成器 v2：全光谱积分 + 2D 视场 + 像差刻度盘 + 能量定标。

设计文档：``docs/psf-dataset-v2.md``（流程的单一事实源，本文件是实现）。

流程概要：逐镜头加载 checkpoint（256 结构）→ 两层 FPS 选 8 结构（0 固定 +
前 16 选 3 + 16–64 选 4）→ 每镜头随机 1 台相机（**全量波长**随相机网格，
31–186 个）→ 每结构 1 轴上参考（兼作 T(0) 能量基准）+ 12 方位 × 4 径向
= 48 抖动视场 → 每样本视场相关阈值 τ_j（Beta 采样）门控 + scale 补救 →
逐样本独立概率扰动（q_j 下降量门控 + 0.70 绝对下限，q 退火）→ 全光谱
通道合成 + 能量定标（``w_c·cos^n(θ)·T(θ)/T(0)``，连续无硬阈值）。

用法::

    python scripts/build_psf_dataset_v2.py --lib <库根> --out <输出目录> \
        [--lenses Zebase/A_001 ...] [--pupil 65536] [--device cuda]

产出：``<out>/psf/<样本名>.pt``（``{"psf": (64,64,3) float32, "meta": ...}``，
通道和 = 该工况相对能量 E_c，**非** Σ=1）+ ``<out>/meta/<镜头>.jsonl``
（断点续跑单元：已存在则跳过该镜头）+ ``<out>/metadata.jsonl``（合并产物）。
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import Issue, PsfSpec, psf
from component import Gap, InfiniteSource, Refractor, Sensor, Sequential, Stop
from optimization.target import FOV
from optimization.utils import load
from perturb import Pose
from sampling import SampleOptions

_NM_TO_MM = 1e-6
_SCHEMA_VERSION = 2

# 默认旋钮（docs/psf-dataset-v2.md §5/§7/§8）
_TAU_LO, _TAU_CAP = 0.80, 0.97  # τ_j 采样范围（视场相关 Beta）
_SCALE_LO = 0.3  # scale 补救下限
_PERTURB_PROB = 0.1  # 逐样本独立扰动概率
_Q_MIN, _Q_MAX = 0.02, 0.15  # q_j = max(q_min, x_q·q_max)
_COV_FLOOR_PERTURB = 0.70  # 扰动样本绝对覆盖率下限
_Q_LADDER = (1.0, 0.5, 0.25, 0.125)  # 扰动幅度退火
_THETA_REF, _E_REF = 45.0, 0.60  # 能量衰减标定点：45° → 60%
_PERTURB_WEIGHTS = (
    ("element", 0.30),
    ("single", 0.25),
    ("defocus", 0.20),
    ("spacing", 0.15),
    ("sensor_tilt", 0.10),
)
_N_AZIMUTH, _N_RADIAL = 12, 4  # 视场分档（档内全幅抖动 → 相邻档重叠）


# ── 镜头库与相机库 ──


def find_lenses(lib: Path) -> list[Path]:
    """库根 → 全部非废弃镜头目录（含唯一 .pth 的目录；跳过 ``_rejected``）。"""
    out = []
    for pth in sorted(lib.rglob("*.pth")):
        if "_rejected" in pth.parts:
            continue
        out.append(pth.parent)
    return sorted(set(out))


@dataclass(frozen=True)
class Camera:
    """一台相机的全光谱数据（加载时防御性重归一化 + 梯形积分权重）。"""

    name: str
    wavelengths: tuple[float, ...]  # 升序，W 个（31–186 随网格）
    rgb: torch.Tensor  # (W,3) float64，全局峰值归一
    trapz: torch.Tensor  # (W,) 梯形积分权重（该相机自己的网格）
    w_c: torch.Tensor  # (3,) 通道积分权重（max 通道归一）


def load_cameras(directory: Path) -> list[Camera]:
    """相机库 → 全光谱 Camera 列表。波长网格逐相机原样使用，不抽稀不重插。"""
    cams = []
    for f in sorted(directory.glob("*.pt")):
        d = torch.load(f, weights_only=True, map_location="cpu")
        wl, rgb = d["wavelengths"].double(), d["rgb"].double()
        rgb = rgb / rgb.max()  # 防御性重归一化（现库实测全部已为 1）
        order = torch.argsort(wl)  # 防御：保证波长升序（rgb 行同步）
        wl, rgb = wl[order], rgb[order]
        # 梯形权重：端点半权，内部取相邻 Δλ 均值（非均匀网格自动正确）
        trapz = torch.empty_like(wl)
        trapz[0], trapz[-1] = (wl[1] - wl[0]) / 2, (wl[-1] - wl[-2]) / 2
        if len(wl) > 2:
            trapz[1:-1] = (wl[2:] - wl[:-2]) / 2
        integ = (rgb * trapz.unsqueeze(1)).sum(dim=0)  # (3,) = ∫S_c dλ
        if bool((integ <= 0).any()):
            print(f"[warn] camera {f.stem}: 退化通道，跳过")
            continue
        cams.append(
            Camera(
                name=str(d.get("camera", f.stem)),
                wavelengths=tuple(float(w) for w in wl),
                rgb=rgb,
                trapz=trapz,
                w_c=integ / integ.max(),
            )
        )
    if not cams:
        raise FileNotFoundError(f"no cameras in {directory}")
    return cams


def _h_max(fov: FOV) -> float:
    """训练 FOV 配置 → 最大半视场角 (deg)。"""
    match fov:
        case float() as f:
            return f / 2.0
        case (float() as a, float() as b):
            return max(a, b) / 2.0
        case ((x0, x1), (y0, y1)):
            return max(abs(x0), abs(x1), abs(y0), abs(y1))
    raise TypeError(f"invalid fov: {fov!r}")


# ── 结构选择：两层 FPS ──


def _param_matrix(seq: Sequential, pool: int) -> torch.Tensor:
    """state_dict 前 pool 行 → (pool, D) 参数矩阵（键序固定，逐键拉平）。"""
    cols = []
    for _k, t in sorted(seq.state_dict().items()):
        t = t.detach().to(torch.float64)
        cols.append(t[:pool].reshape(min(pool, t.shape[0]), -1))
    return torch.cat(cols, dim=1)


def _fps(d: torch.Tensor, cand: list[int], seeds: list[int], k: int) -> list[int]:
    """在 cand 中 FPS 选 k 个（距离相对 seeds+已选集合的最大最小距离）。"""
    out: list[int] = []
    cand = list(cand)
    while len(out) < k and cand:
        d_min = d[cand][:, seeds + out].min(dim=1).values
        out.append(cand.pop(int(d_min.argmax())))
    return out


def select_structures(
    seq: Sequential, pool1: int = 16, picks1: int = 3, pool2: int = 64, picks2: int = 4
) -> tuple[int, ...]:
    """两层 FPS：0 固定 + 前 pool1 选 picks1 + [pool1, pool2) 选 picks2。

    参数空间 min-max 归一化（跨度取前 pool2 个结构的并集实测），欧氏距离。
    """
    pool2 = min(pool2, seq.population)
    v = _param_matrix(seq, pool2)
    span = v.max(dim=0).values - v.min(dim=0).values
    v = (v - v.min(dim=0).values) / span.clamp_min(1e-12)
    d = torch.cdist(v, v)
    first = [0] + _fps(d, list(range(1, pool1)), [0], picks1)
    second = _fps(d, list(range(pool1, pool2)), first, picks2)
    return tuple(first + second)


# ── 视场采样（12 方位 × 4 径向，档内全幅抖动 → 相邻档重叠） ──


def sample_fields(h_max: float, rng: torch.Generator) -> list[tuple[float, float, int, int, bool]]:
    """48 个视场点：``(fx, fy, 方位档 i, 径向档 j, 越界标志)``，单位 deg。

    φ_i = (i+0.5)·30° + U(−1,+1)·30°；r_j = (j+0.5)·(h/4) + U(−1,+1)·(h/4)，
    r clamp 到 [0.005·h, h]（下端避免与轴上参考样本精确重复；越上界记标志）。
    """

    def u() -> float:
        return float(torch.rand((), generator=rng)) * 2.0 - 1.0

    out = []
    for j in range(_N_RADIAL):
        for i in range(_N_AZIMUTH):
            phi = (i + 0.5) * (2 * math.pi / _N_AZIMUTH) + u() * (2 * math.pi / _N_AZIMUTH)
            r = (j + 0.5) * (h_max / _N_RADIAL) + u() * (h_max / _N_RADIAL)
            extra = r > h_max
            r = min(max(r, 0.005 * h_max), h_max)
            out.append((r * math.cos(phi), r * math.sin(phi), i, j, extra))
    return out


# ── 视场相关阈值采样（Beta 分布，α/β 随 θ̂ 线性滑动） ──


def tau_j(theta_hat: float, np_rng: np.random.Generator, tau_lo: float, tau_cap: float) -> float:
    """覆盖率阈值：x~Beta(α,β)，τ=τ_cap−x·(τ_cap−τ_lo)；轴上贴近 τ_cap。"""
    a, b = 1.5 + 1.5 * theta_hat, 8.0 - 6.8 * theta_hat
    x = float(np_rng.beta(a, b))
    return tau_cap - x * (tau_cap - tau_lo)


def q_j(theta_hat: float, np_rng: np.random.Generator, q_min: float, q_max: float) -> float:
    """扰动允许的最大覆盖率下降量：同族 Beta，轴上贴近 q_min（严）。"""
    a, b = 1.5 + 1.5 * theta_hat, 6.0 - 4.8 * theta_hat
    return max(q_min, float(np_rng.beta(a, b)) * q_max)


# ── 扰动采样（类型/目标/实现值，幅度与结构挂钩；同 v1 §6） ──


def _surfaces(chain: Sequential) -> list[int]:
    return [i for i, c in enumerate(chain) if isinstance(c, (Refractor, Stop, Sensor))]


def _perturb_targets(chain: Sequential) -> dict[str, Any]:
    """P=1 链的扰动目标清单：折射面号（1 起）、元件对、传感器面号。"""
    refr = [n for n, i in enumerate(_surfaces(chain), start=1) if isinstance(chain[i], Refractor)]
    n_s = len(_surfaces(chain))
    pairs = []
    for a, b in zip(refr, refr[1:], strict=False):  # 相邻配对：右操作数短 1
        ia, ib = _surfaces(chain)[a - 1], _surfaces(chain)[b - 1]
        between = [c for c in list(chain)[ia + 1 : ib] if not isinstance(c, Gap)]
        if not between:  # 两折射面间仅有 Gap → 同一元件
            pairs.append((a, b))
    return {"refr": refr, "pairs": pairs, "n_s": n_s}


def sample_perturbation(
    chain: Sequential, f_eff: float, f_number: float, lam_min_mm: float, rng: torch.Generator
) -> tuple[Pose, dict[str, Any]]:
    """抽一次扰动（类型 + 目标 + 实现值）；实现值物化为常数，退火只乘 q。"""
    t = _perturb_targets(chain)
    kinds = [(k, w) for k, w in _PERTURB_WEIGHTS if k != "element" or t["pairs"]]
    total = sum(w for _, w in kinds)
    r = float(torch.rand((), generator=rng)) * total
    kind = kinds[-1][0]
    for k, w in kinds:
        if r < w:
            kind = k
            break
        r -= w

    s_d, s_t = 2e-4 * f_eff, 2e-4
    s_f = 2.0 * lam_min_mm * f_number * f_number

    def normal(std: float) -> float:
        return float(torch.randn((), generator=rng) * std)

    def pick(seq_: list[int]) -> int:
        return seq_[int(torch.randint(len(seq_), (), generator=rng))]

    info: dict[str, Any] = {"kind": kind}
    if kind == "element":
        a, b = t["pairs"][int(torch.randint(len(t["pairs"]), (), generator=rng))]
        pose = Pose(
            first=a, last=b, dx=normal(s_d), dy=normal(s_d), alpha=normal(s_t), beta=normal(s_t)
        )
        info["target"] = [a, b]
    elif kind == "single":
        k = pick(t["refr"])
        if torch.rand((), generator=rng) < 0.5:
            pose = Pose(first=k, last=k, dx=normal(s_d), dy=normal(s_d))
        else:
            pose = Pose(first=k, last=k, alpha=normal(s_t), beta=normal(s_t))
        info["target"] = [k]
    elif kind == "defocus":
        pose = Pose(first=t["n_s"], dz=normal(s_f))
        info["target"] = [t["n_s"]]
    elif kind == "spacing":
        ks = [k for k in t["refr"] if k >= 2]
        k = pick(ks) if ks else t["refr"][0]
        pose = Pose(first=k, dz=normal(s_f))
        info["target"] = [k]
    else:  # sensor_tilt
        pose = Pose(first=t["n_s"], alpha=normal(s_t), beta=normal(s_t))
        info["target"] = [t["n_s"]]
    info["values"] = {
        k: v
        for k, v in vars(pose).items()
        if k in ("dx", "dy", "dz", "alpha", "beta") and v is not None
    }
    return pose, info


def _scale_pose(pose: Pose, q: float) -> Pose:
    """扰动实现值 ×q（幅度退火，不重抽）。"""

    def s(v):  # Spec = float | Tensor | None；物化为 float 后乘 q
        return None if v is None else float(v) * q

    return Pose(
        first=pose.first,
        last=pose.last,
        dx=s(pose.dx),
        dy=s(pose.dy),
        dz=s(pose.dz),
        alpha=s(pose.alpha),
        beta=s(pose.beta),
    )


# ── 光谱合成与能量定标 ──


def _spec(
    cam: Camera, field: tuple[float, float], tau: float, pupil_n: int, chunk_elems: int, s_lo: float
) -> PsfSpec:
    return PsfSpec(
        field=field,
        wavelengths=cam.wavelengths,
        pupil=SampleOptions(method="fibonacci", region="disk", count=pupil_n),
        size=64,
        delta=None,
        energy_threshold=tau,
        remedy="scale",
        scale_bounds=(s_lo, 1.0),
        chunk_elems=chunk_elems,
    )


def channel_synthesize(rep_psf: torch.Tensor, cam: Camera) -> torch.Tensor:
    """全光谱通道合成：(W,H,H) → (3,H,H)，每通道 Σ=1（能量定标前）。"""
    wt = cam.rgb.to(rep_psf.device) * cam.trapz.to(rep_psf.device).unsqueeze(1)  # (W,3)
    den = wt.sum(dim=0).clamp_min(1e-300)  # (3,)
    return torch.einsum("whk,wc->chk", rep_psf, wt) / den.view(3, 1, 1)


def channel_throughput(rep_tput: torch.Tensor, cam: Camera) -> torch.Tensor:
    """光谱加权逐通道吞吐：(W,) → (3,)。"""
    wt = cam.rgb.to(rep_tput.device) * cam.trapz.to(rep_tput.device).unsqueeze(1)
    den = wt.sum(dim=0).clamp_min(1e-300)
    return (rep_tput.unsqueeze(1) * wt).sum(dim=0) / den


def energy_scale(
    w_c: torch.Tensor, theta_deg: float, tput_ratio: torch.Tensor, n_exp: float
) -> torch.Tensor:
    """``E_c(θ) = w_c·cos^n(θ)·T(θ)/T(0)``（连续衰减，无硬阈值）。"""
    cos_n = math.cos(math.radians(theta_deg)) ** n_exp
    return w_c.to(tput_ratio.device) * cos_n * tput_ratio


# ── 单样本流程 ──


def _issues_of(rep, extra: list[str] | None = None) -> list[Any]:
    out: list[Any] = [int(i) for i, *_ in rep.issues]
    cov_raw = float(rep.coverage.min())
    if cov_raw > 1.02:
        out.append("coverage_inflated")
    if float(rep.throughput.min()) < 0.5:
        out.append("low_throughput")
    return out + list(extra or [])


def make_clean(
    seq: Sequential,
    structure: int,
    field: tuple[float, float],
    cam: Camera,
    tau: float,
    pupil_n: int,
    chunk_elems: int,
    s_lo: float,
) -> tuple[Any, dict[str, Any]]:
    """阶段 1：干净评估（τ_j 门控 + scale 补救）。返回 (report, 阶段记录)。"""
    rep = psf(
        seq, replace(_spec(cam, field, tau, pupil_n, chunk_elems, s_lo), structures=structure)
    )
    cov = float(rep.coverage.min())
    rec: dict[str, Any] = {
        "tau_j": tau,
        "scale": float(rep.scale[0]),
        "coverage_natural": cov if float(rep.scale[0]) == 1.0 else None,
        "coverage": min(cov, 1.0),
        "coverage_raw": cov,
        "delta": rep.delta,
        "size": rep.size,
        "na": {"min": float(rep.na.min()), "max": float(rep.na.max())},
        "issues": _issues_of(rep),
    }
    return rep, rec


def try_perturb(
    seq: Sequential,
    structure: int,
    field: tuple[float, float],
    cam: Camera,
    clean_rep,
    q_allow: float,
    cov_floor: float,
    pupil_n: int,
    chunk_elems: int,
    s_lo: float,
    rng: torch.Generator,
) -> tuple[Any, dict[str, Any] | str]:
    """阶段 2：扰动（q 退火，双条件验收）。返回 (最终 report, 扰动记录)。"""
    sub = seq.select(structure).scale(float(clean_rep.scale[0]))
    wl = cam.wavelengths
    na = float(clean_rep.na[0, 0, int(torch.tensor(wl).argmin())])
    epd = cast(InfiniteSource, sub[0]).epd
    f_eff, f_num = (epd / 2.0) / na, 1.0 / (2.0 * na)
    pose, info = sample_perturbation(sub, f_eff, f_num, min(wl) * _NM_TO_MM, rng)
    cov_clean = float(clean_rep.coverage.min())
    spec = replace(
        _spec(cam, field, 0.0, pupil_n, chunk_elems, s_lo),
        structures=0,
        delta=clean_rep.delta,
    )
    for q in _Q_LADDER:
        try:
            rep_q = psf(sub, replace(spec, errors=[_scale_pose(pose, q)]))
        except Exception:  # noqa: BLE001 — 扰动后无光能到达等视为验收失败
            continue
        cov_q = float(rep_q.coverage.min())
        if cov_clean - cov_q <= q_allow and cov_q >= cov_floor:
            return rep_q, {**info, "q_anneal": q}
    return clean_rep, "fallback"


def ship_sample(
    rep,
    cam: Camera,
    theta_deg: float,
    tput0_c: torch.Tensor,
    n_exp: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """光谱合成 + 能量定标 → ((64,64,3) float32, energy 元数据)。"""
    psf_c = channel_synthesize(rep.psf[0, 0], cam)  # (3,H,H)，逐通道 Σ≈1
    tput_c = channel_throughput(rep.throughput[0, 0], cam)  # (3,)
    ratio = tput_c / tput0_c.clamp_min(1e-300)
    e_c = energy_scale(cam.w_c, theta_deg, ratio, n_exp)  # (3,)
    img = (psf_c * e_c.view(3, 1, 1)).permute(1, 2, 0).to(torch.float32).clamp_min(0)
    energy = {
        "w_c": [round(float(x), 6) for x in cam.w_c],
        "cos_n": round(math.cos(math.radians(theta_deg)) ** n_exp, 6),
        "tput_ratio": [round(float(x), 6) for x in ratio],
        "E_c": [round(float(x), 6) for x in e_c],
    }
    tput_meta = [round(float(x), 6) for x in tput_c]
    return img, {"energy": energy, "throughput_channel": tput_meta}


# ── 主流程 ──


def _seed(lens_id: str, base: int) -> int:
    h = hashlib.sha256(lens_id.encode()).hexdigest()
    return (int(h[:8], 16) ^ base) % (2**31)


def _split(lens_id: str) -> str:
    h = int(hashlib.sha256(("split:" + lens_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 90 else ("val" if h < 95 else "test")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — 非 git 环境不阻断
        return "unknown"


def build_dataset(
    lib: Path,
    out: Path,
    *,
    lenses: list[str] | None = None,
    pupil_n: int = 65536,
    tau_lo: float = _TAU_LO,
    tau_cap: float = _TAU_CAP,
    s_lo: float = _SCALE_LO,
    perturb_prob: float = _PERTURB_PROB,
    q_min: float = _Q_MIN,
    q_max: float = _Q_MAX,
    cov_floor: float = _COV_FLOOR_PERTURB,
    theta_ref: float = _THETA_REF,
    e_ref: float = _E_REF,
    chunk_elems: int = 32_000_000,
    seed: int = 0,
    device: str = "cpu",
    limit_structures: int | None = None,
    limit_fields: int | None = None,
) -> list[dict[str, Any]]:
    """全库流水线；逐镜头写 meta/<镜头>.jsonl（存在则跳过 = 断点续跑）。"""
    cam_dir = Path(__file__).resolve().parents[1] / "data" / "cameras"
    cameras = load_cameras(cam_dir)
    lens_dirs = find_lenses(lib)
    if lenses:
        wanted = {s.replace("\\", "/") for s in lenses}
        lens_dirs = [
            d for d in lens_dirs if d.relative_to(lib).as_posix() in wanted or d.name in wanted
        ]
    (out / "psf").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    n_exp = math.log(e_ref) / math.log(math.cos(math.radians(theta_ref)))
    print(
        f"[cfg] lenses={len(lens_dirs)} cameras={len(cameras)} n_exp={n_exp:.4f} "
        f"(cos^{n_exp:.2f}θ: {theta_ref}°→{e_ref}) tau=[{tau_lo},{tau_cap}] s_lo={s_lo}"
    )

    for lens_dir in lens_dirs:
        lens_id = lens_dir.relative_to(lib).as_posix()
        safe_id = lens_id.replace("/", "_").replace(" ", "_")
        meta_file = out / "meta" / f"{safe_id}.jsonl"
        if meta_file.exists():
            print(f"[skip] {lens_id}: meta 已存在（断点续跑）")
            continue
        lens_seed = _seed(lens_id, seed)
        rng = torch.Generator().manual_seed(lens_seed)
        np_rng = np.random.default_rng(lens_seed)
        t0 = time.perf_counter()
        try:
            seq, target = load(lens_dir / next(p.name for p in lens_dir.glob("*.pth")))
            seq = seq.to(device)
        except Exception as e:  # noqa: BLE001 — 单个镜头失败不阻断全库
            print(f"[skip] {lens_id}: load failed: {e}")
            continue
        structures = select_structures(seq)
        if limit_structures:
            structures = structures[:limit_structures]
        h = _h_max(target.fov)
        cam = cameras[int(torch.randint(len(cameras), (), generator=rng))]
        print(
            f"[lens] {lens_id} structures={structures} h_max={h:.2f}° "
            f"camera={cam.name} (W={len(cam.wavelengths)})"
        )

        records: list[dict[str, Any]] = []
        n_saved, n_counter = 0, 0

        def save(
            img: torch.Tensor, rec: dict[str, Any], bin_tag: str, safe_id: str = safe_id
        ) -> None:
            nonlocal n_saved, n_counter
            name = f"{safe_id}__s{rec['structure']:02d}__{bin_tag}__n{n_counter:04d}.pt"
            n_counter += 1
            torch.save({"psf": img, "meta": rec}, out / "psf" / name)
            rec["file"] = f"psf/{name}"
            n_saved += 1

        def base_rec(
            s_k: int,
            field: tuple[float, float],
            bins: tuple[int, int],
            extra: bool,
            n_seq: int,
            lens_id: str = lens_id,
            cam: Camera = cam,
            lens_seed: int = lens_seed,
            commit: str = commit,
        ):
            return {
                "schema_version": _SCHEMA_VERSION,
                "git_commit": commit,
                "lens_id": lens_id,
                "structure": s_k,
                "camera": {
                    "name": cam.name,
                    "n_wavelengths": len(cam.wavelengths),
                    "w_c": [round(float(x), 6) for x in cam.w_c],
                },
                "field_deg": [round(field[0], 6), round(field[1], 6)],
                "bins": {"azimuth": bins[0], "radial": bins[1]},
                "fov_extrapolated": extra,
                "split": _split(lens_id),
                "seed_sample": (lens_seed + n_seq) % (2**31),
            }

        for s_k in structures:
            # ── 轴上参考（干净阶段兼作 T(0) 基准；入集样本仍可扰动，孪生出货） ──
            tau0 = tau_j(0.0, np_rng, tau_lo, tau_cap)
            try:
                rep0, rec0 = make_clean(seq, s_k, (0.0, 0.0), cam, tau0, pupil_n, chunk_elems, s_lo)
            except Exception as e:  # noqa: BLE001
                rec = base_rec(s_k, (0.0, 0.0), (-1, -1), False, n_counter)
                rec["dropped"] = f"axis error: {type(e).__name__}: {e}"
                records.append(rec)
                continue
            rec = base_rec(s_k, (0.0, 0.0), (-1, -1), False, n_counter)
            rec.update(rec0)
            rec["rotation_group"] = f"{lens_id}#s{s_k}#axis"
            if any(i == Issue.THRESHOLD_UNREACHABLE for i, *_ in rep0.issues):
                rec["dropped"] = "axis_threshold_unreachable"
                records.append(rec)
                continue  # 无 T(0) 锚点 → 跳过整个结构
            tput0_c = channel_throughput(rep0.throughput[0, 0], cam)
            img, en = ship_sample(rep0, cam, 0.0, tput0_c, n_exp)
            rec.update(en)
            rec["perturb"] = "none"
            rec["q_j"] = None
            save(img, rec, "axis")
            records.append(rec)
            # 轴上扰动孪生（独立抽奖）
            if float(torch.rand((), generator=rng)) < perturb_prob:
                qj = q_j(0.0, np_rng, q_min, q_max)
                rep_t, pinfo = try_perturb(
                    seq, s_k, (0.0, 0.0), cam, rep0, qj, cov_floor, pupil_n, chunk_elems, s_lo, rng
                )
                if pinfo != "fallback":
                    img_t, en_t = ship_sample(rep_t, cam, 0.0, tput0_c, n_exp)
                    rec_t = base_rec(s_k, (0.0, 0.0), (-1, -1), False, n_counter)
                    rec_t.update(rec0)
                    rec_t.update(en_t)
                    rec_t["perturb"] = pinfo
                    rec_t["q_j"] = qj
                    rec_t["coverage"] = min(float(rep_t.coverage.min()), 1.0)
                    rec_t["coverage_raw"] = float(rep_t.coverage.min())
                    rec_t["rotation_group"] = f"{lens_id}#s{s_k}#axis"
                    rec_t["issues"] = _issues_of(rep_t)
                    save(img_t, rec_t, "axisP")
                    records.append(rec_t)

            # ── 48 视场 ──
            fields = sample_fields(h, rng)
            if limit_fields:
                fields = fields[:limit_fields]
            for fx, fy, i, j, extra in fields:
                theta = math.hypot(fx, fy)
                th_hat = theta / h
                tau = tau_j(th_hat, np_rng, tau_lo, tau_cap)
                rec = base_rec(s_k, (fx, fy), (i, j), extra, n_counter)
                rec["rotation_group"] = f"{lens_id}#s{s_k}#r{j}"
                try:
                    rep, crec = make_clean(seq, s_k, (fx, fy), cam, tau, pupil_n, chunk_elems, s_lo)
                except Exception as e:  # noqa: BLE001
                    rec["dropped"] = f"error: {type(e).__name__}: {e}"
                    records.append(rec)
                    continue
                rec.update(crec)
                if any(i2 == Issue.THRESHOLD_UNREACHABLE for i2, *_ in rep.issues) or any(
                    i2 == Issue.NO_ALIVE_RAYS for i2, *_ in rep.issues
                ):
                    rec["dropped"] = "threshold_unreachable"
                    records.append(rec)
                    continue
                # 逐样本独立扰动抽奖
                rec["perturb"], rec["q_j"] = "none", None
                if float(torch.rand((), generator=rng)) < perturb_prob:
                    qj = q_j(th_hat, np_rng, q_min, q_max)
                    rep_p, pinfo = try_perturb(
                        seq, s_k, (fx, fy), cam, rep, qj, cov_floor, pupil_n, chunk_elems, s_lo, rng
                    )
                    rec["perturb"], rec["q_j"] = pinfo, qj
                    rep = rep_p
                    rec["coverage"] = min(float(rep.coverage.min()), 1.0)
                    rec["coverage_raw"] = float(rep.coverage.min())
                    rec["issues"] = _issues_of(rep)
                img, en = ship_sample(rep, cam, theta, tput0_c, n_exp)
                rec.update(en)
                save(img, rec, f"a{i:02d}r{j:02d}")
                records.append(rec)

        meta_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n",
            encoding="utf-8",
        )
        dt = time.perf_counter() - t0
        print(f"[done] {lens_id}: {n_saved} saved, {dt:.1f}s ({dt / max(1, n_saved):.2f}s/sample)")

    # ── 合并 metadata.jsonl（含断点续跑镜头） ──
    all_records: list[dict[str, Any]] = []
    for mf in sorted((out / "meta").glob("*.jsonl")):
        for line in mf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {mf.name}: 跳过损坏行: {e}")
    (out / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in all_records) + "\n",
        encoding="utf-8",
    )
    print(f"[all] {len([r for r in all_records if 'file' in r])} samples → {out}")
    return all_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, required=True, help="镜头库根目录")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--lenses", nargs="*", default=None, help="只跑这些镜头（相对路径或目录名）")
    ap.add_argument("--pupil", type=int, default=65536)
    ap.add_argument("--tau-lo", type=float, default=_TAU_LO)
    ap.add_argument("--tau-cap", type=float, default=_TAU_CAP)
    ap.add_argument("--s-lo", type=float, default=_SCALE_LO)
    ap.add_argument("--perturb-prob", type=float, default=_PERTURB_PROB)
    ap.add_argument("--q-min", type=float, default=_Q_MIN)
    ap.add_argument("--q-max", type=float, default=_Q_MAX)
    ap.add_argument("--cov-floor", type=float, default=_COV_FLOOR_PERTURB)
    ap.add_argument("--theta-ref", type=float, default=_THETA_REF, help="能量衰减标定点角度 (deg)")
    ap.add_argument("--e-ref", type=float, default=_E_REF, help="能量衰减标定点值")
    ap.add_argument("--chunk-elems", type=int, default=32_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--limit-structures", type=int, default=None, help="pilot：每镜头只跑前 k 个结构"
    )
    ap.add_argument("--limit-fields", type=int, default=None, help="pilot：每结构只跑前 k 个视场")
    a = ap.parse_args()
    build_dataset(
        a.lib,
        a.out,
        lenses=a.lenses,
        pupil_n=a.pupil,
        tau_lo=a.tau_lo,
        tau_cap=a.tau_cap,
        s_lo=a.s_lo,
        perturb_prob=a.perturb_prob,
        q_min=a.q_min,
        q_max=a.q_max,
        cov_floor=a.cov_floor,
        theta_ref=a.theta_ref,
        e_ref=a.e_ref,
        chunk_elems=a.chunk_elems,
        seed=a.seed,
        device=a.device,
        limit_structures=a.limit_structures,
        limit_fields=a.limit_fields,
    )


if __name__ == "__main__":
    main()
