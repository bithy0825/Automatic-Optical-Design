"""PSF 数据集生成器：镜头库 → 64×64×3 归一化 PSF 数据集。

设计文档：``docs/psf-dataset.md``（流程的单一事实源，本文件是实现）。

流程概要：逐镜头加载 checkpoint（256 结构）→ 参数空间 FPS 选 4 结构
（固定第 0 个）→ 每结构随机 1 台相机（3 代表波长 + 权重）→ 8 档抖动
视场 → 每视场一次干净评估（τ 门控 + scale 补救，禁 grow）→ 10% 概率
进入扰动子流程（幅度与结构挂钩，能量不足按 q 退火，兜底退回干净样本）。

用法::

    python scripts/build_psf_dataset.py --lib <库根> --out <输出目录> \
        [--lenses Zebase/A_001 ...] [--fields 8] [--pupil 65536] [--tau 0.9]

产出：``<out>/psf/<样本名>.pt``（``{"psf": (64,64,3) float32, ...元数据}``，
逐通道 Σ=1 归一化）+ ``<out>/metadata.jsonl``（逐样本一行）。
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import Issue, PsfSpec, psf
from component import Gap, InfiniteSource, Refractor, Sensor, Sequential, Stop
from optimization.target import FOV
from optimization.utils import load
from perturb import Pose
from sampling import SampleOptions

_NM_TO_MM = 1e-6
_TAU_DEFAULT = 0.90
_SCALE_LO = 0.5
_Q_LADDER = (1.0, 0.5, 0.25, 0.125)
_PERTURB_WEIGHTS = (
    ("element", 0.30),
    ("single", 0.25),
    ("defocus", 0.20),
    ("spacing", 0.15),
    ("sensor_tilt", 0.10),
)


# ── 镜头库与相机库 ──


def find_lenses(lib: Path) -> list[Path]:
    """库根 → 全部非废弃镜头目录（含唯一 .pth 的目录；跳过 ``_rejected``）。"""
    out = []
    for pth in sorted(lib.rglob("*.pth")):
        if "_rejected" in pth.parts:
            continue
        out.append(pth.parent)
    return sorted(set(out))


def load_cameras(directory: Path) -> list[dict[str, Any]]:
    """相机库 → [(名称, 3 代表波长 nm, 3 通道权重)]，代表波长 = 逐通道峰值波长。"""
    cams = []
    for f in sorted(directory.glob("*.pt")):
        d = torch.load(f, weights_only=True, map_location="cpu")
        wl, rgb = d["wavelengths"], d["rgb"]
        peaks = rgb.argmax(dim=0)  # (3,)
        cams.append(
            {
                "camera": d.get("camera", f.stem),
                "wavelengths": tuple(float(wl[i]) for i in peaks),
                "weights": tuple(float(rgb[i, c]) for c, i in enumerate(peaks)),
            }
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


# ── 结构选择：参数空间 FPS ──


def _param_matrix(seq: Sequential, pool: int) -> torch.Tensor:
    """state_dict 前 pool 行 → (pool, D) 参数矩阵（键序固定，逐键拉平）。"""
    cols = []
    for _k, t in sorted(seq.state_dict().items()):
        t = t.detach().to(torch.float64)
        cols.append(t[:pool].reshape(min(pool, t.shape[0]), -1))
    return torch.cat(cols, dim=1)


def select_structures(seq: Sequential, pool: int = 16, picks: int = 3) -> tuple[int, ...]:
    """固定第 0 个 + 前 pool 个里 FPS 选 picks 个（min-max 归一化参数空间）。"""
    pool = min(pool, seq.population)
    v = _param_matrix(seq, pool)
    span = v.max(dim=0).values - v.min(dim=0).values
    v = (v - v.min(dim=0).values) / span.clamp_min(1e-12)  # 零跨度维退化无害
    d = torch.cdist(v, v)  # (pool, pool)

    chosen = [0]
    while len(chosen) <= picks and len(chosen) < pool:
        d_min = d[:, chosen].min(dim=1).values
        d_min[chosen] = -1.0
        chosen.append(int(d_min.argmax()))
    return tuple(chosen)


# ── 扰动采样 ──


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


def _surfaces(chain: Sequential) -> list[int]:
    return [i for i, c in enumerate(chain) if isinstance(c, (Refractor, Stop, Sensor))]


def sample_perturbation(
    chain: Sequential, f_eff: float, f_number: float, lam_min_mm: float, rng: torch.Generator
) -> tuple[Pose, dict[str, Any]]:
    """抽一次扰动（类型 + 目标 + 实现值），幅度与结构挂钩（文档 §6.1/6.2）。

    返回 (Pose 声明, 描述字典)；实现值以常数物化，幅度退火只乘 q。
    """
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

    def s(v: float | None) -> float | None:
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


# ── 单样本两阶段流程 ──


def _spec_base(pupil_n: int, field: float, wavelengths: tuple[float, ...]) -> PsfSpec:
    return PsfSpec(
        field=(field, 0.0),
        wavelengths=wavelengths,
        pupil=SampleOptions(method="fibonacci", region="disk", count=pupil_n),
        size=64,
        delta=None,
    )


def make_sample(
    seq: Sequential,
    structure: int,
    field: float,
    camera: dict[str, Any],
    *,
    pupil_n: int,
    tau: float,
    perturb: bool,
    rng: torch.Generator,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """单视场样本：干净评估（scale 补救）→（可选）扰动评估（q 退火）。

    返回 ``(样本 | None, 记录)``；样本 None = 丢弃（原因在记录里）。
    """
    wl = camera["wavelengths"]
    base = _spec_base(pupil_n, field, wl)
    rec: dict[str, Any] = {
        "structure": structure,
        "field_deg": [field, 0.0],
        "camera": camera,
        "perturb": "none",
    }

    # ── 阶段 1：干净评估，τ 门控 + scale 补救 ──
    rep = psf(
        seq,
        replace(
            base,
            structures=structure,
            energy_threshold=tau,
            remedy="scale",
            scale_bounds=(_SCALE_LO, 1.0),
        ),
    )
    rec["scale"] = float(rep.scale[0])
    rec["coverage_clean"] = float(rep.coverage.min())
    rec["issues_clean"] = [int(i) for i, *_ in rep.issues]
    if any(i == Issue.THRESHOLD_UNREACHABLE for i, *_ in rep.issues):
        rec["dropped"] = "threshold_unreachable"
        return None, rec
    final = rep

    # ── 阶段 2：扰动（q 退火；不再缩放结构） ──
    if perturb:
        sub = seq.select(structure).scale(rec["scale"])
        na = float(rep.na[0, 0, int(torch.tensor(wl).argmin())])
        epd = cast(InfiniteSource, sub[0]).epd
        f_eff, f_num = (epd / 2.0) / na, 1.0 / (2.0 * na)
        pose, info = sample_perturbation(sub, f_eff, f_num, min(wl) * _NM_TO_MM, rng)
        for q in _Q_LADDER:
            rep_q = psf(
                sub,
                replace(
                    base,
                    structures=0,
                    energy_threshold=0.0,
                    delta=rep.delta,
                    errors=[_scale_pose(pose, q)],
                ),
            )
            if float(rep_q.coverage.min()) >= tau:
                final, rec["perturb"], rec["q"] = rep_q, info, q
                break
        else:
            rec["perturb"], rec["q"] = "fallback", 0.0  # 退回干净样本
        rec["coverage"] = float(final.coverage.min())

    # ── 出样：64×64×3，逐通道 Σ=1 归一化 ──
    img = final.psf[0, 0].permute(1, 2, 0).to(torch.float32).clamp_min(0)  # (H,H,3)
    img = img / img.sum(dim=(0, 1), keepdim=True).clamp_min(1e-300)
    rec.update(
        delta=final.delta,
        coverage=float(final.coverage.min()),
        na=final.na[0, 0].tolist(),
        throughput=final.throughput[0, 0].tolist(),
        issues=[int(i) for i, *_ in final.issues],
    )
    return {"psf": img}, rec


# ── 主流程 ──


def _seed(lens_id: str, base: int) -> int:
    h = hashlib.sha256(lens_id.encode()).hexdigest()
    return (int(h[:8], 16) ^ base) % (2**31)


def build_dataset(
    lib: Path,
    out: Path,
    *,
    lenses: list[str] | None = None,
    fields: int = 8,
    pupil_n: int = 65536,
    tau: float = _TAU_DEFAULT,
    perturb_prob: float = 0.1,
    pool: int = 16,
    picks: int = 3,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """全库流水线；返回全部样本记录（同时落盘 psf/*.pt 与 metadata.jsonl）。"""
    cam_dir = Path(__file__).resolve().parents[1] / "data" / "cameras"
    cameras = load_cameras(cam_dir)
    lens_dirs = find_lenses(lib)
    if lenses:
        wanted = {s.replace("\\", "/") for s in lenses}
        lens_dirs = [
            d for d in lens_dirs if d.relative_to(lib).as_posix() in wanted or d.name in wanted
        ]
    (out / "psf").mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for lens_dir in lens_dirs:
        lens_id = lens_dir.relative_to(lib).as_posix()
        pth = next(lens_dir.glob("*.pth"))
        rng = torch.Generator().manual_seed(_seed(lens_id, seed))
        t0 = time.perf_counter()
        try:
            seq, target = load(pth)
            seq = seq.to(device)
        except Exception as e:  # noqa: BLE001 — 单个镜头失败不阻断全库
            print(f"[skip] {lens_id}: load failed: {e}")
            continue
        structures = select_structures(seq, pool, picks)
        h = _h_max(target.fov)
        print(f"[lens] {lens_id} structures={structures} h_max={h:.2f}°")

        n_saved = 0
        for s_k in structures:
            cam = cameras[int(torch.randint(len(cameras), (), generator=rng))]
            for k in range(fields):
                fx = (k + 0.5 + (float(torch.rand((), generator=rng)) - 0.5)) * h / fields
                do_perturb = float(torch.rand((), generator=rng)) < perturb_prob
                try:
                    sample, rec = make_sample(
                        seq, s_k, fx, cam, pupil_n=pupil_n, tau=tau, perturb=do_perturb, rng=rng
                    )
                except Exception as e:  # noqa: BLE001 — 记档后继续
                    rec = {
                        "structure": s_k,
                        "field_deg": [fx, 0.0],
                        "dropped": f"error: {type(e).__name__}: {e}",
                    }
                    sample = None
                rec.update(lens_id=lens_id, seed=_seed(lens_id, seed))
                if sample is not None:
                    name = f"{lens_id.replace('/', '_')}__s{s_k}__f{k}.pt"
                    torch.save({**sample, "meta": rec}, out / "psf" / name)
                    rec["file"] = f"psf/{name}"
                    n_saved += 1
                records.append(rec)
        dt = time.perf_counter() - t0
        print(
            f"[done] {lens_id}: {n_saved}/{len(structures) * fields} saved, "
            f"{dt:.1f}s ({dt / max(1, len(structures) * fields):.2f}s/sample)"
        )

    lines = [json.dumps(rec, ensure_ascii=False, default=str) for rec in records]
    (out / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[all] {len([r for r in records if 'file' in r])} samples → {out}")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lib", type=Path, required=True, help="镜头库根目录")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--lenses", nargs="*", default=None, help="只跑这些镜头（相对路径或目录名）")
    ap.add_argument("--fields", type=int, default=8)
    ap.add_argument("--pupil", type=int, default=65536)
    ap.add_argument("--tau", type=float, default=_TAU_DEFAULT)
    ap.add_argument("--perturb-prob", type=float, default=0.1)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--picks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu / cuda（GPU 约 7× 加速，显存 <1GB）")
    a = ap.parse_args()
    build_dataset(
        a.lib,
        a.out,
        lenses=a.lenses,
        fields=a.fields,
        pupil_n=a.pupil,
        tau=a.tau,
        perturb_prob=a.perturb_prob,
        pool=a.pool,
        picks=a.picks,
        seed=a.seed,
        device=a.device,
    )


if __name__ == "__main__":
    main()
