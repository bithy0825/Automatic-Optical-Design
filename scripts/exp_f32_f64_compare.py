"""float32 vs float64 对拍验证 + 5080 性能基准（一次性 QA 工具）。

三重校验：
1. 回归：新代码 float64 结果 vs 已提取数据集的 f64 元数据（A_001 s0 轴上，
   coverage/delta 应逐位接近）——证明相位分裂重构没有改变 f64 数学语义；
2. 精度：同工况 float32 vs float64（轴上 + 离轴视场），覆盖率差、PSF 相对
   误差、τ 门控决策一致性；
3. 性能：同工况两 dtype 计时（pupil 可缩小以控制 f64 耗时）。

用法::

    python scripts/exp_f32_f64_compare.py [--lens Zebase/A_001] [--pupil 65536] [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import PsfSpec, psf
from optimization.utils import load
from sampling import SampleOptions
from scripts.build_psf_dataset_v2 import load_cameras

_LIB = Path.home() / "workspace" / "Automatic-Optical-Design-Lib"
_PSF = Path.home() / "workspace" / "Automatic-Optical-Design-PSF"


def run(seq, cam_wl, field, tau, pupil_n, dtype, device):
    spec = PsfSpec(
        structures=0,
        field=field,
        wavelengths=cam_wl,
        pupil=SampleOptions(method="fibonacci", region="disk", count=pupil_n),
        size=64,
        energy_threshold=tau,
        remedy="scale",
        scale_bounds=(0.3, 1.0),
        chunk_elems=64_000_000,
        dtype=dtype,
    )
    t0 = time.perf_counter()
    rep = psf(seq, spec)
    if device == "cuda":
        torch.cuda.synchronize()
    return rep, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lens", default="Zebase/A_001")
    ap.add_argument("--pupil", type=int, default=65536)
    ap.add_argument("--quick", action="store_true", help="只做精度对拍（pupil 减半，f64 省时）")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False

    cam_dir = Path(__file__).resolve().parents[1] / "data" / "cameras"
    cam = load_cameras(cam_dir)[0]
    pth = next((_LIB / a.lens).glob("*.pth"))
    seq, _ = load(pth)
    seq = seq.to(device)
    pupil_n = a.pupil // 2 if a.quick else a.pupil

    # ── 1. f64 回归：对比已提取数据集（旧全 f64 代码）的 A_001 s0 轴上元数据 ──
    meta_file = _PSF / a.lens.replace("/", "_").replace("Zebase_", "Zebase_")  # noqa: F841
    meta_path = _PSF / "Zebase" / "meta" / f"{a.lens.split('/')[-1]}.jsonl"
    if meta_path.exists():
        first = json.loads(meta_path.read_text().splitlines()[0])
        print(f"[ref] 旧 f64 数据集: coverage={first['coverage_raw']:.10f} "
              f"delta={first['delta']:.12f} tau_j={first['tau_j']:.6f}")
    else:
        first = None
        print("[ref] 未找到参考元数据，跳过回归对比")

    # ── 2. f32 vs f64 精度对拍（轴上 + 离轴） ──
    fields = [(0.0, 0.0), (5.23462, 0.0)]
    for field in fields:
        rep64, t64 = run(seq, cam.wavelengths, field, 0.90, pupil_n, torch.float64, device)
        rep32, t32 = run(seq, cam.wavelengths, field, 0.90, pupil_n, torch.float32, device)
        cov_d = (rep32.coverage.double() - rep64.coverage).abs().max()
        na_d = (rep32.na.double() - rep64.na).abs().max()
        # PSF 相对误差（逐张 Σ=1 后比峰值归一）
        p64, p32 = rep64.psf[0, 0], rep32.psf[0, 0].double()
        rel_l2 = ((p32 - p64).square().sum() / p64.square().sum()).sqrt()
        peak = p64.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-300)
        rel_max = ((p32 - p64).abs() / peak).amax()
        d_delta = abs(rep32.delta - rep64.delta) / rep64.delta
        d_scale = abs(float(rep32.scale[0]) - float(rep64.scale[0]))
        print(
            f"[cmp] field={field} dtype=f32 vs f64: "
            f"|Δcov|max={float(cov_d):.2e} |Δna|max={float(na_d):.2e} "
            f"PSF relL2={float(rel_l2):.2e} relMax={float(rel_max):.2e} "
            f"|Δdelta|/delta={d_delta:.2e} |Δscale|={d_scale:.2e} "
            f"t32={t32:.2f}s t64={t64:.2f}s speedup={t64 / max(t32, 1e-9):.1f}x"
        )
        if first is not None and field == (0.0, 0.0):
            dcov = abs(float(rep64.coverage.min()) - first["coverage_raw"])
            ddel = abs(rep64.delta - first["delta"]) / first["delta"]
            print(f"[reg] 新 f64 vs 旧 f64 数据集: |Δcov|={dcov:.2e} |Δdelta|/delta={ddel:.2e}")

    # ── 3. τ 门控决策一致性（scale 补救是否同判） ──
    for tau in (0.80, 0.90, 0.95):
        r64, _ = run(seq, cam.wavelengths, (5.23462, 0.0), tau, pupil_n, torch.float64, device)
        r32, _ = run(seq, cam.wavelengths, (5.23462, 0.0), tau, pupil_n, torch.float32, device)
        same = (float(r64.scale[0]) == 1.0) == (float(r32.scale[0]) == 1.0)
        print(
            f"[gate] tau={tau}: scale64={float(r64.scale[0]):.6f} scale32={float(r32.scale[0]):.6f} "
            f"cov64={float(r64.coverage.min()):.6f} cov32={float(r32.coverage.min()):.6f} "
            f"决策一致={same}"
        )


if __name__ == "__main__":
    main()
