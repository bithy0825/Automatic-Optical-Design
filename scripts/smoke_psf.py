"""analysis/psf 重设计冒烟测试：平凸单透镜 + pth 往返 + 阈值双模式 + 误差注入。

用法：python scripts/smoke_psf.py
"""

import tempfile
from pathlib import Path

import torch

from analysis import Issue, PsfSpec, mtf, psf, psf_from_checkpoint, strehl, wavefront
from optimization.utils import build_sequential, load_config, save

CFG = """\
[target]
fov = [[0, 3], [0, 0]]
F = 10.0
effl = 96.0
wavelength = [486, 555, 656]

[ga]
population = 4
topk = 2
generation = 1

[[component]]
type = "source"
pupil = { method = "fibonacci", region = "disk", count = 256 }
field = { method = "uniform", region = "rect", count = [3, 1] }
wavel = { method = "uniform", region = "line", count = 3 }

[[component]]
type = "refractor"
shape = "sphere"
curvature = { method = "raw", value = 0.02 }
diameter = { method = "raw", value = 10.0 }
material = { method = "raw", value = "N-BK7" }

[[component]]
type = "gap"
thickness = { method = "raw", value = 5.0 }

[[component]]
type = "refractor"
shape = "disk"
diameter = { method = "raw", value = 10.0 }
material = { method = "raw", value = "air" }

[[component]]
type = "gap"
thickness = { method = "raw", value = 92.0 }

[[component]]
type = "sensor"
diameter = { method = "raw", value = 5.0 }
"""


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        raise AssertionError(name)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    cfg_path = tmp / "singlet.toml"
    cfg_path.write_text(CFG, encoding="utf-8")
    cfg = load_config(str(cfg_path))
    seq = build_sequential(cfg)
    pth = tmp / "singlet.pth"
    save(seq, cfg, pth)

    # ── 1. 缺省调用（pth 一站式，内嵌波长 3 个，64×64，轴上，第 0 结构） ──
    rep = psf_from_checkpoint(pth)
    print(rep.psf.shape, "delta=", rep.delta, "coverage=", rep.coverage.flatten().tolist())
    check("shape", tuple(rep.psf.shape) == (1, 1, 3, 64, 64))
    check(
        "psf sum=1",
        torch.allclose(
            rep.psf.sum(dim=(-2, -1)), torch.ones(1, 1, 3, dtype=torch.float64), atol=1e-9
        ),
    )
    check("scale=1", rep.scale.item() == 1.0)
    check("structures", rep.structures == (0,))
    check("wavelengths", rep.wavelengths.tolist() == [486.0, 555.0, 656.0])
    check(
        "coverage in (0,1]",
        bool((rep.coverage > 0).all() and (rep.coverage <= 1.05).all()),
        f"{rep.coverage.flatten().tolist()}",
    )

    # ── 2. Parseval 覆盖率健全性：近衍射极限单透镜（epd=1，13.1×Airy 窗口） ──
    # 理论包围能量 ≈ 0.988（Airy），容许 ±0.02（离散底噪 + 方窗角部）
    from scripts.exp_coverage2 import build

    seq_dl = build(1.0, 93.0)
    rep_dl = psf(seq_dl, PsfSpec(size=64, wavelengths=(555.0,)))
    cov_dl = rep_dl.coverage.flatten()[0].item()
    print("diffraction-limited coverage=", cov_dl)
    check("coverage≈Airy 0.988", abs(cov_dl - 0.988) < 0.02, f"{cov_dl:.4f}")

    # ── 3. scale 补救：τ=0.95 应缩小系统并恰好达标 ──
    rep_s = psf_from_checkpoint(pth, PsfSpec(energy_threshold=0.95, remedy="scale"))
    s = rep_s.scale.item()
    cov_s = rep_s.coverage.min().item()
    print("scale factor=", s, "worst coverage=", cov_s, "issues=", rep_s.issues)
    check("scale < 1", s < 1.0)
    if not any(i[0] == Issue.THRESHOLD_UNREACHABLE for i in rep_s.issues):
        check("coverage in band", 0.95 - 1e-6 <= cov_s <= 0.95 + 0.01 + 1e-6, f"{cov_s}")

    # ── 4. grow 补救：τ=0.95 应扩大窗口并达标 ──
    rep_g = psf_from_checkpoint(pth, PsfSpec(energy_threshold=0.95, remedy="grow"))
    cov_g = rep_g.coverage.min().item()
    print("grow size=", rep_g.size, "worst coverage=", cov_g)
    check("grow size >= 64", rep_g.size >= 64)
    check("grow coverage >= τ", cov_g >= 0.95 - 1e-6, f"{cov_g}")

    # ── 5. 多结构 + 显式波长 + 视场角 ──
    rep_m = psf_from_checkpoint(
        pth, PsfSpec(structures=[0, 2], field=(1.5, 0.0), wavelengths=(555.0,), size=64)
    )
    check("multi shape", tuple(rep_m.psf.shape) == (2, 1, 1, 64, 64), str(tuple(rep_m.psf.shape)))
    check("multi scale", rep_m.scale.tolist() == [1.0, 1.0])
    check("multi structures", rep_m.structures == (0, 2))
    check("multi wl", rep_m.wavelengths.tolist() == [555.0])

    # ── 6. 误差注入（perturb 声明，逐链重编译） ──
    rep_e = psf_from_checkpoint(
        pth, PsfSpec(errors=[{"type": "pose", "first": 2, "dx": 0.02}], wavelengths=(555.0,))
    )
    shift = (rep_e.centers - rep_m.centers[0]).abs().max().item()
    print("perturbed center shift (mm)=", shift)
    check("perturb moved image", shift > 1e-4, f"{shift}")

    # ── 7. 原始 callback 注入 ──
    def null_cb(comp, flow, _i):
        return flow

    rep_c = psf_from_checkpoint(
        pth, PsfSpec(structures=0, field=(1.5, 0.0), wavelengths=(555.0,), callback=null_cb)
    )
    check(
        "callback no-op identical",
        bool(torch.equal(rep_c.psf, rep_m.psf[0:1])),
    )

    # ── 8. 派生指标消费 PsfReport ──
    m = mtf(rep)
    w = wavefront(rep)
    st = strehl(rep)
    check("mtf shape", tuple(m.mtf.shape) == (1, 1, 3, 64, 64))
    check("strehl in (0,1]", bool((st > 0).all() and (st <= 1).all()))
    print("wavefront rms=", w.rms.flatten().tolist(), "strehl=", st.flatten().tolist())

    # ── 9. 底层原语：select / scale / scale_（where/where_ 语义） ──
    from typing import cast

    from torch import Tensor

    from component import Gap, InfiniteSource, Refractor
    from optimization.utils import load

    seq_l, _target = load(pth)
    sub = seq_l.select([1, 3])
    check("select population", sub.population == 2)
    # scale 返回新对象，原链不变
    one = sub.select(0)
    scaled = one.scale(0.5)
    src_o = one[0]
    assert isinstance(src_o, InfiniteSource)
    src_s = scaled[0]
    assert isinstance(src_s, InfiniteSource)
    check("epd scaled", abs(src_s.epd - src_o.epd * 0.5) < 1e-12, f"{src_s.epd} vs {src_o.epd}")
    check("scale 返回新对象", scaled is not one)
    check("scale 不改原链", abs(src_o.epd - 9.6) < 1e-12)
    # scale_ 原地修改
    one.scale_(0.5)
    check("scale_ 原地生效", abs(cast(InfiniteSource, one[0]).epd - 4.8) < 1e-12)
    refr_s = [c for c in scaled if isinstance(c, Refractor)][0]
    refr_o = [c for c in seq_l.select(0) if isinstance(c, Refractor)][0]
    cur_s = cast(Tensor, refr_s.shape.Curvature)
    cur_o = cast(Tensor, refr_o.shape.Curvature)
    check("curvature scaled", bool(torch.allclose(cur_s, cur_o / 0.5)))
    gaps_s = [c for c in scaled if isinstance(c, Gap)]
    gaps_o = [c for c in seq_l.select(0) if isinstance(c, Gap)]
    th_s = cast(Tensor, gaps_s[0].Thickness)
    th_o = cast(Tensor, gaps_o[0].Thickness)
    check("gap scaled", bool(torch.allclose(th_s, th_o * 0.5)))

    # 逐类实现：Shape / Material 也具备同名方法（where/where_ 语义）
    from materials import Material

    sh = refr_o.shape
    sh_scaled = sh.scale(0.5)
    check("shape.scale 返回新对象", sh_scaled is not sh)
    check(
        "shape diameter scaled",
        bool(torch.allclose(cast(Tensor, sh_scaled.Diameter), cast(Tensor, sh.Diameter) * 0.5)),
    )
    sh.scale_(0.5)
    check(
        "shape.scale_ 原地生效",
        bool(torch.allclose(cast(Tensor, sh.Diameter), cast(Tensor, sh_scaled.Diameter))),
    )
    mat = refr_o.transmitted
    assert isinstance(mat, Material)
    msel = mat.select([0, 0])
    check("material select 返回新对象", msel is not mat and msel.Indices.shape[0] == 2)
    mscaled = mat.scale(0.3)
    check(
        "material scale 空操作但返回新对象",
        mscaled is not mat and bool((mscaled.Indices == mat.Indices).all()),
    )

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
