"""integrate 优化 A/B 验证：--save 保存参考输出，--check 对比当前实现。

覆盖轴上/边缘视场 × 干净/扰动（扰动制造死亡光线，验证掩码语义等价）。
判定：max abs diff 与 rel diff（相对峰值）均 < 1e-10（float64 求值序噪声
应在 1e-15 量级，1e-10 已非常宽松但仍能抓住任何真实错误）。
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import PsfSpec, psf
from optimization.utils import load
from perturb import Pose
from sampling import SampleOptions

_REF = Path(__file__).resolve().parent / "out" / "integrate_ref.pt"


def _run() -> dict[str, torch.Tensor]:
    d = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("E:/workspace/Automatic-Optical-Design-Lib/Zebase/A_001")
    )
    seq, _target = load(d / next(p.name for p in d.glob("*.pth")))
    seq = seq.to("cuda")
    cam = torch.load(
        Path(__file__).resolve().parents[1] / "data" / "cameras" / "Canon_20D.pt",
        weights_only=True,
    )
    wl = tuple(float(w) for w in cam["wavelengths"])
    base = dict(
        structures=0,
        wavelengths=wl,
        pupil=SampleOptions(method="fibonacci", region="disk", count=4096),
        size=64,
        delta=None,
        energy_threshold=0.0,
        remedy="scale",
    )
    out: dict[str, torch.Tensor] = {}
    cases = {
        "axis_clean": dict(field=(0.0, 0.0)),
        "edge_clean": dict(field=(12.3, 8.7)),
        "edge_perturbed": dict(
            field=(12.3, 8.7), errors=[Pose(first=2, dx=0.01, dy=-0.005, alpha=1e-4)]
        ),
    }
    for name, kw in cases.items():
        rep = psf(seq, PsfSpec(**{**base, **kw}))
        out[f"{name}.psf"] = rep.psf.cpu()
        out[f"{name}.coverage"] = rep.coverage.cpu()
    return out


def main() -> None:
    mode = sys.argv[1]
    cur = _run()
    if mode == "--save":
        _REF.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cur, _REF)
        print(f"saved → {_REF}")
        return
    ref = torch.load(_REF, weights_only=True)
    worst = 0.0
    for k, v in cur.items():
        r = ref[k].to(v.dtype)
        diff = (v - r).abs().max().item()
        rel = diff / max(r.abs().max().item(), 1e-300)
        worst = max(worst, rel)
        print(f"{k:24s} max_abs={diff:.3e} max_rel={rel:.3e}")
    print("PASS" if worst < 1e-10 else "FAIL", f"(worst rel {worst:.3e})")


if __name__ == "__main__":
    main()
