"""缩放对比可视化：cox1_02 原始 vs 1/10 缩放（同一物理窗口，验证 Airy 斑不变性）。

物理预期：缩放 ×0.1 后几何像差 ×0.1（波像差 ~0.45λ → ~0.045λ），
f/# 不变 ⇒ Airy 斑物理尺寸不变；同一 Nyquist delta ⇒ 两图网格完全同尺度。

运行：PYTHONPATH=. python scripts/viz_scale_compare.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_lib_psf import LOG_VMIN, OUT, _to_np, mtf_diffraction, rgb_composite

from analysis import PsfSpec, mtf, psf, strehl, wavefront
from optimization.utils import load

PTH = Path(
    r"E:/workspace/Automatic-Optical-Design-Lib/Navigation"
    r"/Photographic lenses - prime/len/cox1_02/cox1_02_17130.pth"
)
SIZE = 128
SCALE = 0.1


def main() -> None:
    seq, _ = load(PTH)
    sub = seq.select(0)
    scaled = sub.scale(SCALE)

    reps = {}
    for tag, s in (("original", sub), (f"scaled x{SCALE}", scaled)):
        reps[tag] = psf(s, PsfSpec(size=SIZE))
        r = reps[tag]
        st = _to_np(strehl(r)[0, 0])
        wf = wavefront(r)
        rms_w = _to_np(wf.rms[0, 0]) / _to_np(r.wavelengths) * 1e6
        cov = _to_np(r.coverage[0, 0])
        print(
            f"{tag:<14} delta={r.delta * 1e3:.4f} um/px  "
            f"strehl={[round(float(x), 4) for x in st]}  "
            f"rmsWFE={[round(float(x), 3) for x in rms_w]} waves  "
            f"cov={[round(float(x), 3) for x in cov]}"
        )

    # 两者 delta 应一致（NA 不变 ⇒ Nyquist 相同）；微小差异是相似变换后
    # 浮点运算顺序不同导致的舍入噪声（~3e-8），容许到 1e-6
    d0, d1 = reps["original"].delta, reps[f"scaled x{SCALE}"].delta
    assert abs(d0 - d1) / d0 < 1e-6, f"delta mismatch: {d0} vs {d1}"
    extent = np.array([-1, 1, -1, 1]) * (SIZE * d0 / 2 * 1e3)  # µm，两图共用

    tags = list(reps)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2))
    for row, tag in enumerate(tags):
        rep = reps[tag]
        psf_np = _to_np(rep.psf[0, 0])
        wls = _to_np(rep.wavelengths)
        nas = _to_np(rep.na[0, 0])
        st = _to_np(strehl(rep)[0, 0])
        cov = _to_np(rep.coverage[0, 0])
        mid = len(wls) // 2

        axes[row, 0].imshow(rgb_composite(psf_np, wls), extent=extent, origin="lower")
        axes[row, 0].set_title(f"{tag} — RGB composite")
        axes[row, 0].set_ylabel("y (µm)")

        log_psf = np.log10(psf_np[mid] / psf_np[mid].max() + 1e-12)
        axes[row, 1].imshow(
            log_psf, extent=extent, origin="lower", vmin=LOG_VMIN, vmax=0, cmap="inferno"
        )
        airy_r_um = 0.61 * wls[mid] * 1e-3 / nas[mid]
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[row, 1].plot(
            airy_r_um * np.cos(theta),
            airy_r_um * np.sin(theta),
            "w--",
            lw=1.0,
            alpha=0.8,
            label="Airy 1st zero (invariant!)",
        )
        axes[row, 1].legend(loc="upper right", fontsize=8)
        axes[row, 1].set_title(f"{tag} — log10 PSF @ {wls[mid]:.0f} nm")

        # MTF @ 中波长：本行实线 + 衍射极限虚线
        m = mtf(rep)
        row_mtf = _to_np(m.mtf[0, 0, mid, SIZE // 2, :])
        freqs = _to_np(m.freqs)
        pos = freqs >= 0
        axes[row, 2].plot(freqs[pos], row_mtf[pos], lw=1.8, label=f"{tag} @ {wls[mid]:.0f} nm")
        axes[row, 2].plot(
            freqs[pos],
            mtf_diffraction(freqs[pos], nas[mid], wls[mid]),
            "k--",
            lw=1.0,
            label="diffraction limit",
        )
        axes[row, 2].set_xlim(0, freqs[pos].max())
        axes[row, 2].set_ylim(0, 1.02)
        axes[row, 2].set_xlabel("spatial frequency (cycles/mm)")
        axes[row, 2].grid(alpha=0.3)
        axes[row, 2].legend(fontsize=8)
        axes[row, 2].set_title(
            f"Strehl={st[mid]:.4f}   coverage(min)={cov.min():.3f}   NA={nas[mid]:.3f}"
        )
    for ax in axes[-1, :2]:
        ax.set_xlabel("x (µm)")
    fig.suptitle(
        f"cox1_02: original vs ×{SCALE} scaled — same physical window "
        f"(±{extent[1]:.0f} µm, delta={d0 * 1e3:.3f} µm/px both)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / "scale_compare_cox1_02.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[saved] {out}")


if __name__ == "__main__":
    sys.exit(main())
