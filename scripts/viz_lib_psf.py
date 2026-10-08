"""库模型 PSF 可视化：用新 analysis 接口渲染 Automatic-Optical-Design-Lib 在库模型。

只读使用 Lib（绝不写入）；输出保存到本仓库 outputs/lib_viz/。

交付：
* 每模型三联图：多色 RGB 合成 PSF（绝对网格零插值）| log PSF + Airy 环 | MTF vs 衍射极限
* 摄影镜头视场扫描：0% / 70.7% / 100% 视场的 RGB 合成与 log PSF
* 五模型 log PSF 画廊

运行：PYTHONPATH=. python scripts/viz_lib_psf.py
"""

import sys
import tomllib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import PsfSpec, mtf, psf_from_checkpoint, strehl, wavefront

LIB = Path(r"E:/workspace/Automatic-Optical-Design-Lib")
OUT = Path(__file__).resolve().parent.parent / "outputs" / "lib_viz"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("Zebase A_001", LIB / "Zebase" / "A_001" / "A_001_8232.pth"),
    ("Zebase A_010", LIB / "Zebase" / "A_010" / "A_010_17852.pth"),
    ("Zebase A_008", LIB / "Zebase" / "A_008" / "A_008_29994.pth"),
    (
        "Photo prime Cox1-06",
        LIB
        / "Navigation"
        / "Photographic lenses - prime"
        / "zmx"
        / "Cox1-06"
        / "Cox1-06_23345.pth",
    ),
    (
        "Microscope obj. (IO 11-40)",
        LIB
        / "Navigation"
        / "Microscope objectives"
        / "zmx"
        / "Imaging_Optics_Figure_11_40_optimised"
        / "Imaging_Optics_Figure_11_40_optimised_29377.pth",
    ),
    (
        "Eyepiece US01159233-2",
        LIB / "Navigation" / "Eyepieces" / "zmx" / "US01159233-2" / "US01159233-2_25339.pth",
    ),
    (
        "Telescope US08194318-1",
        LIB / "Navigation" / "Telescopes" / "zmx" / "US08194318-1" / "US08194318-1_8993.pth",
    ),
]

# 视场扫描用 A_001（fov 25°，像质好，离轴像差演化明显）
FIELD_MODEL = MODELS[0]

SIZE = 128
GAMMA = 0.4  # RGB 合成伽马（提亮环带同时保色）
LOG_VMIN = -4.0  # log10 显示下限（约等于 2e4 光线的噪声底）


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def rgb_composite(psf: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """逐波长 PSF → RGB 合成（共享绝对网格，逐点着色，零插值）。

    波长升序映射 B→G→R；每通道按各自峰值归一后做伽马编码。
    """
    W = psf.shape[0]
    order = np.argsort(wavelengths)
    picks = [order[0], order[W // 2], order[-1]] if W >= 3 else [order[0]] * 3
    chans = []
    for idx in picks:  # B, G, R 顺序收集
        ch = psf[idx].astype(np.float64)
        m = ch.max()
        chans.append(ch / m if m > 0 else ch)
    # picks 是 [短, 中, 长] → [B, G, R]；matplotlib 要 RGB
    b, g, r = chans
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0) ** GAMMA


def mtf_diffraction(freqs: np.ndarray, na: float, wl_nm: float) -> np.ndarray:
    """非相干衍射极限 MTF：fc = 2NA/λ。"""
    fc = 2.0 * na / (wl_nm * 1e-6)  # cycles/mm
    nu = np.clip(np.abs(freqs) / fc, 0.0, 1.0)
    return (2.0 / np.pi) * (np.arccos(nu) - nu * np.sqrt(1.0 - nu**2))


def model_panel(name: str, pth: Path) -> dict:
    """单模型三联图：RGB 合成 | log PSF + Airy 环 | MTF vs 衍射极限。"""
    rep = psf_from_checkpoint(pth, PsfSpec(size=SIZE))
    psf = _to_np(rep.psf[0, 0])  # (W,H,H)
    wls = _to_np(rep.wavelengths)
    nas = _to_np(rep.na[0, 0])
    cov = _to_np(rep.coverage[0, 0])
    st = _to_np(strehl(rep)[0, 0])
    wf = wavefront(rep)
    rms_waves = _to_np(wf.rms[0, 0]) / (wls * 1e-6)
    mid = len(wls) // 2
    delta_um = rep.delta * 1e3
    extent = np.array([-1, 1, -1, 1]) * (SIZE * rep.delta / 2 * 1e3)  # µm

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

    # (a) 多色 RGB 合成
    axes[0].imshow(rgb_composite(psf, wls), extent=extent, origin="lower")
    wl_txt = "/".join(f"{w:.0f}" for w in wls)
    axes[0].set_title(f"polychromatic RGB (B/G/R ← {wl_txt} nm)")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")

    # (b) log PSF + Airy 第一暗环
    log_psf = np.log10(psf[mid] / psf[mid].max() + 1e-12)
    axes[1].imshow(log_psf, extent=extent, origin="lower", vmin=LOG_VMIN, vmax=0, cmap="inferno")
    airy_r_um = 0.61 * wls[mid] * 1e-3 / nas[mid]  # 0.61·λ/NA，µm
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[1].plot(
        airy_r_um * np.cos(theta),
        airy_r_um * np.sin(theta),
        "w--",
        lw=1.0,
        alpha=0.8,
        label=f"Airy 1st zero ({wls[mid]:.0f} nm)",
    )
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title(f"log10 PSF @ {wls[mid]:.0f} nm")
    axes[1].set_xlabel("x (µm)")

    # (c) MTF vs 衍射极限
    m = mtf(rep)
    mtf_np = _to_np(m.mtf[0, 0])  # (W,H,H)
    freqs = _to_np(m.freqs)
    pos = freqs >= 0
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(wls)))
    for w, wl in enumerate(wls):
        row = mtf_np[w, SIZE // 2, :]
        axes[2].plot(freqs[pos], row[pos], color=colors[w], lw=1.6, label=f"{wl:.0f} nm")
        axes[2].plot(
            freqs[pos],
            mtf_diffraction(freqs[pos], nas[w], wl),
            color=colors[w],
            ls="--",
            lw=1.0,
            alpha=0.7,
        )
    axes[2].plot([], [], "k--", lw=1.0, label="diffraction limit")
    axes[2].set_xlim(0, freqs[pos].max())
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xlabel("spatial frequency (cycles/mm)")
    axes[2].set_ylabel("MTF")
    axes[2].set_title("MTF (tangential slice)")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fno = 1.0 / (2.0 * nas[mid])
    fig.suptitle(
        f"{name}   |   f/{fno:.2f}  NA={nas[mid]:.3f}  Δ={delta_um:.2f} µm/px   |   "
        f"Strehl={st[mid]:.3f}  RMS WFE={rms_waves[mid]:.3f} λ   |   "
        f"coverage(min)={cov.min():.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT / f"panel_{name.replace(' ', '_').replace('/', '-')}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[saved] {out.name}")
    return {"name": name, "log_psf": log_psf, "extent": extent, "strehl": st[mid]}


def field_sweep(name: str, pth: Path) -> None:
    """视场扫描：0% / 70.7% / 100% 最大视场的 RGB 合成与 log PSF。"""
    cfg = tomllib.loads(pth.with_name("config.toml").read_text(encoding="utf-8"))
    fov = cfg["target"]["fov"]
    if isinstance(fov, (int, float)):
        fx_max, fy_max = 0.0, float(fov)
    elif isinstance(fov[0], (int, float)):
        fx_max, fy_max = float(fov[0]), float(fov[1])
    else:
        (h0, h1), (v0, v1) = fov
        fx_max, fy_max = max(abs(h0), abs(h1)), max(abs(v0), abs(v1))
    fields = [(0.0, 0.0)]
    # 沿有视场范围的轴取 70.7% / 100%（0.707 对应"半视场面积"平均点）
    for frac in (0.707, 1.0):
        fields.append((fx_max * frac, fy_max * frac))

    fig, axes = plt.subplots(len(fields), 2, figsize=(10.5, 4.6 * len(fields)))
    for i, fld in enumerate(fields):
        rep = psf_from_checkpoint(pth, PsfSpec(size=SIZE, field=fld))
        psf = _to_np(rep.psf[0, 0])
        wls = _to_np(rep.wavelengths)
        nas = _to_np(rep.na[0, 0])
        st = _to_np(strehl(rep)[0, 0])
        mid = len(wls) // 2
        extent = np.array([-1, 1, -1, 1]) * (SIZE * rep.delta / 2 * 1e3)

        axes[i, 0].imshow(rgb_composite(psf, wls), extent=extent, origin="lower")
        axes[i, 0].set_ylabel("y (µm)")
        axes[i, 0].set_title(f"field ({fld[0]:.2f}°, {fld[1]:.2f}°) RGB   Strehl={st[mid]:.3f}")
        log_psf = np.log10(psf[mid] / psf[mid].max() + 1e-12)
        axes[i, 1].imshow(
            log_psf, extent=extent, origin="lower", vmin=LOG_VMIN, vmax=0, cmap="inferno"
        )
        airy_r_um = 0.61 * wls[mid] * 1e-3 / nas[mid]
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[i, 1].plot(airy_r_um * np.cos(theta), airy_r_um * np.sin(theta), "w--", lw=1.0)
        axes[i, 1].set_title(f"log10 PSF @ {wls[mid]:.0f} nm")
        print(
            f"  field {fld}: Strehl(mid)={st[mid]:.3f}, "
            f"coverage(min)={_to_np(rep.coverage[0, 0]).min():.3f}"
        )
    for ax in axes[-1]:
        ax.set_xlabel("x (µm)")
    fig.suptitle(f"{name} — field sweep (fov max ≈ ({fx_max:g}°, {fy_max:g}°))", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT / f"fields_{name.replace(' ', '_')}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[saved] {out.name}")


def gallery(results: list[dict]) -> None:
    """五模型 log PSF 画廊。"""
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.6))
    for ax, res in zip(axes, results, strict=True):
        ax.imshow(
            res["log_psf"],
            extent=res["extent"],
            origin="lower",
            vmin=LOG_VMIN,
            vmax=0,
            cmap="inferno",
        )
        ax.set_title(f"{res['name']}\nStrehl={res['strehl']:.3f}", fontsize=10)
        ax.set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    fig.suptitle("Library model PSF gallery (log10, mid wavelength)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "gallery.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"[saved] {out.name}")


def main() -> None:
    results = []
    for name, pth in MODELS:
        print(f"[run] {name}")
        results.append(model_panel(name, pth))
    print(f"[run] field sweep: {FIELD_MODEL[0]}")
    field_sweep(*FIELD_MODEL)
    gallery(results)
    print(f"done → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
