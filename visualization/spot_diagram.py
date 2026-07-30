"""点列图：标准 F×W 网格，每格一个视场 + 一个波长。"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from core import RayBundle, Verdict


def _wl_color(wavelengths: np.ndarray) -> np.ndarray:
    """``(N,)`` nm → ``(N, 3)`` sRGB，连续可见光谱近似。

    380–780 nm，两端向黑衰减以模拟人眼边缘敏感度下降。
    """
    wl = wavelengths.clip(380.0, 780.0)
    b = np.where(wl < 490, 1.0, np.where(wl < 510, (510 - wl) / 20, 0.0))
    g = np.where(
        wl < 440, 0.0,
        np.where(wl < 490, (wl - 440) / 50,
        np.where(wl < 580, 1.0,
        np.where(wl < 645, (645 - wl) / 65, 0.0))),
    )
    r = np.where(
        wl < 440, (440 - wl) / 60,
        np.where(wl < 510, 0.0,
        np.where(wl < 580, (wl - 510) / 70, 1.0)),
    )
    # 可见光谱两端向黑衰减
    fade = np.where(wl < 420, 0.3 + 0.7 * (wl - 380) / 40,
                    np.where(wl > 700, 0.3 + 0.7 * (780 - wl) / 80, 1.0))
    return np.stack([r * fade, g * fade, b * fade], axis=-1)


def spot_diagram(
    rays: RayBundle,
    *,
    verdict: Verdict | None = None,
    pop_index: int = 0,
    title: str | None = None,
    save: str | None = None,
    dpi: int = 150,
) -> Figure:
    """单个种群成员的点列图。

    Parameters
    ----------
    rays : RayBundle
        追迹后的光线，形状 ``(P, F, W, N, 3)``。
    verdict : Verdict, optional
        追迹裁决。提供时死光线以灰色低透明度叠画，中心与视野按存活光线计。
    pop_index : int
        种群成员索引。
    title : str | None
        总标题。
    save : str | None
        输出路径（``.png`` / ``.pdf`` / ``.svg``）。
    dpi : int
        输出分辨率。

    Returns
    -------
    plt.Figure
    """
    pts = rays.points[pop_index].detach().cpu().numpy()       # (F, W, N, 3)
    wls_nm = rays.wavelength[pop_index, 0, :, 0].detach().cpu().numpy()  # (W,)
    hold = (
        verdict.hold[pop_index].detach().cpu().numpy() if verdict is not None else None
    )  # (F, W, N) bool
    F, W = pts.shape[0], pts.shape[1]
    colors = _wl_color(wls_nm)  # (W, 3)

    fig, axes = plt.subplots(
        F, W,
        figsize=(2.5 * W, 2.5 * F),
        squeeze=False,
    )

    for fi in range(F):
        for wi in range(W):
            ax = axes[fi, wi]
            xy = pts[fi, wi, :, :2]  # (N, 2)
            if hold is not None:
                alive = hold[fi, wi]
                xy_alive, xy_dead = xy[alive], xy[~alive]
            else:
                xy_alive, xy_dead = xy, None
            ref = xy_alive if len(xy_alive) else xy  # 全死时退化为全体
            cx, cy = float(ref[:, 0].mean()), float(ref[:, 1].mean())
            ax.scatter(
                xy_alive[:, 0], xy_alive[:, 1], s=1.0, alpha=0.6,
                color=colors[wi], edgecolors="none", linewidths=0,
            )
            if xy_dead is not None and len(xy_dead):
                ax.scatter(
                    xy_dead[:, 0], xy_dead[:, 1], s=1.0, alpha=0.15,
                    color="gray", edgecolors="none", linewidths=0,
                )
            ax.set_aspect("equal")
            r = max(float(np.linalg.norm(ref - (cx, cy), axis=1).max()) * 1.2, 1.0)
            ax.set_xlim(cx - r, cx + r)
            ax.set_ylim(cy - r, cy + r)
            ax.tick_params(labelsize=6)
            ax.axhline(cy, color="gray", lw=0.3, alpha=0.5)
            ax.axvline(cx, color="gray", lw=0.3, alpha=0.5)

            if fi == 0:
                ax.set_title(f"λ = {wls_nm[wi]:.0f} nm", fontsize=8, pad=2)
            if wi == 0:
                ax.set_ylabel(f"F{fi}", fontsize=8, rotation=0, labelpad=12, va="center")

    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.93)

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
    return fig
