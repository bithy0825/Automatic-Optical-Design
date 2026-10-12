"""相机光谱曲线统一重采样到公共波长网格（原地改写 data/cameras/*.pt）。

方法：PCHIP（保形分段三次 Hermite 插值，Fritsch–Carlson）。相比线性插值
保持曲线光滑性；相比普通三次样条**不产生过冲/振铃**——响应曲线单调段
保持单调、值域不越界（无非物理负值）。原生测量范围外的网格点补 0
（未测即无响应）。重采样后按全局峰值再归一（保持库内不变量 max=1）。

原始文件备份到 ``data/cameras_backup_native/``（仅在首次运行时创建）。
已在目标网格上的文件自动跳过（幂等，可重复运行）。

用法::

    python scripts/unify_camera_wavelengths.py [--wl-min 400] [--wl-max 720] [--wl-step 10]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def unify(directory: Path, wl_min: float, wl_max: float, wl_step: float) -> None:
    grid = np.arange(wl_min, wl_max + wl_step * 0.5, wl_step, dtype=np.float64)
    backup = directory.parent / "cameras_backup_native"
    files = sorted(directory.glob("*.pt"))
    n_done = n_skip = 0
    for f in files:
        d = torch.load(f, weights_only=True, map_location="cpu")
        wl = d["wavelengths"].double().numpy()
        rgb = d["rgb"].double().numpy()
        if len(wl) == len(grid) and np.allclose(wl, grid):
            n_skip += 1
            continue
        # 首次运行才备份（幂等）
        if not backup.exists():
            backup.mkdir(parents=True)
            for g in files:
                shutil.copy2(g, backup / g.name)
            print(f"[backup] {len(files)} 个原始文件 → {backup}")
        order = np.argsort(wl)
        wl, rgb = wl[order], rgb[order]
        # 去重防御（PCHIP 要求严格递增）
        keep = np.concatenate(([True], np.diff(wl) > 0))
        wl, rgb = wl[keep], rgb[keep]
        out = np.zeros((len(grid), 3))
        in_range = (grid >= wl[0]) & (grid <= wl[-1])
        for c in range(3):
            p = PchipInterpolator(wl, rgb[:, c])
            out[in_range, c] = p(grid[in_range])
        peak = out.max()
        if peak <= 0:
            print(f"[warn] {f.stem}: 与目标网格无重叠，跳过改写")
            continue
        out /= peak  # 再归一，保持 max=1 不变量
        # 保真度：原生网格点上 PCHIP 恒过原数据点；报告通道积分相对变化
        integ_old = np.trapezoid(rgb, wl, axis=0)
        integ_new = np.trapezoid(out, grid, axis=0)
        rel = np.abs(integ_new - integ_old) / np.maximum(integ_old, 1e-300)
        d["wavelengths"] = torch.tensor(grid, dtype=d["wavelengths"].dtype)
        d["rgb"] = torch.tensor(out, dtype=d["rgb"].dtype)
        torch.save(d, f)
        n_done += 1
        print(
            f"[ok] {f.stem:32s} {len(wl):3d}({wl[0]:.0f}-{wl[-1]:.0f}nm) → {len(grid)} "
            f"∫S 相对变化 R/G/B = {rel[0]:.2%}/{rel[1]:.2%}/{rel[2]:.2%}"
        )
    print(
        f"[done] 改写 {n_done}，已在网格上跳过 {n_skip}，目标网格 {len(grid)} 点 "
        f"[{wl_min:.0f}, {wl_max:.0f}]nm @{wl_step:.0f}nm"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wl-min", type=float, default=400.0)
    ap.add_argument("--wl-max", type=float, default=720.0)
    ap.add_argument("--wl-step", type=float, default=10.0)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cameras",
    )
    a = ap.parse_args()
    unify(a.dir, a.wl_min, a.wl_max, a.wl_step)


if __name__ == "__main__":
    main()
