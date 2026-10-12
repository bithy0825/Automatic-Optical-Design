"""PSF 数据集产物 QA：完整性断言 + 统计表 + 可视化拼图。

用法::

    python scripts/check_psf_samples.py <输出目录> [--montage N]

检查项：
1. 张量形状 (64,64,3) float32、有限、非负；
2. 通道和 == 元数据 E_c（能量定标一致）；
3. 轴上样本 E_c == w_c（cos_n=1、tput_ratio=1 锚点语义）；
4. 干净样本 coverage ≥ τ_j；扰动样本 coverage ≥ 0.70（验收双条件）；
5. 轴上 PSF 质心在近中心（旋转对称系统轴上像无偏心）；
6. 能量随视场衰减单调性（径向档均值表，人工看）；
7. 拼图：log10 RGB 合成图（肉眼检查衍射环/彗差形态）。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    ap.add_argument("--montage", type=int, default=16)
    a = ap.parse_args()

    recs = []
    for line in (a.out / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"[warn] 跳过损坏行: {e}")
    kept = [r for r in recs if "file" in r]
    dropped = [r for r in recs if "dropped" in r]
    print(f"记录 {len(recs)}，入库 {len(kept)}，丢弃 {len(dropped)}")
    for r in dropped:
        print(f"  [dropped] s{r['structure']} {r['bins']}: {r['dropped']}")

    fails: list[str] = []
    for r in kept:
        d = torch.load(a.out / r["file"], weights_only=False, map_location="cpu")
        img, tag = d["psf"], r["file"].split("/")[-1]
        # 1
        if img.shape != (64, 64, 3) or img.dtype != torch.float32:
            fails.append(f"{tag}: 形状/dtype {img.shape} {img.dtype}")
        if not torch.isfinite(img).all():
            fails.append(f"{tag}: 含 NaN/Inf")
        if (img < 0).any():
            fails.append(f"{tag}: 含负值")
        # 2
        sums = img.sum(dim=(0, 1))
        for c in range(3):
            if abs(float(sums[c]) - r["energy"]["E_c"][c]) > 1e-4:
                fails.append(f"{tag}: 通道和 {float(sums[c]):.5f} != E_c {r['energy']['E_c'][c]}")
        # 3
        if r["bins"]["radial"] == -1 and r["perturb"] == "none":
            for c in range(3):
                if abs(r["energy"]["E_c"][c] - r["energy"]["w_c"][c]) > 1e-5:
                    fails.append(f"{tag}: 轴上 E_c != w_c")
        # 4
        if r["perturb"] == "none":
            if r["coverage"] < r["tau_j"] - 5e-3:
                fails.append(
                    f"{tag}: 干净样本 coverage {r['coverage']:.4f} < tau_j {r['tau_j']:.4f}"
                )
        elif isinstance(r["perturb"], dict) and r["coverage"] < 0.70 - 1e-6:
            fails.append(f"{tag}: 扰动样本 coverage {r['coverage']:.4f} < 0.70")
        # 5（仅轴上干净：质心应在半像素内）
        if r["bins"]["radial"] == -1 and r["perturb"] == "none":
            g = img[:, :, 1].double()
            yy, xx = torch.meshgrid(
                torch.arange(64, dtype=torch.float64),
                torch.arange(64, dtype=torch.float64),
                indexing="ij",
            )
            cy = float((g * yy).sum() / g.sum())
            cx = float((g * xx).sum() / g.sum())
            if math.hypot(cx - 31.5, cy - 31.5) > 0.5:
                fails.append(f"{tag}: 轴上质心偏移 ({cx:.2f},{cy:.2f})")

    # 6 能量衰减表
    print("\nE_c(green) 按径向档：")
    for j in [-1, 0, 1, 2, 3]:
        es = [
            r["energy"]["E_c"][1]
            for r in kept
            if r["bins"]["radial"] == j and r["perturb"] == "none"
        ]
        if es:
            print(
                f"  r{j}: n={len(es):2d}  mean={sum(es) / len(es):.4f}  range=({min(es):.4f},{max(es):.4f})"
            )

    # 扰动/相机统计
    from collections import Counter

    print(
        "扰动:",
        dict(
            Counter(
                r["perturb"] if isinstance(r["perturb"], str) else r["perturb"]["kind"]
                for r in kept
            )
        ),
    )
    print("相机种数:", len({r["camera"]["name"] for r in kept}), "/", len(kept))

    # 7 拼图（log10，每样本 RGB 合成）
    sel = kept[: a.montage]
    ncols = 4
    nrows = (len(sel) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for ax, r in zip(axes.flat, sel, strict=False):
        img = torch.load(a.out / r["file"], weights_only=False, map_location="cpu")["psf"].double()
        img = img / img.max()
        rgb = torch.log10(img.clamp_min(1e-6)) / 6 + 1  # log10 映射 6 个数量级
        ax.imshow(rgb.numpy())
        th = math.hypot(*r["field_deg"])
        ax.set_title(
            f"s{r['structure']} θ={th:.1f}°\ncov={r['coverage']:.3f} E={r['energy']['E_c'][1]:.3f}",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes.flat[len(sel) :]:
        ax.axis("off")
    fig.tight_layout()
    png = a.out / "montage.png"
    fig.savefig(png, dpi=110)
    print(f"\n拼图 → {png}")

    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} 个问题")
    for f_ in fails:
        print(" ", f_)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
