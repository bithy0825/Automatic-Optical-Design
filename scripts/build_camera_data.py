"""相机光谱响应原始数据 → ``public/data/cameras/*.pt``（torch 直读）。

用法（仓库任意目录）：

    python public/scripts/build_camera_data.py [原始数据目录]

每台相机一个 ``<相机名>.pt``，内容为::

    {
        "wavelengths": (W,) float64,   # nm，严格升序
        "rgb":         (W, 3) float64, # 相对光谱灵敏度
        "camera":      str,            # 元数据（来源 JSON 原样附带）
        "manufacturer": str,
        "source":      str,
        "license":     str,
    }

归一化约定（写死，勿逐源跟随）：**除以全曲线全局最大值**——峰值 = 1.0，
R/G/B 通道比例完整保持（白平衡所需）；各来源原始量纲（约 1 / ~1.9–3 等）
的差异由此抹平。原始峰值记录在生成的 README.md 中备查。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DEFAULT = Path(r"E:\Users\资源\相机光谱响应曲线")
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cameras"


def convert_one(folder: Path) -> dict[str, object]:
    """读一台相机的 CSV + JSON，校验并归一化，返回待保存的字典。"""
    name = folder.name
    data = np.loadtxt(folder / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    meta = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))

    wavelengths = torch.from_numpy(data[:, 0]).double()
    rgb = torch.from_numpy(data[:, 1:4]).double()

    if wavelengths.shape[0] != meta["num_bands"]:
        raise ValueError(f"{name}: bands mismatch {wavelengths.shape[0]} != {meta['num_bands']}")
    if not torch.all(wavelengths[1:] > wavelengths[:-1]):
        raise ValueError(f"{name}: wavelengths not strictly increasing")
    if not torch.isfinite(rgb).all() or (rgb < 0).any():
        raise ValueError(f"{name}: non-finite or negative response values")
    peak = rgb.max()
    if peak <= 0:
        raise ValueError(f"{name}: degenerate response (peak <= 0)")

    return {
        "wavelengths": wavelengths,
        "rgb": rgb / peak,
        "camera": meta["camera"],
        "manufacturer": meta["manufacturer"],
        "source": meta["source"],
        "license": meta["license"],
        # 以下两个键只进 README，不入 .pt（usage 不需要）
        "_peak": float(peak),
        "_convention": meta["value_convention"],
    }


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_DEFAULT
    folders = sorted(p for p in src.iterdir() if p.is_dir())
    if not folders:
        print(f"no camera folders under {src}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in folders:
        rec = convert_one(folder)
        torch.save(
            {k: v for k, v in rec.items() if not k.startswith("_")},
            OUT_DIR / f"{folder.name}.pt",
        )
        wl = rec["wavelengths"]
        rows.append(
            (
                folder.name,
                len(wl),
                float(wl[0]),
                float(wl[-1]),
                float(rec["_peak"]),
                str(rec["_convention"]),
            )
        )
        print(
            f"[{len(rows):>2}/{len(folders)}] {folder.name}: {len(wl)} bands, "
            f"raw peak {rec['_peak']:.4g}"
        )

    readme = [
        "# 相机光谱响应（预处理，torch 直读）",
        "",
        f"共 {len(rows)} 台相机，每台一个 `<相机名>.pt`，`torch.load(path, weights_only=True)` 直读：",
        "",
        "- `wavelengths`: (W,) float64，nm，严格升序",
        "- `rgb`: (W, 3) float64，相对光谱灵敏度（红/绿/蓝列）",
        "- `camera` / `manufacturer` / `source` / `license`: 元数据字符串",
        "",
        "**归一化约定**：除以全曲线全局最大值（峰值 = 1.0，R/G/B 通道比例保持，",
        "各来源量纲差异由此抹平）。绝对尺度本身任意——使用时白平衡见",
        "`analysis.camera.camera_psf`（先吞吐加权、后逐通道归一）。",
        "",
        "随机选一相机：`random.choice(list(THIS_DIR.glob('*.pt')))`。",
        "",
        "重新生成：`python public/scripts/build_camera_data.py <原始数据目录>`。",
        "许可：各曲线许可证见各自 .pt 内的 `license` 字段及原始出处。",
        "",
        "## 原始峰值与来源约定备查",
        "",
        "| 相机 | 波段数 | 范围 (nm) | 原始峰值 | 原始量纲约定 |",
        "|---|---|---|---|---|",
        *(f"| {n} | {b} | {lo:.0f}–{hi:.0f} | {pk:.4g} | {cv} |" for n, b, lo, hi, pk, cv in rows),
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"done: {len(rows)} cameras -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
