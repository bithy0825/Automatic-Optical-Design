"""端到端 f32 vs f64 出货对比：逐样本比对 .pt 图像与 meta 字段。

用法::

    python scripts/exp_ab_dataset_compare.py /tmp/psf_ab_f32 /tmp/psf_ab_f64
"""

import json
import sys
from pathlib import Path

import torch


def main() -> None:
    dir32, dir64 = Path(sys.argv[1]), Path(sys.argv[2])
    m32 = {r["file"]: r for r in map(json.loads, (dir32 / "metadata.jsonl").read_text().splitlines())}
    m64 = {r["file"]: r for r in map(json.loads, (dir64 / "metadata.jsonl").read_text().splitlines())}
    keys32, keys64 = set(m32), set(m64)
    print(f"samples: f32={len(keys32)} f64={len(keys64)} 仅f32={len(keys32 - keys64)} 仅f64={len(keys64 - keys32)}")
    if keys32 - keys64:
        print("  仅f32:", sorted(keys32 - keys64)[:5])
    if keys64 - keys32:
        print("  仅f64:", sorted(keys64 - keys32)[:5])

    worst_l1, worst_mx, worst_cov, worst_scale = (0.0, ""), (0.0, ""), (0.0, ""), (0.0, "")
    n_perturb_diff = 0
    for k in sorted(keys32 & keys64):
        a = torch.load(dir32 / k, weights_only=True)["psf"].double()
        b = torch.load(dir64 / k, weights_only=True)["psf"].double()
        l1 = float((a - b).abs().sum() / b.abs().sum().clamp_min(1e-300))
        mx = float((a - b).abs().max() / b.max().clamp_min(1e-300))
        if l1 > worst_l1[0]:
            worst_l1 = (l1, k)
        if mx > worst_mx[0]:
            worst_mx = (mx, k)
        dcov = abs(m32[k]["coverage_raw"] - m64[k]["coverage_raw"])
        if dcov > worst_cov[0]:
            worst_cov = (dcov, k)
        ds = abs(m32[k]["scale"] - m64[k]["scale"])
        if ds > worst_scale[0]:
            worst_scale = (ds, k)
        pa, pb = m32[k]["perturb"], m64[k]["perturb"]
        if (pa == "fallback" or isinstance(pa, dict)) != (pb == "fallback" or isinstance(pb, dict)):
            n_perturb_diff += 1
            print(f"  [perturb-diverge] {k}: f32={pa} f64={pb}")
    print(f"出货图像最差: relL1={worst_l1[0]:.3e} @ {worst_l1[1]}")
    print(f"            relMax={worst_mx[0]:.3e} @ {worst_mx[1]}")
    print(f"meta 最差: |Δcoverage_raw|={worst_cov[0]:.3e} @ {worst_cov[1]}")
    print(f"           |Δscale|={worst_scale[0]:.3e} @ {worst_scale[1]}")
    print(f"扰动验收分流样本数: {n_perturb_diff}")


if __name__ == "__main__":
    main()
