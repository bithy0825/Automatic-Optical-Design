"""测量各损失对可训练参数的"单位损失作用力"（docs/loss_param_forces.md 的数据来源）。

定义
----
某损失列权为 1、其余为 0 时：
* 力(p)        = 逐元素 |∂L/∂p| 均值（按参数名词聚合全系统同名叶子）；
* 单位损失力   = 力(p) / L0，L0 为该损失的逐个体均值。
  梯度随权重线性缩放，故"该损失加权值为 1 时贡献给 p 的梯度"精确等于单位损失力。
* bounds 列在越界 1 个单位的克隆体上测量（铰链解析律：力 = 2/越界量）。

行归一化：每行（参数）除以该行各列单位损失力之和，单元格 = 比例 (原始单位损失力)。

测试系统：内置专用配置（非 demo）——3 非球面、强曲率分布、中间面出射到空气，
自然触发 aperture_clip / TIR；状态 C 叠加逐个体"死亡矩阵"，强制激活
sag_domain / solver_negative / solver_convergence，并打印各死因死亡光线数作证。

运行: python tests/measure_loss_forces.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core import Verdict, term
from optimization import build_sequential
from optimization.loss import LossWeights, total_loss
from optimization.utils import build_target

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# 专用测试配置：强曲率 + 玻璃→空气中间面，让各类死亡自然发生
# ─────────────────────────────────────────────────────────────────────────────

_LENS_BOUNDS = {
    "diameter": [10.0, 30.0],
    "curvature": [-0.2, 0.2],
    "kappa": [-5.0, 5.0],
    "alpha": [-1e-2, 1e-2],
}


def _lens(curvature: dict, material: dict, *, allow_negative: bool = False) -> dict:
    block = {
        "type": "refractor",
        "shape": "asphere",
        "diameter": {"method": "raw", "value": 20.0},
        "curvature": curvature,
        "kappa": {"method": "raw", "value": 0.0},
        "alpha4": {"method": "raw", "value": 1e-4},
        "alpha6": {"method": "raw", "value": 1e-6},
        "material": material,
        "train": {"diameter": True, "curvature": True, "kappa": True, "alpha": True},
        "mutate": {"curvature": 0.005, "kappa": 0.05, "alpha": 1e-5, "material": 0.1},
        "bounds": dict(_LENS_BOUNDS),
    }
    if allow_negative:
        block["solver"] = {"allow_negative": True}
    return block


def _gap(mean: float, std: float, hi: float) -> dict:
    return {
        "type": "gap",
        "thickness": {"method": "normal", "mean": mean, "std": std},
        "train": {"thickness": True},
        "mutate": {"thickness": 0.5},
        "bounds": {"thickness": [1.0, hi]},
    }


CONFIG = {
    "target": {"fov": 25, "F": 4.0, "effl": 50.0, "wavelength": [486, 589, 656]},
    "ga": {"population": 8, "topk": 4, "generation": 100},
    "component": [
        {
            "type": "source",
            "pupil": {"method": "uniform", "region": "disk", "count": [5, 8]},
            "field": {"method": "uniform", "region": "rect", "count": [3, 1]},
            "wavel": {"method": "uniform", "region": "line", "count": 3},
        },
        {"type": "gap", "thickness": {"method": "raw", "value": 0.0}},
        # 强正透镜（首面允许负根）：部分个体 c·r 越 sag 域
        _lens(
            {"method": "normal", "mean": 0.06, "std": 0.03},
            {"method": "random", "db": "sellmeier"},
            allow_negative=True,
        ),
        _gap(8.0, 2.0, 20.0),
        # 玻璃→空气出射面：边缘大入射角 → TIR
        _lens(
            {"method": "normal", "mean": -0.05, "std": 0.03},
            {"method": "raw", "value": "air", "db": "constant"},
        ),
        _gap(6.0, 2.0, 20.0),
        # 第三面：允许负根=False（缺省），负距离即 solver_negative
        _lens(
            {"method": "normal", "mean": 0.05, "std": 0.03},
            {"method": "random", "db": "sellmeier"},
        ),
        _gap(30.0, 5.0, 100.0),
        {"type": "sensor", "diameter": {"method": "raw", "value": 25.0}},
    ],
}

NOUNS = [term.DIAMETER, term.CURVATURE, term.KAPPA, term.ALPHA, term.THICKNESS]
CAUSES = [
    Verdict.Cause.SAG_DOMAIN,
    Verdict.Cause.SOLVER_NEGATIVE,
    Verdict.Cause.SOLVER_CONVERGENCE,
    Verdict.Cause.APERTURE_CLIP,
    Verdict.Cause.TIR,
]


def _weights(effl=0.0, spot=0.0, toll_cause=None, bounds_noun=None) -> LossWeights:
    toll = {c: 0.0 for c in CAUSES}
    if toll_cause is not None:
        toll[toll_cause] = 1.0
    bounds = {n: 0.0 for n in NOUNS}
    if bounds_noun is not None:
        bounds[bounds_noun] = 1.0
    return LossWeights(effl=effl, spot=spot, toll=toll, bounds=bounds)


COLUMNS = (
    [("effl", _weights(effl=1.0)), ("spot", _weights(spot=1.0))]
    + [(f"toll:{c.name.lower()}", _weights(toll_cause=c)) for c in CAUSES]
    + [(f"bounds:{n.canonical.lower()}", _weights(bounds_noun=n)) for n in NOUNS]
)


def _leaf_params(seq):
    out = {n.canonical: [] for n in NOUNS}
    for name, p in seq.named_parameters():
        if p.requires_grad:
            leaf = name.rsplit(".", 1)[-1]
            if leaf in out:
                out[leaf].append(p)
    return out


@torch.no_grad()
def _perturb_bounds(seq, blocks):
    """按名词把各元件参数推到自己 bounds 上界外 1 个单位的克隆体字典。"""
    out = {}
    for noun in NOUNS:
        key = noun.canonical
        trial = seq.clone()
        for name, t in trial.named_parameters():
            if name.rsplit(".", 1)[-1] != key:
                continue
            comp_idx = int(name.split(".")[1])
            bounds = term.BOUNDS.resolve(blocks[comp_idx], default={})
            hi = next((h for k, (_l, h) in bounds.items() if noun.match(k)), None)
            if hi is not None:
                t.fill_(hi + 1.0)
        out[key.lower()] = trial
    return out


@torch.no_grad()
def death_report(seq, label: str) -> None:
    """各死因的死亡光线数（全 0 的列即"该状态未触发"，非梯度通路缺失）。"""
    flow = seq()
    v = flow.verdict
    print(f"{label}: 存活 {int(v.hold.sum())}/{v.hold.numel()} 光线")
    for c in CAUSES:
        n = int((~v.hold & v.cause.eq(c)).sum())
        print(f"  {c.name.lower()}: {n}")


def measure(seq, target, blocks) -> dict:
    """{列名: {"L0": float, "force": {名词: 单位损失力}}}。"""
    perturbed = _perturb_bounds(seq, blocks)
    result = {}
    for col, w in COLUMNS:
        subject = (
            perturbed[col.split(":", 1)[1]] if col.startswith("bounds:") else seq
        )
        params = _leaf_params(subject)
        for p in subject.parameters():
            p.grad = None
        flow = subject()
        total, _ = total_loss(flow, subject, target, blocks, w)
        L0 = float(total.detach().mean())
        total.mean().backward()
        force = {}
        for noun in NOUNS:
            key = noun.canonical
            g = [p.grad.abs().mean() for p in params[key] if p.grad is not None]
            raw = float(torch.stack(g).mean()) if g else 0.0
            force[key] = raw / L0 if L0 > 1e-12 else 0.0
        result[col] = {"L0": L0, "force": force}
    return result


def print_tables(label: str, data: dict) -> None:
    cols = [c for c, _ in COLUMNS]
    nouns = [n.canonical for n in NOUNS]

    print(f"\n### {label} — 典型损失值 L0（权=1，逐个体均值）\n")
    print("| 损失 | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    print("| L0 | " + " | ".join(f"{data[c]['L0']:.3g}" for c in cols) + " |")

    print(f"\n### {label} — 行归一化（每行合计 = 1；单元格 = 比例 (原始单位损失力)）\n")
    print("| 参数 \\ 损失 | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for noun in nouns:
        vals = [data[c]["force"][noun] for c in cols]
        s = sum(vals)
        row = " | ".join(f"{v / s:.3g} ({v:.3g})" if s > 0 else "—" for v in vals)
        print(f"| {noun.lower()} | {row} |")


@torch.no_grad()
def apply_death_matrix(seq) -> None:
    """逐个体强制构造各死因（原地修改）：

    * 个体 0：r1 强曲率（rim 越 sag 域）+ r2 玻璃→空气大入射角（TIR）
    * 个体 1：首个训练间隔为负 → 下游负距离（solver_negative）
    * 个体 2：r3 巨型 alpha + 强曲率 → Newton 难收敛（solver_convergence）
    * 个体 3：r1 正 κ + 强曲率（sag_domain 的另一形态）
    """
    r1, r2, r3 = seq[2], seq[4], seq[6]
    r1.shape.Curvature[0] = 0.13   # |c·r| = 1.3 > 1 → rim 死 sag_domain
    r2.shape.Curvature[0] = 0.09   # rim 入射 ~64° > 临界角 → TIR（玻璃→空气）
    r3.shape.Curvature[0] = -0.12
    seq[3].Thickness[1] = -3.0     # 下一面在光线身后 → 负距离
    r3.shape.Alpha[2, 0] = 0.3     # 野性面形 → Newton 挣扎
    r3.shape.Curvature[2] = 0.10
    r1.shape.Kappa[3] = 2.0
    r1.shape.Curvature[3] = 0.14   # (1+κ)c²r² = 5.9 → sag_domain


def main() -> None:
    torch.manual_seed(0)
    target = build_target(CONFIG)
    seq = build_sequential(CONFIG, target).to(DEVICE)
    blocks = list(CONFIG["component"])

    print(f"device: {DEVICE}")
    death_report(seq, "状态 A 死亡分布")
    print_tables("状态 A：初始种群（自然死亡）", measure(seq, target, blocks))

    crisis = seq.clone()
    apply_death_matrix(crisis)
    death_report(crisis, "状态 C 死亡分布")
    print_tables("状态 C：死亡矩阵（全死因强制激活）", measure(crisis, target, blocks))


if __name__ == "__main__":
    main()
