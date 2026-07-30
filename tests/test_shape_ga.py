"""四种面形的 from_options 分派、GA 三方法与 __getitem__ 导出测试。

覆盖 Sphere / Conic / Asphere / Disk：
* from_options —— 双态分派、种群大小、参数值、trainable opt-in、错误路径
* sort / breed / mutate —— 决定论精确验证（含 2D Alpha 与 Mask）
* __getitem__ —— 物理边界越界损失的数据出口；Asphere 的 ALPHA 导出
  mask 屏蔽后的有效系数（保留梯度链，无效尾部不参与惩罚）

用法: python tests/test_shape_ga.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from core import term
from shape import Asphere, Conic, Disk, Shape, Sphere

_FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    flag = "ok  " if cond else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _FAILED.append(name)


POP = 6
PERM = torch.tensor([2, 4, 0, 5, 1, 3])
IDLE = torch.tensor([0, 2, 5])  # 不参与 sort 验证以外的索引
IDX = torch.tensor([1, 3])

SPECS = {
    "sphere": {
        "shape": "sphere",
        "diameter": {"method": "raw", "value": 12.0},
        "curvature": {"method": "raw", "value": 0.05},
        "train": {"curvature": True},
    },
    "conic": {
        "shape": "conic",
        "diameter": {"method": "raw", "value": 12.0},
        "curvature": {"method": "raw", "value": 0.05},
        "kappa": {"method": "raw", "value": -0.5},
        "train": {"curvature": True, "kappa": True},
    },
    "asphere": {
        "shape": "asphere",
        "diameter": {"method": "raw", "value": 12.0},
        "curvature": {"method": "raw", "value": 0.05},
        "kappa": {"method": "raw", "value": -0.5},
        "alpha4": {"method": "raw", "value": 1e-4},
        "alpha6": {"method": "raw", "value": 1e-6},
        "train": {"alpha": True},
    },
    "disk": {
        "shape": "disk",
        "diameter": {"method": "raw", "value": 12.0},
    },
}

PARAMS = {  # 每种面形的主要参数名（用于 sort/breed 全量比对）
    "sphere": ("Diameter", "Curvature"),
    "conic": ("Diameter", "Curvature", "Kappa"),
    "asphere": ("Diameter", "Curvature", "Kappa", "Alpha", "Mask"),
    "disk": ("Diameter",),
}


def build(kind: str) -> Shape:
    return Shape.from_options(POP, SPECS[kind])


# ═══════════════════════════════════════════════════════════════════════════
# 1. from_options
# ═══════════════════════════════════════════════════════════════════════════

def test_from_options() -> None:
    print("from_options: 分派 / 种群 / 参数值 / trainable")

    s, c, a, d = build("sphere"), build("conic"), build("asphere"), build("disk")
    check("dispatch sphere", type(s) is Sphere)
    check("dispatch conic", type(c) is Conic)
    check("dispatch asphere", type(a) is Asphere)
    check("dispatch disk", type(d) is Disk)
    check("population", (s.population, c.population, a.population, d.population)
          == (POP, POP, POP, POP))

    check("sphere values", s.Diameter.eq(12.0).all() and s.Curvature.eq(0.05).all())
    check("conic kappa", c.Kappa.eq(-0.5).all())
    check("asphere alpha", a.Alpha.shape == (POP, 2)
          and torch.allclose(a.Alpha[0], torch.tensor([1e-4, 1e-6])))
    check("asphere mask default = n_coeffs", a.Mask.eq(2.0).all())

    check("diameter always trainable", s.Diameter.requires_grad
          and c.Diameter.requires_grad and a.Diameter.requires_grad
          and d.Diameter.requires_grad)
    check("sphere curvature opt-in", s.Curvature.requires_grad)
    check("conic kappa opt-in", c.Kappa.requires_grad)
    check("asphere alpha opt-in", a.Alpha.requires_grad)
    check("asphere curvature NOT trained (not listed)", not a.Curvature.requires_grad)
    check("asphere kappa NOT trained", not a.Kappa.requires_grad)
    check("mask never trainable", not a.Mask.requires_grad)

    # normal / uniform 规格
    torch.manual_seed(0)
    n1 = Shape.from_options(4, {"shape": "sphere",
                                "diameter": {"method": "normal", "mean": 10.0, "std": 0.1},
                                "curvature": {"method": "uniform", "low": 0.0, "high": 0.1}})
    torch.manual_seed(0)
    n2 = Shape.from_options(4, {"shape": "sphere",
                                "diameter": {"method": "normal", "mean": 10.0, "std": 0.1},
                                "curvature": {"method": "uniform", "low": 0.0, "high": 0.1}})
    check("normal/uniform deterministic per seed",
          torch.equal(n1.Diameter, n2.Diameter) and torch.equal(n1.Curvature, n2.Curvature))
    check("uniform range", bool(((n1.Curvature >= 0.0) & (n1.Curvature < 0.1)).all()))

    # 错误路径
    try:
        Shape.from_options(2, {"shape": "prism", "diameter": {"method": "raw", "value": 1.0}})
        check("unknown shape raises", False)
    except ValueError:
        check("unknown shape raises", True)
    try:
        Shape.from_options(2, {"diameter": {"method": "raw", "value": 1.0}})
        check("missing shape key raises", False)
    except KeyError:
        check("missing shape key raises", True)

    # alpha 阶次必须连续：仅 alpha4 合法（1 项），alpha4+alpha8 跳阶报错
    short = Shape.from_options(2, {"shape": "asphere",
                                   "diameter": {"method": "raw", "value": 1.0},
                                   "curvature": {"method": "raw", "value": 0.01},
                                   "kappa": {"method": "raw", "value": 0.0},
                                   "alpha4": {"method": "raw", "value": 1e-4}})
    check("asphere alpha4-only valid (1 coeff)", short.Alpha.shape == (2, 1))
    try:
        Shape.from_options(2, {"shape": "asphere",
                               "diameter": {"method": "raw", "value": 1.0},
                               "curvature": {"method": "raw", "value": 0.01},
                               "kappa": {"method": "raw", "value": 0.0},
                               "alpha4": {"method": "raw", "value": 1e-4},
                               "alpha8": {"method": "raw", "value": 1e-8}})
        check("asphere alpha order gap raises", False)
    except ValueError:
        check("asphere alpha order gap raises", True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. sort
# ═══════════════════════════════════════════════════════════════════════════

def test_sort() -> None:
    print("sort: 精确置换（全参数）")

    for kind, params in PARAMS.items():
        s = build(kind)
        snapshots = {p: getattr(s, p).detach().clone() for p in params}
        s.sort_(PERM)
        ok = all(
            torch.equal(getattr(s, p), snapshots[p][PERM]) for p in params
        )
        check(f"sort {kind} all params permuted", ok)


# ═══════════════════════════════════════════════════════════════════════════
# 3. breed
# ═══════════════════════════════════════════════════════════════════════════

def test_breed() -> None:
    print("breed: 精确滚动填充（全参数，含 2D Alpha）")

    for kind, params in PARAMS.items():
        s = build(kind)
        topk = 2
        snapshots = {p: getattr(s, p).detach().clone() for p in params}
        idx = torch.arange(POP - topk).remainder(topk)
        s.breed_(topk)
        ok = True
        for p in params:
            want = snapshots[p].clone()
            want[topk:] = snapshots[p][:topk][idx]
            ok &= torch.equal(getattr(s, p), want)
        check(f"breed {kind} all params rolling-filled", ok)

    s = build("sphere")
    before = s.Diameter.clone()
    s.breed_(POP)
    check("breed topk==P is no-op", torch.equal(s.Diameter, before))
    try:
        s.breed_(POP + 1)
        check("breed out-of-range raises", False)
    except AssertionError:
        check("breed out-of-range raises", True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. mutate
# ═══════════════════════════════════════════════════════════════════════════

def test_mutate() -> None:
    print("mutate: 选择性 / std=0 跳过 / 逐词表控制")

    torch.manual_seed(42)
    for kind, params in PARAMS.items():
        s = build(kind)
        snapshots = {p: getattr(s, p).detach().clone() for p in params}
        options = {"curvature": 0.01, "kappa": 1e-3, "alpha": 1e-5, "diameter": 0.0}
        s.mutate_(IDX, options)

        # diameter std=0 → 不变（Asphere 例外：它的 diameter 段只重归一化 α，自身也不应变）
        check(f"mutate {kind} diameter untouched (std=0)",
              torch.equal(s.Diameter, snapshots["Diameter"]))

        rest = torch.ones(POP, dtype=torch.bool)
        rest[IDX] = False
        if "Curvature" in params:
            check(f"mutate {kind} curvature selective",
                  torch.equal(s.Curvature[rest], snapshots["Curvature"][rest])
                  and not torch.equal(s.Curvature[IDX], snapshots["Curvature"][IDX]))
        if "Kappa" in params:
            check(f"mutate {kind} kappa selective",
                  torch.equal(s.Kappa[rest], snapshots["Kappa"][rest])
                  and not torch.equal(s.Kappa[IDX], snapshots["Kappa"][IDX]))
        if "Alpha" in params:
            check(f"mutate {kind} alpha selective",
                  torch.equal(s.Alpha[rest], snapshots["Alpha"][rest])
                  and not torch.equal(s.Alpha[IDX], snapshots["Alpha"][IDX]))
        if "Mask" in params:
            check(f"mutate {kind} mask untouched (not in options)",
                  torch.equal(s.Mask, snapshots["Mask"]))

    # Asphere 专项：直径变异保持物理系数 α/ρ^p
    torch.manual_seed(3)
    a = build("asphere")
    rho_old = a.Diameter.mul(0.5).unsqueeze(-1)
    phys_old = a.Alpha.detach().clone().div(rho_old.pow(torch.tensor([4.0, 6.0])))
    a.mutate_(torch.arange(POP), {"diameter": 1.0})
    rho_new = a.Diameter.mul(0.5).unsqueeze(-1)
    phys_new = a.Alpha.detach().clone().div(rho_new.pow(torch.tensor([4.0, 6.0])))
    check("asphere diameter-mutation keeps physical alpha",
          torch.allclose(phys_old, phys_new, rtol=1e-4))

    # Asphere 专项：mask 随机游走有界
    torch.manual_seed(5)
    a.mutate_(torch.arange(POP), {"mask": 1.0})
    check("asphere mask walk bounded [0, n_coeffs]",
          bool(((a.Mask >= 0.0) & (a.Mask <= 2.0)).all()))


# ═══════════════════════════════════════════════════════════════════════════
# 5. __getitem__
# ═══════════════════════════════════════════════════════════════════════════

def test_getitem() -> None:
    print("__getitem__: 词表导出（边界损失数据口）")

    s = build("sphere")
    check("sphere[Diameter] identity", s[term.DIAMETER] is s.Diameter)
    check("sphere[Curvature] identity", s[term.CURVATURE] is s.Curvature)

    c = build("conic")
    check("conic[Kappa] identity", c[term.KAPPA] is c.Kappa)

    d = build("disk")
    check("disk[Diameter] identity", d[term.DIAMETER] is d.Diameter)

    a = build("asphere")
    # ALPHA 导出 = mask 屏蔽视图，不是原始张量
    a.Mask.copy_(torch.tensor([1.0, 0.0, 2.0, 1.0, 0.0, 2.0]))
    exported = a[term.ALPHA]
    check("asphere[Alpha] is masked view, not raw buffer",
          exported.data_ptr() != a.Alpha.data_ptr())
    expect = a.Alpha.mul(
        torch.arange(2).lt(a.Mask.unsqueeze(-1))
    )
    check("asphere[Alpha] equals mask-applied values",
          torch.equal(exported, expect))
    check("asphere[Alpha] mask=0 行全零",
          bool((exported[1] == 0.0).all() and (exported[4] == 0.0).all()))

    # 可训练 alpha：导出保留梯度链（边界损失可反传）
    a.Alpha.grad = None
    exported.square().sum().backward()
    check("asphere[Alpha] gradient flows to raw Alpha",
          a.Alpha.grad is not None
          and bool((a.Alpha.grad[1] == 0.0).all())  # mask=0 行无梯度
          and bool((a.Alpha.grad[0, 0] != 0.0).any()))

    # 其余词表照常导出
    check("asphere[Mask] identity", a[term.MASK] is a.Mask)
    check("asphere[Kappa] identity", a[term.KAPPA] is a.Kappa)

    # 未持有的名词：快速失败
    try:
        s[term.THICKNESS]
        check("unknown noun raises", False)
    except AttributeError:
        check("unknown noun raises", True)


def main() -> int:
    torch.set_default_dtype(torch.float32)
    test_from_options()
    test_sort()
    test_breed()
    test_mutate()
    test_getitem()

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} assertions: {_FAILED}")
        return 1
    print("ALL PASS (4 shapes × from_options / sort / breed / mutate / __getitem__)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
