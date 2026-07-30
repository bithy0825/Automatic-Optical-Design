"""where_ 原地择优的正确性验证：

1. Indices（材料编号 buffer）确实被 _batched_tensors 捕获；
2. 逐行选择语义：mask=True 行取 new，False 行保持 old；
3. incident MaterialRef 链身份不变（无需 rebind）；
4. 模块对象身份不变（梯度优化器绑定不失效）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from component import Sequential
from component.refractor import Refractor
from optimization import build_sequential
from optimization.utils import build_target, load_config

cfg = load_config(str(Path(__file__).resolve().parents[1] / "demo" / "config.toml"))
seq = build_sequential(cfg, build_target(cfg))
blocks = list(cfg["component"])
P = seq.population
dev = seq.device

# ── 1. Indices 被 _batched_tensors 捕获 ──
names = [n for n, _ in seq._batched_tensors()]
indices_names = [n for n in names if n.endswith("transmitted.Indices")]
print(f"batched tensors: {len(names)}, 其中 Indices: {len(indices_names)}")
for n in indices_names:
    print(f"  captured: {n}")
assert len(indices_names) == 5, "source + 4 refractors 的 Indices 应全部被捕获"

# ── 2-4. 制造分歧 trial，执行 where_ ──
trial = seq.clone()
trial.mutate_(torch.arange(P, device=dev), blocks)

mask = torch.tensor([True, False, True, True, False, False, True, False], device=dev)

before_seq = {n: t.clone() for n, t in seq._batched_tensors()}
before_trial = {n: t.clone() for n, t in trial._batched_tensors()}
params_before = [p for p in seq.parameters()]
materials_before = [seq[i].transmitted for i in (0, 2, 4, 6, 8)]
incidents_before = [seq[i].incident for i in (2, 4, 6, 8)]

seq.where_(mask, trial)

# 2. 逐行语义
for n, t in seq._batched_tensors():
    m = mask.view(P, *([1] * (t.ndim - 1)))
    expect = torch.where(m, before_trial[n], before_seq[n])
    assert torch.equal(t, expect), f"where_ 行选择错误: {n}"
print("逐行选择: OK（含 Indices 材料编号）")

# 3. incident 链身份与指向
for i, up in ((2, 0), (4, 2), (6, 4), (8, 6)):
    refr = seq[i]
    assert isinstance(refr, Refractor)
    assert refr.incident is incidents_before[(i // 2) - 1], f"seq[{i}].incident 对象被更换"
    assert refr.incident._material is seq[up].transmitted, f"seq[{i}].incident 指向错误"
print("incident MaterialRef 链: 对象身份与指向均不变")

# 4. 参数与材料对象身份
assert all(p is q for p, q in zip(params_before, seq.parameters())), "Parameter 对象被更换"
assert all(m is seq[i].transmitted for m, i in zip(materials_before, (0, 2, 4, 6, 8)))
print("Parameter / Material 对象身份: 不变")

print("\nALL PASS")
