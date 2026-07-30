"""端到端优化验证：缩小规模的 demo 配置实跑 + 正确性断言。

检查项：
1. 优化全程无异常，损失逐代下降；
2. SA / Adam 两阶段在每一代都活着（步间 loss 变化）；
3. 精英直通：sort 后 breed_+mutate_ 不改变 top-k 行；
4. 优化器参数绑定全程有效（where_ 原地语义）。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from optimization import (
    AdamOptions,
    GAOptions,
    GeneticAlgorithm,
    GradientOptimizer,
    LossHistory,
    SAOptions,
    SimulatedAnnealing,
    build_sequential,
)
from optimization.utils import build_target, load_config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

cfg = load_config(str(Path(__file__).resolve().parents[1] / "demo" / "config.toml"))
target = build_target(cfg)
seq = build_sequential(cfg, target).to(DEVICE)
blocks = list(cfg["component"])

torch.manual_seed(0)
params_before = list(seq.parameters())

sa = SimulatedAnnealing(SAOptions(step=50, T0=10.0, T1=0.01))
adam = GradientOptimizer(AdamOptions(step=50, scheduler="cosine", grad_norm=10.0, lr={}))
hist = LossHistory()
ga = GeneticAlgorithm(
    GAOptions(population=8, generation=10, topk=4), stages=[sa, adam]
)

t0 = time.perf_counter()
ga.run(seq, target, blocks, callbacks=[hist])
dt = time.perf_counter() - t0
print(f"elapsed: {dt:.1f}s  ({ga.options.generation} gens)")

# ── 1. 逐代损失 ──
gen_rows = [r for r in hist.records if r["stage"] == "ga"]
losses = [r["loss"] for r in gen_rows]
print("gen losses:", "  ".join(f"{v:.4g}" for v in losses))
assert losses[-1] < losses[0], f"损失未下降: {losses[0]:.4g} -> {losses[-1]:.4g}"

# ── 2. 各阶段在每一代都活着（步间有变化）──
for stage in ("SA", "adam"):
    for gen in (0, 5, 9):
        vals = [
            r["spot"]
            for r in hist.records
            if r["stage"] == stage and r["gen"] == gen
        ]
        alive = len(set(vals)) > 1
        print(f"stage {stage:5s} gen {gen}: {'alive' if alive else 'FROZEN'}  {vals[0]:.4g} -> {vals[-1]:.4g}")
        assert alive, f"{stage} 在 gen {gen} 空转"

# ── 3. 精英直通 ──
snap = {n: t.clone() for n, t in seq._batched_tensors()}
seq.breed_(4)
seq.mutate_(torch.arange(4, seq.population, device=seq.device), blocks)
elite_ok = all(
    torch.equal(t[:4], snap[n][:4]) for n, t in seq._batched_tensors()
)
print("elite pass-through:", "OK" if elite_ok else "BROKEN")
assert elite_ok

# ── 4. 参数绑定 ──
bound = all(p is q for p, q in zip(params_before, seq.parameters()))
print("param identity preserved:", bound)
assert bound

print("\nALL PASS")
