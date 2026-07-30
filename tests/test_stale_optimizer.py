"""验证：SA 阶段整体更换组件后，GradientOptimizer 持有的参数引用是否失效。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimization import (
    AdamOptions,
    GAOptions,
    GeneticAlgorithm,
    LossHistory,
    SAOptions,
    SimulatedAnnealing,
    GradientOptimizer,
    build_sequential,
)
from optimization.utils import build_target, load_config

cfg = load_config(str(Path(__file__).resolve().parents[1] / "demo" / "config.toml"))
target = build_target(cfg)
seq = build_sequential(cfg, target)
blocks = list(cfg["component"])

sa = SimulatedAnnealing(SAOptions(step=3, T0=10.0, T1=0.01))
adam = GradientOptimizer(AdamOptions(step=3, scheduler="none"))
hist = LossHistory()

ga = GeneticAlgorithm(
    GAOptions(population=8, generation=3, topk=4), stages=[sa, adam]
)
ga.run(seq, target, blocks, callbacks=[hist])

assert adam._opt is not None, "optimizer 未创建"
opt_params = [p for g in adam._opt.param_groups for p in g["params"]]
seq_params = list(seq.parameters())
same = sum(1 for p in opt_params if any(p is q for q in seq_params))
print(f"optimizer params still bound to seq: {same}/{len(opt_params)}")

# gen>=1 的 adam 阶段 loss 是否逐步变化（不变化 = step 空转）
adam_rows = [r for r in hist.records if r["stage"] == "adam"]
for r in adam_rows:
    print(f"gen={r['gen']} step={r['step']} loss={r.get('spot', 0):.6g}")
