"""遗传算法编排器：逐代调用优化器列表 → 排序 → 精英保留 → 变异。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import torch

from component import Sequential
from core import term
from optimization.annealing import SimulatedAnnealing
from optimization.callback import Callback
from optimization.gradient import GradientOptimizer
from optimization.loss import LossWeights, total_loss
from optimization.target import Target

type Stager = GradientOptimizer | SimulatedAnnealing


@dataclass(slots=True)
class GAOptions:
    """GA 选项：种群规模、代数、精英数与变异标准差退火。"""

    population: int = 1
    generation: int = 100
    topk: int | None = None
    mutate_std_start: float = 1.0
    mutate_std_end: float = 0.1
    damping: Literal["linear", "exponential", "none"] = "linear"

    def __post_init__(self) -> None:
        if self.population < 1:
            raise ValueError(f"population must be >= 1, got {self.population}")
        if self.topk is not None and not 1 <= self.topk <= self.population:
            raise ValueError(f"topk must be in [1, {self.population}], got {self.topk}")

    @classmethod
    def from_options(cls, cfg: Mapping[str, Any] | None = None) -> Self:
        """从 ``[ga]`` 配置节构造；``None`` → 全默认。"""
        if cfg is None:
            return cls()
        topk = term.TOPK.resolve(cfg, default=None)
        return cls(
            population=int(term.POPULATION.resolve(cfg, default=1)),
            generation=int(term.GENERATION.resolve(cfg, default=100)),
            topk=None if topk is None else int(topk),
            mutate_std_start=float(term.MUTATE_STD_START.resolve(cfg, default=1.0)),
            mutate_std_end=float(term.MUTATE_STD_END.resolve(cfg, default=0.1)),
            damping=term.DAMPING.resolve(cfg, default="linear"),
        )


def _scale_mutate_std(block: Mapping[str, Any], scale: float) -> dict[str, Any]:
    """按 *scale* 缩放元件块的 ``mutate`` 标准差映射；无该键原样返回。"""
    mut = term.MUTATE.resolve(block, default=None)
    if mut is None:
        return dict(block)
    return {
        **{k: v for k, v in block.items() if k not in term.MUTATE},
        term.MUTATE.canonical: {k: v * scale for k, v in mut.items()},
    }


class GeneticAlgorithm:
    """GA 编排器：每代依次跑各优化阶段，再按总损失排序、精英保留、变异。"""

    def __init__(
        self,
        options: GAOptions,
        stages: Sequence[Stager] = (),
    ) -> None:
        self.options = options
        self.stages = list(stages)

    def run(
        self,
        seq: Sequential,
        target: Target,
        blocks: Sequence[Mapping[str, Any]],
        weights: LossWeights | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
        dtype_switch: float | None = None,
    ) -> None:
        opts = self.options
        topk = opts.topk if opts.topk is not None else max(opts.population // 2, 1)
        total_gen = opts.generation
        switch_gen = int(total_gen * dtype_switch) if dtype_switch else None

        for gen in range(total_gen):
            if switch_gen is not None and gen == switch_gen < total_gen:
                # 中途切换 float64：默认 dtype 同步切换且不再还原
                # （本代起全程 f64，求解器状态经 reset() 丢弃重建）。
                torch.set_default_dtype(torch.float64)
                seq.to(dtype=torch.float64)
                for stage in self.stages:
                    if isinstance(stage, GradientOptimizer):
                        stage.reset()
                print(f"[dtype] gen {gen}/{total_gen}: float32 → float64")
            scale = _damping(
                opts.damping, gen, total_gen, opts.mutate_std_start, opts.mutate_std_end
            )
            mutate_blocks = [_scale_mutate_std(block, scale) for block in blocks]

            for stage in self.stages:
                stage.run(seq, target, gen, mutate_blocks, weights, callbacks=callbacks)

            with torch.no_grad():
                flow = seq()
                _, parts = total_loss(flow, seq, target, blocks, weights)
                loss = torch.zeros(seq.population, device=seq.device)
                for p in parts.values():
                    loss = loss.add(p.detach())
                order = loss.argsort()

            seq.sort_(order)

            if gen < total_gen - 1:
                seq.breed_(topk)
                # 精英直通下一代：仅变异 topk 之后的个体
                seq.mutate_(
                    torch.arange(topk, seq.population, device=seq.device),
                    mutate_blocks,
                )

            if callbacks:
                # 全部均值堆成一个张量再 tolist：一次 GPU 同步，而非每项一次
                keys = list(parts) + ["loss"]
                vals = torch.stack(
                    [parts[k].detach().float().mean() for k in parts] + [loss.float().mean()]
                ).tolist()
                for cb in callbacks:
                    cb.on_gen_end(gen, dict(zip(keys, vals, strict=True)))


def _damping(kind: str, gen: int, total: int, start: float, end: float) -> float:
    """变异标准差退火倍率：首代 ``start``、末代 ``end``（``none`` = 恒 start）。"""
    if kind == "none":
        return start
    progress = gen / max(total - 1, 1)
    if kind == "linear":
        return start + (end - start) * progress
    if kind == "exponential":
        return start * (end / start) ** progress
    raise ValueError(f"unknown damping: {kind!r}")
