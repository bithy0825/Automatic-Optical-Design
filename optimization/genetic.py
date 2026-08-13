"""遗传算法编排器：逐代调用优化器列表 → 排序 → 精英保留 → 变异。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import torch

from component import Sequential
from optimization.annealing import SimulatedAnnealing
from optimization.callback import Callback
from optimization.gradient import GradientOptimizer
from optimization.loss import LossWeights, total_loss
from optimization.target import Target

Stager = GradientOptimizer | SimulatedAnnealing


@dataclass(slots=True)
class GAOptions:
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
        if cfg is None:
            return cls()
        return cls(
            population=int(cfg.get("population", 1)),
            generation=int(cfg.get("generation", 100)),
            topk=int(cfg["topk"]) if "topk" in cfg else None,
            mutate_std_start=float(cfg.get("mutate_std_start", 1.0)),
            mutate_std_end=float(cfg.get("mutate_std_end", 0.1)),
            damping=cfg.get("damping", "linear"),
        )


class GeneticAlgorithm:
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
                torch.set_default_dtype(torch.float64)
                seq.to(dtype=torch.float64)
                for stage in self.stages:
                    if isinstance(stage, GradientOptimizer):
                        stage._opt = None
                print(f"[dtype] gen {gen}/{total_gen}: float32 → float64")
            scale = _damping(
                opts.damping, gen, total_gen, opts.mutate_std_start, opts.mutate_std_end
            )
            mutate_blocks = [
                {
                    **block,
                    "mutate": {k: v * scale for k, v in block["mutate"].items()},
                }
                if "mutate" in block
                else block
                for block in blocks
            ]

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
                # 全部均值堆成一个张量再 tolist:一次 GPU 同步,而非每项一次
                keys = list(parts) + ["loss"]
                vals = torch.stack(
                    [parts[k].detach().float().mean() for k in parts]
                    + [loss.float().mean()]
                ).tolist()
                for cb in callbacks:
                    cb.on_gen_end(gen, dict(zip(keys, vals)))


def _damping(
    kind: str,
    gen: int,
    total: int,
    start: float,
    end: float,
) -> float:
    if kind == "none":
        return start
    progress = gen / max(total - 1, 1)
    if kind == "linear":
        return start + (end - start) * progress
    if kind == "exponential":
        return start * (end / start) ** progress
    raise ValueError(f"Unknown damping: {kind!r}")
