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
    ) -> None:
        opts = self.options
        topk = opts.topk or opts.population // 2
        total_gen = opts.generation

        for gen in range(total_gen):
            scale = _damping(opts.damping, gen, total_gen, opts.mutate_std_start, opts.mutate_std_end)
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

            seq.sort(order)

            if gen < total_gen - 1:
                seq.breed(topk)
                seq.mutate(torch.arange(seq.population), mutate_blocks)

            if callbacks:
                metrics = {k: v.detach().float().mean().item() for k, v in parts.items()}
                metrics["loss"] = loss.float().mean().item()
                for cb in callbacks:
                    cb.on_gen_end(gen, metrics)


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
