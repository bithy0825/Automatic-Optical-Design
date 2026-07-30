"""模拟退火：无梯度，Metropolis 接受 + ``where`` 逐个体择优。"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self

import torch

from component import Sequential
from optimization.callback import Callback
from optimization.loss import LossWeights, total_loss
from optimization.target import Target


@dataclass(slots=True)
class SAOptions:
    step: int = 200
    T0: float = 1.0
    T1: float = 0.001
    cooling: Literal["exponential", "linear", "logarithmic"] = "exponential"

    @classmethod
    def from_options(cls, cfg: Mapping[str, Any] | None = None) -> Self:
        if cfg is None:
            return cls()
        return cls(
            step=int(cfg.get("step", 200)),
            T0=float(cfg.get("T0", 1.0)),
            T1=float(cfg.get("T1", 0.001)),
            cooling=cfg.get("cooling", "exponential"),
        )


class SimulatedAnnealing:
    def __init__(self, options: SAOptions) -> None:
        self.options = options

    def run(
        self,
        seq: Sequential,
        target: Target,
        gen: int,
        blocks: Sequence[Mapping[str, Any]],
        weights: LossWeights | None = None,
        *,
        callbacks: Sequence[Callback] | None = None,
    ) -> None:
        opts = self.options

        with torch.no_grad():
            flow = seq()
            current, _ = total_loss(flow, seq, target, blocks, weights)

        for step_m1 in range(opts.step):
            progress = (step_m1 + 1) / opts.step
            T = self._temperature(progress)

            trial = seq.clone()
            trial.mutate(torch.arange(seq.population), blocks)

            with torch.no_grad():
                flow = trial()
                trial_loss, parts = total_loss(flow, trial, target, blocks, weights)

            accept = current.sub(trial_loss).div(T).exp().clamp_max(1.0)
            accept = accept.gt(torch.rand_like(accept))
            new_seq = Sequential.where(accept, trial, seq)
            seq.components = new_seq.components
            seq.rebind()
            current = torch.where(accept, trial_loss, current)

            if callbacks:
                metrics = {k: v.detach().mean().item() for k, v in parts.items()}
                metrics["temperature"] = T
                for cb in callbacks:
                    cb.on_step_end(gen, step_m1, "SA", metrics)

    def _temperature(self, progress: float) -> float:
        opts = self.options
        match opts.cooling:
            case "exponential":
                return opts.T0 * (opts.T1 / opts.T0) ** progress
            case "linear":
                return opts.T0 + (opts.T1 - opts.T0) * progress
            case "logarithmic":
                return opts.T0 / (1.0 + progress * (opts.T0 / opts.T1 - 1.0))
