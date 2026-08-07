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

        all_idx = torch.arange(seq.population, device=seq.device)
        for step_m1 in range(opts.step):
            progress = (step_m1 + 1) / opts.step
            T = self._temperature(progress)

            trial = seq.clone()
            trial.mutate_(all_idx, blocks)

            with torch.no_grad():
                flow = trial()
                trial_loss, parts = total_loss(flow, trial, target, blocks, weights)

            accept = current.sub(trial_loss).div(T).exp().clamp_max(1.0)
            accept = accept.gt(torch.rand_like(accept))
            # 原地择优：组件对象（Parameter / Material / MaterialRef 链）身份不变，
            # 梯度优化器的参数绑定跨代保持有效；纯张量值回写，无需 rebind。
            seq.where_(accept, trial)
            current = torch.where(accept, trial_loss, current)

            if callbacks:
                # 全部均值堆成一个张量再 tolist:一次 GPU 同步,而非每项一次
                keys = list(parts)
                vals = torch.stack([parts[k].detach().mean() for k in keys]).tolist()
                metrics = dict(zip(keys, vals))
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
