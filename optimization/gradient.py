"""梯度下降优化器（Adam / SGD）。

``lr`` 配置键字段经词表名词匹配到 ``named_parameters()`` 的叶子名；
未命中走 ``default_lr``。不同 lr 值自动分组，调度一致衰减。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Self, cast

import torch

from component import Sequential
from core import term
from optimization.callback import Callback
from optimization.loss import LossWeights, total_loss
from optimization.target import Target


@dataclass(slots=True)
class AdamOptions:
    step: int = 200
    default_lr: float = 1e-3
    lr: dict[str, float] = field(default_factory=dict)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_norm: float | None = 10.0
    scheduler: Literal["cosine", "linear", "exponential", "none"] = "cosine"

    @classmethod
    def from_options(cls, cfg: Mapping[str, Any] | None = None) -> Self:
        if cfg is None:
            return cls()
        return cls(
            step=int(cfg.get("step", 200)),
            default_lr=float(cfg.get("default_lr", 1e-3)),
            lr={str(k): float(v) for k, v in cfg.get("lr", {}).items()},
            betas=cast(tuple[float, float], tuple(cfg.get("betas", (0.9, 0.999)))),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            grad_norm=cfg.get("grad_norm", 10.0),
            scheduler=cfg.get("scheduler", "cosine"),
        )


@dataclass(slots=True)
class SGDOptions:
    step: int = 200
    default_lr: float = 1e-2
    lr: dict[str, float] = field(default_factory=dict)
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_norm: float | None = 10.0
    scheduler: Literal["cosine", "linear", "exponential", "none"] = "cosine"

    @classmethod
    def from_options(cls, cfg: Mapping[str, Any] | None = None) -> Self:
        if cfg is None:
            return cls()
        return cls(
            step=int(cfg.get("step", 200)),
            default_lr=float(cfg.get("default_lr", 1e-2)),
            lr={str(k): float(v) for k, v in cfg.get("lr", {}).items()},
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
            grad_norm=cfg.get("grad_norm", 10.0),
            scheduler=cfg.get("scheduler", "cosine"),
        )


_NAME_NOUNS = (
    term.CURVATURE,
    term.KAPPA,
    term.ALPHA,
    term.THICKNESS,
)


def _lr_map(lr_config: dict[str, float]) -> dict[str, float]:
    """配置键经名词匹配 → 叶子名 → lr 映射。"""
    mapping: dict[str, float] = {}
    for key, val in lr_config.items():
        for noun in _NAME_NOUNS:
            if noun.match(key):
                mapping[noun.canonical] = val
                break
    return mapping


class GradientOptimizer:
    def __init__(self, options: AdamOptions | SGDOptions, *, stage: str = "") -> None:
        self.options = options
        self._stage = stage or (
            "adam" if isinstance(options, AdamOptions) else "sgd"
        )
        self._opt: torch.optim.Optimizer | None = None

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
        lr_by_leaf = _lr_map(opts.lr)

        if self._opt is None:
            groups: dict[float, list[torch.nn.Parameter]] = {}
            for name, param in seq.named_parameters():
                if not param.requires_grad:
                    continue
                leaf = name.rsplit(".", 1)[-1]
                lr = lr_by_leaf.get(leaf, opts.default_lr)
                groups.setdefault(lr, []).append(param)
            param_groups = [{"params": ps, "lr": lr} for lr, ps in groups.items()]

            if isinstance(opts, AdamOptions):
                self._opt = torch.optim.Adam(
                    param_groups,
                    betas=opts.betas,
                    weight_decay=opts.weight_decay,
                )
            else:
                self._opt = torch.optim.SGD(
                    param_groups,
                    momentum=opts.momentum,
                    weight_decay=opts.weight_decay,
                )

        scheduler = self._make_scheduler()

        for step in range(opts.step):
            self._opt.zero_grad(set_to_none=True)
            flow = seq()
            loss_total, parts = total_loss(flow, seq, target, blocks, weights)
            loss_total.mean().backward()

            if opts.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(seq.parameters(), opts.grad_norm)

            self._opt.step()
            if scheduler is not None:
                scheduler.step()

            if callbacks:
                for cb in callbacks:
                    cb.on_step_end(
                        gen, step, self._stage,
                        {k: v.detach().mean().item() for k, v in parts.items()},
                    )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        opts = self.options
        if opts.scheduler == "none" or self._opt is None:
            return None
        match opts.scheduler:
            case "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._opt, T_max=opts.step
                )
            case "linear":
                return torch.optim.lr_scheduler.LinearLR(
                    self._opt, start_factor=1.0, end_factor=0.01, total_iters=opts.step
                )
            case "exponential":
                return torch.optim.lr_scheduler.ExponentialLR(
                    self._opt, gamma=1e-6 ** (1.0 / opts.step)
                )
        raise ValueError(f"Unknown scheduler: {opts.scheduler}")
