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
class AdamWOptions(AdamOptions):
    """AdamW：配置与 Adam 相同，仅权重衰减解耦（不进梯度、不被自适应缩放）。

    ``weight_decay = 0`` 时与 Adam 严格等价。
    """


@dataclass(slots=True)
class SGDOptions:
    step: int = 200
    default_lr: float = 1e-3
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
            default_lr=float(cfg.get("default_lr", 1e-3)),
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
    term.DIAMETER,
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
            "sgd"
            if isinstance(options, SGDOptions)
            else "adamw"
            if isinstance(options, AdamWOptions)
            else "adam"
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
            # 各组基准 lr（逐代重启退火时重置回去，见 _make_scheduler）
            self._base_lrs = [g["lr"] for g in param_groups]

            if isinstance(opts, AdamOptions):
                torch_cls = (
                    torch.optim.AdamW
                    if isinstance(opts, AdamWOptions)
                    else torch.optim.Adam
                )
                self._opt = torch_cls(
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

            # 反向端兜底：上游掩码后的 0 × NaN-Jacobian 仍可能产出 NaN 梯度
            # （前向损失有限≠反向干净），归为 0 即本步不更新该参数；
            # inf 梯度由 clip_grad_norm_ 收敛，无需映射。
            for param in seq.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0)

            if opts.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(seq.parameters(), opts.grad_norm)

            self._opt.step()
            if scheduler is not None:
                scheduler.step()

            if callbacks:
                for cb in callbacks:
                    cb.on_step_end(
                        gen,
                        step,
                        self._stage,
                        {k: v.detach().mean().item() for k, v in parts.items()},
                    )

    def _make_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        opts = self.options
        if opts.scheduler == "none" or self._opt is None:
            return None
        # 逐代重启退火：余弦/指数类调度按"当前 lr"递推，上一代末尾 lr≈0
        # 会把后续所有代锁死在 0——重建前先把各组 lr 重置回基准值。
        for group, base in zip(self._opt.param_groups, self._base_lrs):
            group["lr"] = base
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
            case _:
                raise ValueError(f"Unknown scheduler: {opts.scheduler}")
