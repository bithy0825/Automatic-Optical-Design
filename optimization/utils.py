from collections.abc import Mapping
from pathlib import Path
from typing import Any
import tomllib

import torch

from core import term
from component import Sequential
from optimization.annealing import SAOptions, SimulatedAnnealing
from optimization.genetic import Stager
from optimization.gradient import AdamOptions, GradientOptimizer, SGDOptions
from optimization.target import Target


def load_config(path: str) -> dict[str, Any]:
    """从 TOML 文件加载配置。"""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if not isinstance(cfg, Mapping):
        raise TypeError(f"Invalid configuration: {cfg!r}")
    return cfg


def build_target(cfg: Mapping[str, Any]) -> Target:
    """从配置构造目标光学系统。"""
    target_ = term.TARGET.resolve(cfg)
    return Target.from_options(target_)


def build_sequential(
    cfg: Mapping[str, Any] | str, target: Target | None = None
) -> Sequential:
    if isinstance(cfg, str):
        cfg = load_config(cfg)

    if not isinstance(cfg, Mapping):
        raise TypeError(f"Invalid configuration: {cfg!r}")

    if target is None:
        target = build_target(cfg)

    target_spec = target.to_dict()
    components = [
        {**comp, **target_spec}
        if term.SOURCE.match(term.TYPE.resolve(comp))
        else comp
        for comp in term.COMPONENT.resolve(cfg)
    ]

    population = term.POPULATION.resolve(term.GA.resolve(cfg))

    return Sequential.from_options(population, components)


def save(seq: Sequential, cfg: Mapping[str, Any], path: str | Path) -> None:
    """保存训练检查点:``(完整配置, 训练后参数状态)`` 二元组。

    配置原样附带(可无损重建系统);参数经 ``seq.state_dict()`` 序列化,
    键由模块树自动生成(含曲率、厚度、直径、材料编号等全部批量张量)。
    加载见 :func:`load`。
    """
    torch.save((dict(cfg), seq.state_dict()), path)


def load(path: str | Path) -> tuple[Sequential, Target]:
    """加载训练检查点:按附带配置重建系统并注入训练后状态(strict)。"""
    cfg, state = torch.load(path, weights_only=True, map_location="cpu")
    target = build_target(cfg)
    seq = build_sequential(cfg, target)
    seq.load_state_dict(state, strict=True)
    return seq, target


def build_stage(block: Mapping[str, Any]) -> Stager:
    """按 ``type`` 分发构造一个优化阶段（sa / adam / sgd）。"""
    match term.TYPE.resolve(block):
        case "sa":
            return SimulatedAnnealing(SAOptions.from_options(block))
        case "adam":
            return GradientOptimizer(AdamOptions.from_options(block))
        case "sgd":
            return GradientOptimizer(SGDOptions.from_options(block))
        case other:
            raise ValueError(f"Unknown optimizer type: {other!r}")
