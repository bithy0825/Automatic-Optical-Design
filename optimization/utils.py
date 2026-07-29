from collections.abc import Mapping
from typing import Any
import tomllib

from core import term
from component import Sequential
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

    components_ = term.COMPONENT.resolve(cfg)
    for i, comp in enumerate(components_):
        if term.SOURCE.match(term.TYPE.resolve(comp)):
            components_[i] = {**components_[i], **target.to_dict()}

    components = components_

    population = term.POPULATION.resolve(term.GA.resolve(cfg))

    return Sequential.from_options(population, components)
