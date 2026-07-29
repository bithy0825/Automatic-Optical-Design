from collections.abc import Mapping
from typing import Any
import tomllib

from core import term
from component import Sequential
from optimization.target import Target


def build_sequential(cfg: Mapping[str, Any] | str) -> Sequential:
    if isinstance(cfg, str):
        with open(cfg, "rb") as f:
            cfg = tomllib.load(f)

    if not isinstance(cfg, Mapping):
        raise TypeError(f"Invalid configuration: {cfg!r}")

    target_ = term.TARGET.resolve(cfg)
    target = Target.from_dict(target_)

    components_ = term.COMPONENT.resolve(cfg)
    for i, comp in enumerate(components_):
        if term.SOURCE.match(term.TYPE.resolve(comp)):
            components_[i] = {**components_[i], **target.to_dict()}

    components = components_

    population = term.POPULATION.resolve(term.GA.resolve(cfg))

    return Sequential.from_options(population, components)
