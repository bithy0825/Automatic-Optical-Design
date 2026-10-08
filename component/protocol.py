"""元件抽象基类：``TraceFlow`` 的消费者 / 生产者与注册表分派。

子类声明 ``kind`` 名词即自动注册；``from_options`` 按 ``type`` 键分派。
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Self, cast

from core import Noun, OpticalModule, TraceFlow, term


class Component(OpticalModule, ABC):
    """光学系统中的一个元件：消费并产出 :class:`~core.flow.TraceFlow`。"""

    kind: ClassVar[Noun]
    _REGISTRY: ClassVar[dict[Noun, type["Component"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if kind := cls.__dict__.get("kind"):
            cls._REGISTRY[kind] = cls

    @abstractmethod
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """追迹入口：消费上游 ``flow``，产出下游 ``flow``。"""

    @classmethod
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        if cls is not Component:
            raise NotImplementedError(f"{cls.__name__} must implement from_options()")
        kind = term.TYPE.resolve(options)
        for noun, sub in cls._REGISTRY.items():
            if kind in noun:
                return cast(Self, sub.from_options(population, options))
        raise ValueError(
            f"unknown component: {kind!r} (available: {[n.canonical for n in cls._REGISTRY]})"
        )
