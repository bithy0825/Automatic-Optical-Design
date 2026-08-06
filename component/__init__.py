"""光学系统元件模块：追迹链上的各类元件。

模块组织
--------
* :mod:`component.protocol` — 抽象基类 :class:`Component`。
* :mod:`component.source` — 无限远物方光源（追迹链的生产者）。
* :mod:`component.gap` — 间隔（位姿沿光轴推进）。
* :mod:`component.refractor` — 折射器（Shape + 两侧材料）。
* :mod:`component.stop` — 光阑（flat 面圆形开口，只拦截光线）。
* :mod:`component.sensor` — 像面传感器（终端元件）。
* :mod:`component.sequential` — 元件链（完整光学系统）。
"""

from component.protocol import Component
from component.gap import Gap
from component.refractor import Refractor
from component.sensor import Sensor
from component.stop import Stop
from component.sequential import Sequential
from component.source import InfiniteSource

__all__ = [
    "Component",
    "Gap",
    "Refractor",
    "Sensor",
    "Stop",
    "Sequential",
    "InfiniteSource",
]
