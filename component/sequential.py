from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Self, cast, override

from torch import nn

from core import Noun, OpticalModule, SystemBoolScalar, SystemLongScalar, TraceFlow, term
from core.repr import styled
from component.protocol import Component
from component.refractor import Refractor
from component.sensor import Sensor
from component.source import InfiniteSource
from materials import Material
from shape import Shape

FlowCallback = Callable[[Component, TraceFlow, int], TraceFlow]


class Sequential(Component):
    """按序级联的光学系统。``flow=None`` 发流时首元件必须是 Source。"""

    kind = term.SEQUENTIAL

    @override
    def _label(self) -> str:
        return styled("Sequential", f"{len(self)} surfaces, P={self.population}")

    def __init__(self, *components: Component) -> None:
        super().__init__()
        self.components = nn.ModuleList(components)

    # ------------------------------------------------------------------
    # 结构
    # ------------------------------------------------------------------

    def append(self, component: Component) -> None:
        self.components.append(component)
        self.rebind()

    def rebind(self, upstream: Material | None = None) -> None:
        """沿链重绑入射材料。链外前置材料经 *upstream* 传入（注意自行克隆）。"""
        for comp in self:
            if isinstance(comp, Refractor) and upstream is not None:
                comp.bind_incident(upstream)
            if isinstance(comp, (InfiniteSource, Refractor)):
                upstream = comp.transmitted

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[Component]:
        return cast(Iterator[Component], iter(self.components))

    @override
    def __getitem__(self, key: int | slice | Noun) -> Any:
        """int/slice 取元件（或元件切片）；Noun 导出参数张量（边界损失数据口）。"""
        if isinstance(key, slice):
            return self.components[key]
        if isinstance(key, int):
            return cast(Component, self.components[key])
        return super().__getitem__(key)

    def shapes(self) -> Iterator[Shape]:
        """链上所有面形（Refractor / Sensor 的 shape），供边界损失等导出。"""
        for comp in self:
            if isinstance(comp, (Refractor, Sensor)):
                yield comp.shape

    # ------------------------------------------------------------------
    # 追迹
    # ------------------------------------------------------------------

    @override
    def forward(
        self,
        flow: TraceFlow | None = None,
        *,
        callback: FlowCallback | Iterable[FlowCallback] | None = None,
    ) -> TraceFlow:
        callbacks: tuple[FlowCallback, ...] = (
            ()
            if callback is None
            else (callback,)
            if callable(callback)
            else tuple(callback)
        )
        if flow is None and (
            len(self.components) == 0
            or not isinstance(self.components[0], InfiniteSource)
        ):
            raise RuntimeError(
                "flow=None requires the first component to be an InfiniteSource, "
                f"got {type(self.components[0]).__name__ if len(self.components) else 'empty chain'}"
            )
        for i, comp in enumerate(self):
            flow = comp(
                cast(TraceFlow, flow)
            )  # i=0 且 flow=None 时已校验首元件为 Source
            for cb in callbacks:
                flow = cb(comp, cast(TraceFlow, flow), i)
        return cast(TraceFlow, flow)

    # ------------------------------------------------------------------
    # GA / 快照
    # ------------------------------------------------------------------

    @override
    def mutate_(
        self,
        indices: SystemLongScalar,
        options: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    ) -> None:
        """逐元件下发各自的配置块（与 ``[[component]]`` 一一对应），
        各元件解出自己的 ``mutate`` 变异映射（缺省为空）。"""
        if isinstance(options, Mapping):
            raise TypeError(
                "Sequential.mutate_ expects a sequence of per-component option "
                "blocks, not a single mapping"
            )
        for comp, opts in zip(self, options, strict=True):
            comp.mutate_(indices, term.MUTATE.resolve(opts, default={}))

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐元件多态分派各自的 ``where``（命中子类实现或基类默认），随后 rebind。"""
        OpticalModule._check_operands(mask, new, old)
        components: list[Component] = []
        for n, o in zip(new, old, strict=True):
            if type(n) is not type(o):
                raise TypeError(
                    f"where: component {type(n).__name__} vs {type(o).__name__}"
                )
            components.append(type(n).where(mask, n, o))
        seq = cls(*components)
        seq.rebind()
        return cast(Self, seq)

    @override
    def clone(self) -> Self:
        """深拷贝整条链并重绑材料：新实例与原实例完全断开（快照-择优场景安全）。"""
        copy = type(self)(*(comp.clone() for comp in self))
        copy.rebind()
        return copy

    @classmethod
    @override
    def from_options(
        cls, population: int, options: Sequence[Mapping[str, Any]] | Mapping[str, Any]
    ) -> Self:
        """从整系统配置构造：``component`` 键下的元件块逐个分派，随后 rebind。"""
        if isinstance(options, Mapping):
            raise TypeError(
                "Sequential.from_options expects a sequence of per-component option "
                "blocks, not a single mapping"
            )
        blocks = list(options)
        seq = cls(*(Component.from_options(population, block) for block in blocks))
        seq.rebind()
        return cast(Self, seq)
