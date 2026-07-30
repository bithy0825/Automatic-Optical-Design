from typing import Any, Self, override
from collections.abc import Iterator, Mapping
from dataclasses import replace

from core import OpticalModule, SystemBoolScalar, SystemLongScalar, TraceFlow, fmt_param, term
from component.protocol import Component
from materials import Material, MaterialRef
from physics import refract
from shape import Shape


class Refractor(Component):
    """折射器：一个 Shape + 两侧材料，执行求交与 Snell 折射。

    拥有者持有出射侧 :class:`~materials.material.Material`（可原地变异）；
    入射侧经 :class:`~materials.material.MaterialRef` 引用上游材料。
    """

    kind = term.REFRACTOR

    def __init__(
        self,
        *,
        shape: Shape,
        transmitted: Material,
        incident: MaterialRef | None = None,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.transmitted = transmitted
        self.incident = incident

    @override
    def _params(self) -> Iterator[str]:
        incident = (
            fmt_param(self.incident.names())
            if self.incident is not None
            else "(unbound)"
        )
        yield f"incident: {incident}"
        yield f"transmitted: {fmt_param(self.transmitted.names())}"

    def bind_incident(self, transmitted_prev: Material) -> None:
        """绑定上游入射材料（可选，若不绑定则折射器无法工作）。"""
        self.incident = MaterialRef.from_material(transmitted_prev)

    @override
    def forward(self, flow: TraceFlow) -> TraceFlow:
        """求交 → Snell 折射 → 推进光线；裁决沿流链入（上游已死光线保持死亡）。"""
        if self.incident is None:
            raise RuntimeError(
                f"{type(self).__name__}.incident is None; bind it to an upstream "
                "transmitted material before tracing."
            )

        hit = self.shape(flow.rays.points, flow.rays.directions, flow.transformer)
        res = refract(
            flow.rays.directions,
            hit.normals,
            self.incident(flow.rays.wavelength),
            self.transmitted(flow.rays.wavelength),
        )

        rays = replace(flow.rays, points=hit.points_global, directions=res.directions)
        return flow.with_rays(rays).at_verdict(hit.verdict).at_verdict(res.verdict)

    @override
    def mutate_(self, indices: SystemLongScalar, options: Mapping[str, Any]) -> None:
        """shape 按其 ``mutable`` 变异指定个体；材料按 ``MATERIAL`` 标准差整体变异。"""
        self.shape.mutate_(indices, options)
        std = term.MATERIAL.resolve(options, default=0.0)
        if std != 0.0:
            self.transmitted.mutate_(indices, std)

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            shape=Shape.from_options(population, options),
            transmitted=Material.from_options(population, options),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """shape 多态分派、transmitted 经 ``Material.where`` 合并；
        ``incident`` 留空，由链式容器 rebind（语义同 ``clone``）。"""
        OpticalModule._check_operands(mask, new, old)
        if type(new.shape) is not type(old.shape):
            raise TypeError(
                f"where: shape {type(new.shape).__name__} vs {type(old.shape).__name__}"
            )
        return cls(
            shape=type(new.shape).where(mask, new.shape, old.shape),
            transmitted=Material.where(mask, new.transmitted, old.transmitted),
        )

    @override
    def clone(self) -> Self:
        """深拷贝：形状与**出射材料**各自克隆，生成独立的新 Material。

        陷门：``transmitted`` 必须克隆为新 Material（独立 Indices），否则
        两个系统会共享同一材料状态——``mutate`` / GA 演化会跨个体泄漏。
        ``incident`` 留空，由链式容器在重构链时重绑到上游克隆体的
        Material 视图。
        """
        return type(self)(
            shape=self.shape.clone(),
            transmitted=self.transmitted.clone(),
        )
