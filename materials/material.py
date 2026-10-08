"""可变材料容器（:class:`Material`）与只读视图（:class:`MaterialRef`）。

拥有者组件持有 :class:`Material` 并可原地修改（GA 演化）；观察者组件经
:class:`MaterialRef` 引用同一对象，自动感知变化。数据库单例不在模块树内，
经 ``_apply`` 跟随迁移（既定行为，见 :mod:`materials.protocol`）。
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import torch
from torch import nn

from core import (
    RayFloatScalar,
    SystemBoolScalar,
    SystemLongScalar,
    fmt_param,
    term,
)
from core.repr import render_line, render_tree, styled
from materials.protocol import MaterialDatabase


class Material(nn.Module):
    """可变材料容器。继承 nn.Module 以融入 PyTorch 参数树。

    拥有者 Component 持有此对象并可原地修改；
    观察者 Component 通过 :class:`MaterialRef` 引用同一对象，自动感知变化。
    """

    Indices: SystemLongScalar  # buffer，由 __init__ 注册
    database: MaterialDatabase  # 单例，挂在模块树之外（见 __init__）

    def __init__(self, database: MaterialDatabase, indices: SystemLongScalar) -> None:
        super().__init__()
        # 单例数据库不属于本模块状态：挂在模块树之外，GA 递归重排不可见；
        # 设备迁移由 _apply 原地跟进——任何设备上都是同一个实例。
        object.__setattr__(self, "database", database)
        self.register_buffer(term.INDICES.canonical, indices)

        if self.Indices.device != self.database.device:
            raise ValueError(
                f"material indices device {self.Indices.device} does not match "
                f"database device {self.database.device}"
            )
        lo, hi = 0, len(self.database) - 1
        if not torch.all((self.Indices >= lo) & (self.Indices <= hi)):
            raise ValueError(f"material indices must be in [{lo}, {hi}], got {self.Indices}")

    def _apply(self, fn, recurse: bool = True):
        ret = super()._apply(fn, recurse)
        self.database._apply(fn, False)  # 单例跟随迁移（原地，实例不变）
        return ret

    @classmethod
    def random(cls, population: int, database: MaterialDatabase) -> Self:
        """在 *database* 上均匀随机抽取 *population* 个材料编号。"""
        indices = torch.randint(
            0, len(database), (population,), device=database.device, dtype=torch.long
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_names(cls, names: list[str], database: MaterialDatabase) -> Self:
        """按名称列表构造（逐名查号）。"""
        indices = torch.tensor(
            [database.index_of(name) for name in names],
            device=database.device,
            dtype=torch.long,
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_name(cls, population: int, name: str) -> Self:
        """全种群同一材料：*name* 的数据库由注册表自动定位。"""
        database = MaterialDatabase.for_material(name)
        indices = torch.full(
            (population,),
            database.index_of(name),
            device=database.device,
            dtype=torch.long,
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        """从配置构造：``{method: raw, value: 名称}`` 定死，或
        ``{method: random, database: 库名}`` 随机抽取。"""
        param = term.MATERIAL.resolve(options)
        method = term.METHOD.resolve(param)

        if term.RAW.match(method):
            value = term.VALUE.resolve(param)
            if not isinstance(value, str):
                raise TypeError(f"raw material spec expects a string, got {type(value).__name__}")
            return cls.from_name(population, value)
        if term.RANDOM.match(method):
            database = MaterialDatabase.create_kind(term.DATABASE.resolve(param))
            return cls.random(population, database=database)
        raise ValueError(
            f"unknown material method: {method!r} "
            f"(expected: {term.RAW.canonical} | {term.RANDOM.canonical})"
        )

    @torch.no_grad()
    def clone(self) -> "Material":
        """深拷贝：用独立 ``Indices`` buffer 重建，database 单例原样共享。

        像 Material 这类 owned-mutable 对象必须显式 clone——``copy/deepcopy``
        已被禁用。新 Material 持有克隆后的 indices，与其原对象互不干扰。
        """
        return Material(indices=self.Indices.clone(), database=self.database)

    @classmethod
    @torch.no_grad()
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """种群级个体选择：``mask=True`` 取 *new* 的玻璃编号，``False`` 取 *old*。

        两操作数须共享同一 database 单例（同一 clone 谱系），database 从 *new* 继承。
        """
        if type(new) is not type(old) or new.database is not old.database:
            raise TypeError("where: materials must share one database singleton")
        if mask.dtype != torch.bool or mask.shape != (new.population,):
            raise ValueError(
                f"where: mask must be a ({new.population},) bool tensor, "
                f"got shape={tuple(mask.shape)}, dtype={mask.dtype}"
            )
        return cls(indices=torch.where(mask, new.Indices, old.Indices), database=new.database)

    def forward(self, wavelength: RayFloatScalar) -> RayFloatScalar:
        """逐个体折射率（按各自编号与逐光线波长）。"""
        return self.database(self.Indices, wavelength)

    def names(self) -> list[str]:
        """逐个体材料名称。"""
        return [self.database.name_of(i) for i in self.Indices.tolist()]

    @torch.no_grad()
    def sort_(self, order: SystemLongScalar) -> None:
        """原地排序材料编号。所有引用者自动可见。"""
        self.Indices.copy_(self.Indices.index_select(0, order))

    @torch.no_grad()
    def breed_(self, topk: int) -> None:
        """用前 *topk* 个精英滚动复制填充整个种群。所有引用者自动可见。"""
        P = self.Indices.shape[0]
        assert 0 < topk <= P, f"topk must be in (0, {P}], got {topk}"
        idx = torch.arange(P - topk, device=self.Indices.device).remainder(topk)
        self.Indices[topk:].copy_(self.Indices[:topk][idx])

    @torch.no_grad()
    def mutate_(
        self,
        indices: SystemLongScalar,
        std: float,
        generator: torch.Generator | None = None,
    ) -> None:
        """原地变异材料编号。所有引用者自动可见。"""
        selected = self.Indices.index_select(0, indices)
        new_selected = self.database.mutate(selected, std, generator)
        self.Indices.index_put_((indices,), new_selected)

    def _label(self) -> str:
        return styled("Material", fmt_param(self.names()))

    def __repr__(self) -> str:
        return render_tree(self)

    def __copy__(self):
        raise RuntimeError("Material is an owned mutable object")

    def __deepcopy__(self, memo):
        raise RuntimeError("Material is an owned mutable object")

    @property
    def device(self) -> torch.device:
        return self.Indices.device

    @property
    def population(self) -> int:
        return self.Indices.shape[0]


@dataclass(frozen=True, slots=True)
class MaterialRef:
    """材料的只读视图。

    下游 Component 仅能读取材料属性，无法修改；
    上游 Component 对原 :class:`Material` 的原地修改自动可见。
    """

    _material: Material

    def __repr__(self) -> str:
        return render_line(styled("MaterialRef", fmt_param(self.names())))

    @classmethod
    def from_material(cls, material: Material) -> Self:
        """包装一个既有 :class:`Material`。"""
        return cls(_material=material)

    def __call__(self, wavelength: RayFloatScalar) -> RayFloatScalar:
        """逐个体折射率（转发本体）。"""
        return self._material(wavelength)

    def names(self) -> list[str]:
        """逐个体材料名称（转发本体）。"""
        return self._material.names()

    @property
    def device(self) -> torch.device:
        return self._material.device

    @property
    def population(self) -> int:
        return self._material.population
