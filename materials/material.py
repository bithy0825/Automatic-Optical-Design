from dataclasses import dataclass
from typing import Self, Any
from collections.abc import Mapping

import torch
from torch import nn

from core import RayFloatScalar, SystemLongScalar, fmt_param, term
from core.repr import render_line, render_tree, styled
from materials.protocol import MaterialDatabase
from materials.constant import ConstantMaterialDatabase
from materials.sellmeier import SellmeierMaterialDatabase


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
                f"Material indices device {self.Indices.device} "
                f"does not match database device {self.database.device}"
            )

        lo, hi = 0, len(self.database) - 1
        if not torch.all((self.Indices >= lo) & (self.Indices <= hi)):
            raise ValueError(
                f"Material indices must be in range [{lo}, {hi}], "
                f"but got {self.Indices}"
            )

    def _apply(self, fn, recurse: bool = True):
        ret = super()._apply(fn, recurse)
        self.database._apply(fn, False)  # 单例跟随迁移（原地，实例不变）
        return ret

    @classmethod
    def random(cls, population: int, database: MaterialDatabase) -> Self:
        indices = torch.randint(
            0, len(database), (population,), device=database.device, dtype=torch.long
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_names(cls, names: list[str], database: MaterialDatabase) -> Self:
        indices = torch.tensor(
            [database.index_of(name) for name in names],
            device=database.device,
            dtype=torch.long,
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_name(cls, population: int, name: str) -> Self:
        if name in ConstantMaterialDatabase._NAMES:
            database = ConstantMaterialDatabase.create()
        elif name in SellmeierMaterialDatabase._NAMES:
            database = SellmeierMaterialDatabase.create()
        else:
            raise KeyError(f"Unknown material name: {name}. ")

        indices = torch.full(
            (population,),
            database.index_of(name),
            device=database.device,
            dtype=torch.long,
        )
        return cls(indices=indices, database=database)

    @classmethod
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        param = term.MATERIAL.resolve(options)
        if param is None:
            raise KeyError(f"Missing parameter: {term.MATERIAL.canonical}")

        method = term.METHOD.resolve(param)

        if term.RAW.match(method):
            value = term.VALUE.resolve(param)
            if not isinstance(value, str):
                raise TypeError(
                    f"Expected {term.VALUE} to be a string, got {type(value)}"
                )
            return cls.from_name(population, value)

        if term.RANDOM.match(method):
            match term.DATABASE.resolve(param):
                case "constant":
                    database = ConstantMaterialDatabase.create()
                case "sellmeier":
                    database = SellmeierMaterialDatabase.create()
                case db:
                    raise ValueError(f"Unknown material database: {db}")
            return cls.random(population, database=database)

        raise ValueError(
            f"Unknown material method: {method}, expected one of {term.RAW.canonical}, {term.RANDOM.canonical}"
        )

    @torch.no_grad()
    def clone(self) -> "Material":
        """深拷贝：用独立 ``Indices`` buffer 重建一个新 Material，database 单例原样共享。

        像 Material 这类 owned-mutable 对象必须显式 clone——``copy/deepcopy``
        已被禁用。新 Material 持有克隆后的 indices，与其原对象互不干扰。
        """
        return Material(indices=self.Indices.clone(), database=self.database)

    def forward(self, wavelength: RayFloatScalar) -> RayFloatScalar:
        return self.database(self.Indices, wavelength)

    def names(self) -> list[str]:
        return [self.database.name_of(i) for i in self.Indices.tolist()]

    @torch.no_grad()
    def sort(self, order: SystemLongScalar) -> None:
        """原地排序材料编号。所有引用者自动可见。"""
        self.Indices.copy_(self.Indices.index_select(0, order))

    @torch.no_grad()
    def breed(self, topk: int) -> None:
        assert 0 < topk <= self.Indices.shape[0], (
            "Topk must be in the range (0, population]."
        )
        P = self.Indices.shape[0]
        elite = self.Indices[:topk]
        idx = torch.arange(P - topk, device=self.Indices.device).remainder(topk)
        self.Indices[topk:].copy_(elite[idx])

    @torch.no_grad()
    def mutate(
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

    下游 Component 仅能读取材料属性，无法修改。
    上游 Component 对原 :class:`Material` 的原地修改自动可见。
    """

    _material: Material

    def __repr__(self) -> str:
        return render_line(styled("MaterialRef", fmt_param(self.names())))

    @classmethod
    def from_material(cls, material: Material) -> Self:
        return cls(_material=material)

    def __call__(self, wavelength: RayFloatScalar) -> RayFloatScalar:
        return self._material(wavelength)

    def names(self) -> list[str]:
        return self._material.names()

    @property
    def device(self) -> torch.device:
        return self._material.device

    @property
    def population(self) -> int:
        return self._material.population
