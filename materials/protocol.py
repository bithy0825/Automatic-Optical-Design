"""材料数据库抽象基类：按编号 + 波长计算折射率的只读参考库。

数据库是跨组件共享的只读参考数据，故为**逐子类单例**（:meth:`create` 获取，
直接构造抛错）；单例不进入任何模块的参数树，设备 / 精度迁移由
:class:`~materials.material.Material` 的 ``_apply`` 原地跟进（单进程单链
约定下的既定行为）。

子类注册：声明 ``kind`` 名词即自动进入注册表，供 :meth:`create_kind` /
:meth:`for_material` 分派。
"""

import threading
from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Self, cast

import torch
from torch import nn

from core import Noun, RayFloatScalar, SystemLongScalar
from core.repr import render_tree, styled


class MaterialDatabase(nn.Module, ABC):
    kind: ClassVar[Noun]
    _NAMES: ClassVar[tuple[str, ...]]  # 子类覆写：与 names 属性一致
    _REGISTRY: ClassVar[dict[Noun, type["MaterialDatabase"]]] = {}
    _instance: ClassVar["MaterialDatabase | None"] = None
    _lock: ClassVar[threading.RLock]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._instance = None  # 每个子类各自的单例槽
        cls._lock = threading.RLock()
        if kind := cls.__dict__.get("kind"):
            cls._REGISTRY[kind] = cls

    # ── 单例机械 ──

    def __init__(self) -> None:
        super().__init__()
        if type(self)._instance is not self:
            raise RuntimeError(
                f"{type(self).__name__} is a singleton; use "
                f"{type(self).__name__}.create() instead of direct construction"
            )

    @classmethod
    def create(cls) -> Self:
        """创建（或返回既有的）数据库单例。"""
        with cls._lock:
            if cls._instance is None:
                instance = cls.__new__(cls)
                cls._instance = instance  # 先占位：放行 __init__ 并防重入
                try:
                    instance.__init__()
                except Exception:
                    cls._instance = None
                    raise
            return cast(Self, cls._instance)

    @classmethod
    def destroy(cls) -> None:
        """销毁数据库单例（测试与流程重建用）。"""
        with cls._lock:
            cls._instance = None

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        memo[id(self)] = self
        return self

    def __reduce__(self) -> tuple:
        raise RuntimeError(f"{type(self).__name__} is a singleton and cannot be pickled")

    # ── 注册表分派 ──

    @classmethod
    def create_kind(cls, kind: str) -> "MaterialDatabase":
        """按数据库名词（``constant`` / ``sellmeier``）创建对应单例。"""
        for noun, sub in cls._REGISTRY.items():
            if noun.match(kind):
                return sub.create()
        raise ValueError(
            f"unknown material database: {kind!r} "
            f"(available: {[n.canonical for n in cls._REGISTRY]})"
        )

    @classmethod
    def for_material(cls, name: str) -> "MaterialDatabase":
        """返回收录材料 *name* 的数据库单例（未收录抛 ``KeyError``）。"""
        for sub in cls._REGISTRY.values():
            if name in sub._NAMES:
                return sub.create()
        raise KeyError(
            f"unknown material: {name!r} (searched: {[n.canonical for n in cls._REGISTRY]})"
        )

    # ── 抽象契约 ──

    @abstractmethod
    def forward(self, indices: SystemLongScalar, wavelength: RayFloatScalar) -> RayFloatScalar:
        """按材料编号与波长计算折射率。"""

    @abstractmethod
    def mutate(
        self,
        indices: SystemLongScalar,
        std: float,
        generator: torch.Generator | None = None,
    ) -> SystemLongScalar:
        """扰动 ``indices`` 并吸附到最近材料，返回变异后的编号。"""

    @property
    @abstractmethod
    def names(self) -> tuple[str, ...]:
        """材料名称元组（排序规则由子类定义，编号语义随之固定）。"""

    # ── 通用实现 ──

    @property
    def device(self) -> torch.device:
        for t in self.buffers():
            return t.device
        return torch.get_default_device()

    @property
    def dtype(self) -> torch.dtype:
        for t in self.buffers():
            if t.is_floating_point():
                return t.dtype
        return torch.get_default_dtype()

    def __len__(self) -> int:
        return len(self.names)

    @cached_property
    def name_to_index(self) -> dict[str, int]:
        """材料名称 → 编号映射（names 为类级常量，缓存一次终身有效）。"""
        return {n: i for i, n in enumerate(self.names)}

    def name_of(self, index: int) -> str:
        """材料编号 → 名称。越界抛 :class:`IndexError`。"""
        try:
            return self.names[index]
        except IndexError:
            raise IndexError(f"material index {index} out of range [0, {len(self)})") from None

    def index_of(self, name: str) -> int:
        """材料名称 → 编号。缺失抛 :class:`KeyError`。"""
        try:
            return self.name_to_index[name]
        except KeyError:
            raise KeyError(
                f"unknown material: {name!r} "
                f"(available: {len(self)} entries, e.g. {', '.join(self.names[:5])})"
            ) from None

    def __contains__(self, name: str) -> bool:
        """材料名称是否在库中。"""
        return name in self.name_to_index

    def _label(self) -> str:
        return styled(
            type(self).__name__,
            f"{len(self)} materials, {self.device}, {self.dtype}",
        )

    def __repr__(self) -> str:
        return render_tree(self)
