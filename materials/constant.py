from typing import ClassVar

import torch

from core import RayFloatScalar, SystemLongScalar, term
from materials.protocol import MaterialDatabase

# ---- 原始数据 (名称, 折射率) ----
_CONSTANT_DATA: list[tuple[str, float]] = [
    ("vacuum", 1.0),
    ("air", 1.00027),
    ("water", 1.31984),
    ("water20C", 1.31984),
    ("water80C", 1.31044),
]

_SORTED = sorted(_CONSTANT_DATA, key=lambda r: r[0])

_CONSTANT_NAMES: tuple[str, ...] = tuple(r[0] for r in _SORTED)


class ConstantMaterialDatabase(MaterialDatabase):
    """常折射率材料数据库（不随波长变化，无法变异）。"""

    kind = term.CONSTANT
    _NAMES: ClassVar[tuple[str, ...]] = _CONSTANT_NAMES

    n: torch.Tensor  # buffer，由 __init__ 注册

    def __init__(self) -> None:
        super().__init__()
        n = torch.tensor([r[1] for r in _SORTED])
        self.register_buffer("n", n, persistent=False)

    @property
    def names(self) -> tuple[str, ...]:
        return self._NAMES

    def forward(
        self, indices: SystemLongScalar, wavelength: RayFloatScalar
    ) -> RayFloatScalar:
        n = self.n.index_select(0, indices)
        n = n.view(indices.shape[0], *([1] * (wavelength.ndim - 1)))  # (P, 1, 1, 1)
        return n.expand_as(wavelength)

    def mutate(
        self,
        indices: SystemLongScalar,
        std: float,
        generator: torch.Generator | None = None,
    ) -> SystemLongScalar:
        """常折射率材料无法变异，原样返回索引。"""
        return indices
