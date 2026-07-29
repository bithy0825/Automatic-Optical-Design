from dataclasses import dataclass
from typing import Self
from functools import reduce
from operator import matmul

import torch
import torch.nn.functional as F

from core.aliases import HomMatrix, SystemFloat3D, SystemFloatScalar, RayFloat3D
from core.container import TensorContainer


@dataclass(slots=True, eq=False)
class Transformer(TensorContainer):
    forward: HomMatrix
    inverse: HomMatrix

    @classmethod
    def identity(
        cls,
        population: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        eye = torch.eye(4, device=device, dtype=dtype).expand(population, -1, -1)
        return cls(forward=eye, inverse=eye.clone())

    @classmethod
    def identity_like(cls, ref: HomMatrix) -> Self:
        return cls.identity(ref.shape[0], device=ref.device, dtype=ref.dtype)

    @classmethod
    def translation(cls, translation: SystemFloat3D) -> Self:
        # 正/逆矩阵同一批构造成一块连续显存，再切视图；正向末列 t、逆向末列 −t。
        B = translation.shape[0]
        t2 = torch.cat((translation, translation.neg()))
        eye3 = torch.eye(3, device=translation.device, dtype=translation.dtype)
        top = torch.cat((eye3.expand(2 * B, -1, -1), t2.unsqueeze(-1)), dim=-1)
        bottom = F.pad(translation.new_ones(1, 1), (3, 0)).unsqueeze(-2)
        both = torch.cat((top, bottom.expand(2 * B, -1, -1)), dim=-2)
        return cls(forward=both[:B], inverse=both[B:])

    @classmethod
    def rotation(cls, axis: SystemFloat3D, angle: SystemFloatScalar) -> Self:
        if axis.shape[0] != angle.shape[0]:
            raise ValueError(
                f"Population mismatch: axis.shape[0]={axis.shape[0]} != angle.shape[0]={angle.shape[0]}"
            )

        B = axis.shape[0]
        k = F.normalize(axis, dim=-1)
        c = angle.cos().view(B, 1, 1)
        s = angle.sin().view(B, 1, 1)
        x, y, z = k.unbind(dim=-1)
        zeros = torch.zeros_like(angle)
        K = torch.stack(
            [
                torch.stack([zeros, -z, y], dim=-1),
                torch.stack([z, zeros, -x], dim=-1),
                torch.stack([-y, x, zeros], dim=-1),
            ],
            dim=-2,
        )
        outer = k.unsqueeze(-1).mul(k.unsqueeze(-2))
        eye3 = torch.eye(3, device=axis.device, dtype=axis.dtype)
        rot = c.mul(eye3).add(K.mul(s)).add(outer.mul(c.neg().add(1.0)))  # 罗德里格斯旋转公式

        bottom = F.pad(k.new_ones(B, 1), (3, 0)).unsqueeze(-2)
        fwd = torch.cat((F.pad(rot, (0, 1)), bottom), dim=-2)
        inv = torch.cat((F.pad(rot.mT, (0, 1)), bottom), dim=-2)
        return cls(forward=fwd, inverse=inv)

    @classmethod
    def scaling(cls, scale: SystemFloatScalar) -> Self:
        B = scale.shape[0]
        diag = torch.cat((scale.unsqueeze(-1).expand(-1, 3), scale.new_ones(B, 1)), -1)
        both = torch.diag_embed(torch.cat((diag, diag.reciprocal())))
        return cls(forward=both[:B], inverse=both[B:])

    @classmethod
    def from_forward(cls, forward: HomMatrix) -> Self:
        return cls(forward=forward, inverse=forward.inverse())

    @classmethod
    def from_inverse(cls, inverse: HomMatrix) -> Self:
        return cls(forward=inverse.inverse(), inverse=inverse)

    @classmethod
    def chain(cls, *transformers: Self) -> Self:
        if not transformers:
            raise ValueError("At least one transformer is required to chain.")
        return reduce(matmul, transformers)

    def then(self, *transformers: Self) -> Self:
        return type(self).chain(self, *transformers)

    def flip(self) -> Self:
        return type(self)(forward=self.inverse, inverse=self.forward)

    def __matmul__(self, other: Self) -> Self:
        return type(self)(
            forward=self.forward @ other.forward, inverse=other.inverse @ self.inverse
        )

    @property
    def device(self) -> torch.device:
        return self.forward.device

    @property
    def dtype(self) -> torch.dtype:
        return self.forward.dtype

    @property
    def population(self) -> int:
        return self.forward.shape[0]

    def transform_points(
        self, points: RayFloat3D, *, inverse: bool = False
    ) -> RayFloat3D:
        # 仿射 = 线性部 + 平移部：免去齐次坐标拼接，3×3 矩阵乘更省。
        m = self.inverse if inverse else self.forward
        out = torch.einsum("pij,p...j->p...i", m[..., :3, :3], points)
        offset = m[..., :3, 3].reshape(m.shape[0], *([1] * (points.ndim - 2)), 3)
        return out.add(offset)

    def transform_vectors(
        self, vectors: RayFloat3D, *, inverse: bool = False
    ) -> RayFloat3D:
        m = self.inverse if inverse else self.forward
        return torch.einsum("pij,p...j->p...i", m[..., :3, :3], vectors)
