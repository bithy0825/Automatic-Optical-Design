"""逐个体 4×4 齐次变换：正向 / 逆向矩阵成对维护。

正 / 逆矩阵在构造时尽量融合分配（同一显存块切视图）；复合用 ``@`` 或
:meth:`Transformer.chain`：``A @ B`` 的正向矩阵为 ``A.forward @ B.forward``
（先应用 ``B`` 再应用 ``A``），逆向按反序自动复合，全程无需显式求逆。
"""

from dataclasses import dataclass
from functools import reduce
from operator import matmul
from typing import Self

import torch
import torch.nn.functional as F

from core.aliases import HomMatrix, RayFloat3D, SystemFloat3D, SystemFloatScalar
from core.container import TensorContainer


@dataclass(slots=True, eq=False, repr=False)
class Transformer(TensorContainer):
    """正 / 逆变换矩阵成对维护的 4×4 齐次变换。"""

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
        """恒等变换（``forward`` 为零拷贝展开视图，``inverse`` 为独立副本）。"""
        eye = torch.eye(4, device=device, dtype=dtype).expand(population, -1, -1)
        return cls(forward=eye, inverse=eye.clone())

    @classmethod
    def identity_like(cls, ref: HomMatrix) -> Self:
        """与 *ref* 同种群、同设备、同精度的恒等变换。"""
        return cls.identity(ref.shape[0], device=ref.device, dtype=ref.dtype)

    @classmethod
    def translation(cls, translation: SystemFloat3D) -> Self:
        """平移变换。"""
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
        """绕过原点 *axis* 轴旋转 *angle*（rad，右手定则）。"""
        if axis.shape[0] != angle.shape[0]:
            raise ValueError(
                f"axis and angle must share the same population, "
                f"got {axis.shape[0]} and {angle.shape[0]}"
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
        rot = c.mul(eye3).add(K.mul(s)).add(outer.mul(c.neg().add(1.0)))  # 罗德里格斯公式

        bottom = F.pad(k.new_ones(B, 1), (3, 0)).unsqueeze(-2)
        fwd = torch.cat((F.pad(rot, (0, 1)), bottom), dim=-2)
        inv = torch.cat((F.pad(rot.mT, (0, 1)), bottom), dim=-2)
        return cls(forward=fwd, inverse=inv)

    @classmethod
    def scaling(cls, scale: SystemFloatScalar) -> Self:
        """均匀缩放变换。"""
        B = scale.shape[0]
        diag = torch.cat((scale.unsqueeze(-1).expand(-1, 3), scale.new_ones(B, 1)), -1)
        both = torch.diag_embed(torch.cat((diag, diag.reciprocal())))
        return cls(forward=both[:B], inverse=both[B:])

    @classmethod
    def from_forward(cls, forward: HomMatrix) -> Self:
        """由正向矩阵构造（逆向显式求逆一次）。"""
        return cls(forward=forward, inverse=forward.inverse())

    @classmethod
    def from_inverse(cls, inverse: HomMatrix) -> Self:
        """由逆向矩阵构造（正向显式求逆一次）。"""
        return cls(forward=inverse.inverse(), inverse=inverse)

    @classmethod
    def chain(cls, *transformers: Self) -> Self:
        """依序复合：``chain(A, B, C) == A @ B @ C``。"""
        if not transformers:
            raise ValueError("chain requires at least one transformer")
        return reduce(matmul, transformers)

    def then(self, *transformers: Self) -> Self:
        """在本变换之后继续复合：``a.then(b, c) == a @ b @ c``。"""
        return type(self).chain(self, *transformers)

    def flip(self) -> Self:
        """正 / 逆互换的反向变换。"""
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

    def transform_points(self, points: RayFloat3D, *, inverse: bool = False) -> RayFloat3D:
        """变换点（仿射 = 线性部 + 平移部）。"""
        # 免去齐次坐标拼接，3×3 矩阵乘更省。
        m = self.inverse if inverse else self.forward
        out = torch.einsum("pij,p...j->p...i", m[..., :3, :3], points)
        offset = m[..., :3, 3].reshape(m.shape[0], *([1] * (points.ndim - 2)), 3)
        return out.add(offset)

    def transform_vectors(self, vectors: RayFloat3D, *, inverse: bool = False) -> RayFloat3D:
        """变换方向矢量（仅线性部，不含平移）。"""
        m = self.inverse if inverse else self.forward
        return torch.einsum("pij,p...j->p...i", m[..., :3, :3], vectors)
