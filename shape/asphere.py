from typing import Any, override, Self
from collections.abc import Mapping
import warnings

import torch

from core import (
    Noun,
    OpticalModule,
    SystemBoolScalar,
    SystemFloatScalar,
    SystemFloatND,
    SystemLongScalar,
    init_param,
    term,
    parse_param,
)
from implicit import NewtonSolverOptions, SagFunction, aspheric_sag
from shape.protocol import Shape


class Asphere(Shape):
    kind = term.ASPHERE
    mutable = (term.DIAMETER, term.CURVATURE, term.KAPPA, term.ALPHA, term.MASK)

    def __init__(
        self,
        diameter: SystemFloatScalar,
        curvature: SystemFloatScalar,
        kappa: SystemFloatScalar,
        alpha: SystemFloatND,
        mask: SystemFloatScalar | None = None,
        *,
        solver_opts: NewtonSolverOptions | Mapping[str, Any] | None = None,
        trainable: Mapping[str, bool] | None = None,
    ):
        super().__init__(diameter, solver_opts=solver_opts, trainable=trainable)

        train_C = False
        train_K = False
        train_A = False
        for k in self.trainable:
            if (
                not term.CURVATURE.match(k)
                and not term.KAPPA.match(k)
                and not term.ALPHA.match(k)
            ):
                warnings.warn(
                    f"Unknown trainable key: {k}. Only 'curvature', 'kappa', 'alpha' are supported for Asphere."
                )
            else:
                if term.CURVATURE.match(k):
                    train_C = self.trainable[k]
                if term.KAPPA.match(k):
                    train_K = self.trainable[k]
                if term.ALPHA.match(k):
                    train_A = self.trainable[k]

        self.Curvature = init_param(self, term.CURVATURE, curvature, train_C)
        self.Kappa = init_param(self, term.KAPPA, kappa, train_K)
        self.Alpha = init_param(self, term.ALPHA, alpha, train_A)

        if mask is None:
            mask = torch.full_like(self.Diameter, float(self.Alpha.shape[-1]))
        # mask 语义 = 激活系数个数（{0..n} 的整数计数，float 存储但值须为整数）
        n_coeffs = self.Alpha.shape[-1]
        if not torch.all(mask == mask.round()):
            raise ValueError(f"mask must be integer-valued, got {mask.tolist()}")
        if not torch.all((mask >= 0.0) & (mask <= float(n_coeffs))):
            raise ValueError(f"mask must be in [0, {n_coeffs}], got {mask.tolist()}")
        self.Mask = init_param(self, term.MASK, mask, False)

    def _active_alpha(self) -> SystemFloatND:
        """按 mask 屏蔽尾部系数的纯函数视图（零副作用，可训练参数安全）。"""
        active = torch.arange(self.Alpha.shape[-1], device=self.device).lt(
            self.Mask.unsqueeze(-1)
        )
        return self.Alpha.mul(active)

    @override
    def sag(self) -> SagFunction:
        alpha = self._active_alpha()
        radius = self.Diameter.mul(0.5)
        return aspheric_sag(self.Curvature, self.Kappa, alpha, radius)

    def _jitter(self, key: Noun, indices: SystemLongScalar, std: float) -> None:
        if std == 0:
            return
        tensor = getattr(self, key.canonical)
        noise = torch.randn_like(tensor[indices]).mul(std)
        tensor.index_put_((indices,), noise, accumulate=True)

    @override
    def mutate(self, indices: SystemLongScalar, options: Mapping[str, Any]) -> None:
        self._jitter(term.ALPHA, indices, term.ALPHA.resolve(options, default=0.0))

        std_diameter = term.DIAMETER.resolve(options, default=0.0)
        if std_diameter != 0:
            powers = torch.arange(
                4,
                4 + 2 * self.Alpha.shape[-1],
                2,
                device=self.device,
                dtype=self.Diameter.dtype,
            )
            # α ∝ ρ^p：按比值 (ρ_new/ρ_old)^p 一步重缩放，物理面形保持不变。
            # 未变异个体比值为 1，Alpha 逐位不变（不经 α_phys 中间量，无精度损耗）。
            rho_old = self.Diameter.mul(0.5).unsqueeze(-1)
            self._jitter(term.DIAMETER, indices, std_diameter)
            rho_new = self.Diameter.mul(0.5).unsqueeze(-1)
            scale = rho_new.div(rho_old).pow(powers)
            self.Alpha.copy_(self.Alpha.mul(scale))

        self._jitter(
            term.CURVATURE, indices, term.CURVATURE.resolve(options, default=0.0)
        )
        self._jitter(term.KAPPA, indices, term.KAPPA.resolve(options, default=0.0))

        std_mask = term.MASK.resolve(options, default=0.0)
        if std_mask != 0:
            # mask 变异 = 计数的圆整高斯步进：步长 ~ N(0, std_mask) 取整（多为 ±1），
            # clamp 到 [0, n]。std_mask 为步长标准差。
            counts = self.Mask[indices]
            stepped = counts.add(torch.randn_like(counts).mul(std_mask)).round()
            self.Mask[indices] = stepped.clamp(0.0, float(self.Alpha.shape[-1]))

    def __getitem__(self, key: Noun) -> torch.Tensor:
        # ALPHA 导出 mask 屏蔽后的有效系数（边界损失不惩罚无效尾部），梯度链保留。
        if key == term.ALPHA:
            return self._active_alpha()
        return getattr(self, key.canonical)

    @override
    def clone(self) -> Self:
        return type(self)(
            diameter=self.Diameter.clone(),
            curvature=self.Curvature.clone(),
            kappa=self.Kappa.clone(),
            alpha=self.Alpha.clone(),
            mask=self.Mask.clone(),
            solver_opts=self._solver_opts,
            trainable=self.trainable.copy(),
        )

    @classmethod
    @override
    def where(cls, mask: SystemBoolScalar, new: Self, old: Self) -> Self:
        """逐个体选择各面形参数；``Alpha`` 为 ``(P, N)``，mask 升维广播。
        求解器与 trainable 配置从 *new* 继承。"""
        OpticalModule._check_operands(mask, new, old)
        return cls(
            diameter=torch.where(mask, new.Diameter, old.Diameter),
            curvature=torch.where(mask, new.Curvature, old.Curvature),
            kappa=torch.where(mask, new.Kappa, old.Kappa),
            alpha=torch.where(mask.unsqueeze(-1), new.Alpha, old.Alpha),
            mask=torch.where(mask, new.Mask, old.Mask),
            solver_opts=new._solver_opts,
            trainable=new.trainable.copy(),
        )

    @classmethod
    @override
    def from_options(cls, population: int, options: Mapping[str, Any]) -> Self:
        return cls(
            diameter=parse_param(options, term.DIAMETER, population),
            curvature=parse_param(options, term.CURVATURE, population),
            kappa=parse_param(options, term.KAPPA, population),
            alpha=_alpha_terms(options, population),
            mask=(
                parse_param(options, term.MASK, population)
                if term.MASK.resolve(options, default=None) is not None
                else None
            ),
            solver_opts=term.SOLVER.resolve(options, default={}),
            trainable=term.TRAIN.resolve(options, default={}),
        )


def _alpha_order_of(key: str) -> int | None:
    for name in term.ALPHA.all_names:
        if not key.startswith(name):
            continue
        suffix = key[len(name) :]
        if not suffix.isdecimal():
            return None
        order = int(suffix)
        if order < 4 or order % 2:
            raise ValueError(f"Invalid alpha coefficient name: {key}")
        return order
    return None


def _alpha_terms(options: Mapping[str, Any], population: int) -> torch.Tensor:
    order_to_key = {}
    for key in options:
        order = _alpha_order_of(key)
        if order is None:
            continue
        if order in order_to_key:
            raise ValueError(f"Duplicate alpha coefficient for order {order}: {key}")
        order_to_key[order] = key

    if not order_to_key:
        raise ValueError("No alpha coefficients found in options")

    sorted_orders = sorted(order_to_key.keys())
    expected = list(range(4, 4 + 2 * len(sorted_orders), 2))
    if sorted_orders != expected:
        missing = set(expected) - set(sorted_orders)
        raise ValueError(f"Missing alpha coefficients for orders {missing}")

    return torch.stack(
        [
            parse_param(options, order_to_key[order], population)
            for order in sorted_orders
        ],
        dim=-1,
    )
