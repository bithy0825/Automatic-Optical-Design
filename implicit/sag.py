import torch

from core import (
    RayFloat2D,
    SystemFloatND,
    SystemFloatScalar,
    Verdict,
    broadcast_system_to_ray,
    sturdy_div,
    sturdy_sqrt,
)
from implicit._tensor_utils import _broadcast_coeff, _sym2x2
from implicit.protocol import FieldResult, SagFunction

_Order = FieldResult.Order


def _conic_core(
    curvature: SystemFloatScalar,
    kappa: SystemFloatScalar,
) -> SagFunction:
    """圆锥曲面矢高闭包（球/椭球/抛物面/双曲面）。

    返回 ``s(r) = c·r² / (1 + √(1 − (1 + κ)·c²·r²))`` 及其一阶/二阶导。
    :func:`spherical_sag` 与 :func:`conical_sag` 共用本实现，仅 ``kappa`` 取值不同。
    """

    def sag(points: RayFloat2D, *, order: _Order) -> FieldResult:
        x, y = points.unbind(dim=-1)
        c = broadcast_system_to_ray(curvature, x)
        k = broadcast_system_to_ray(kappa, x)

        r2 = x.square().add(y.square())
        c2 = c.square()
        one_plus_k = k.add(1.0)
        radicand = r2.mul(c2).mul(one_plus_k).neg().add(1.0)  # 1 − (1+κ)c²r²
        in_domain = radicand.ge(0.0)  # 界内 True；越界 False —— 诚实死亡
        s = sturdy_sqrt(radicand)

        # 分母 s+1 ≥ 1 恒安全，无需 sturdy_div。越界区乘 0：与 grad/hess
        # （经 sturdy_div(c, 0)=0 已为零）一致，整个矢高在 OOB 退化为零而非
        # c·r² 的抛物外推。死亡由 verdict 标记。
        val = r2.mul(c).div(s.add(1.0)).mul(in_domain)

        grad = None
        if order >= _Order.GRADIENT:
            denom = sturdy_div(c, s)  # ds/dr² 的核心因子 c/s
            grad = torch.stack((x.mul(denom), y.mul(denom)), dim=-1)

        hess = None
        if order >= _Order.HESSIAN:
            c2k = c2.mul(one_plus_k)
            hess_scale = sturdy_div(c, radicand.mul(s))
            g_xx = radicand.add(c2k.mul(x.square())).mul(hess_scale)
            g_xy = c2k.mul(x.mul(y)).mul(hess_scale)
            g_yy = radicand.add(c2k.mul(y.square())).mul(hess_scale)
            hess = _sym2x2(g_xx, g_xy, g_yy)

        verdict = Verdict.site(
            hold=in_domain, toll=radicand, cause=Verdict.Cause.SAG_DOMAIN
        )

        return FieldResult(_value=val, _verdict=verdict, _gradient=grad, _hessian=hess)

    return sag


def spherical_sag(curvature: SystemFloatScalar) -> SagFunction:
    """球面矢高：圆锥常数 ``κ = 0`` 的特例。"""
    kappa = torch.zeros_like(curvature)
    return _conic_core(curvature, kappa)


def conical_sag(
    curvature: SystemFloatScalar,
    kappa: SystemFloatScalar,
) -> SagFunction:
    """标准圆锥曲面矢高。

    Args:
        curvature: 曲率 ``c``，形状 ``[P]``。
        kappa: 圆锥常数 ``κ``，形状 ``[P]``。``κ=0`` 退化为球面、``κ=1`` 抛物面、
            ``κ∈(-1,0)`` 椭球、``κ<-1`` 双曲面。
    """
    return _conic_core(curvature, kappa)


def aspheric_sag(
    curvature: SystemFloatScalar,
    kappa: SystemFloatScalar,
    alpha: SystemFloatND,
    normalization: SystemFloatScalar,
) -> SagFunction:
    """圆锥基底 + 偶次非球面多项式矢高。

    ``z(r) = z_conic(r) + Σ_i α_i·(r/ρ)^(4+2i)``，其中 *ρ* 为归一化半径。
    先计算 ``u = r²/ρ²`` 再对 *u* 取幂，避免大半径高次幂的数值溢出。
    归一化域过滤由外界负责，此处不做重复裁决。

    Args:
        curvature: 曲率 ``c``，形状 ``[P]``。
        kappa: 圆锥常数 ``κ``，形状 ``[P]``。
        alpha: 多项式系数，形状 ``[P, Ncoeff]``（已缩放至统一数量级）。
        normalization: 归一化半径 *ρ*，形状 ``[P]``。
    """

    conic_fn = _conic_core(curvature, kappa)

    # 小指数张量预算一次（闭包随 forward 每次重建，dtype/device 跟随当前参数）
    i = torch.arange(alpha.shape[-1], dtype=alpha.dtype, device=alpha.device)
    p1 = i.add(1.0)          # u 的幂次：1 + i
    p2 = i.add(2.0)          # u 的幂次：2 + i
    c1 = i.mul(2.0).add(4.0)  # 系数：4 + 2i
    c12 = c1.mul(p1)         # 系数：(4 + 2i)(1 + i)

    def sag_fn(points: RayFloat2D, *, order: _Order) -> FieldResult:
        x, y = points.unbind(dim=-1)
        r2 = x.square().add(y.square())
        dtype = r2.dtype

        conic = conic_fn(points, order=order)

        # 有效域沿用圆锥基底裁决（归一化域由外界保证，此处不过滤）
        valid = conic.verdict.hold.to(dtype=dtype)

        # ── 多项式（先算 u = r²/ρ² 再取幂，避免大半径高次幂溢出）──
        rho_sq = broadcast_system_to_ray(normalization, r2).square()
        u = sturdy_div(r2, rho_sq)            # u = r²/ρ²
        alpha_b = _broadcast_coeff(alpha, r2)  # (P,F,W,N,Ncoeff)
        u_e = u.unsqueeze(-1)                  # [...,1]

        # 矢高: Σ α_n · u^(2+n)
        sag_poly = alpha_b.mul(u_e.pow(p2)).sum(dim=-1).mul(valid)

        grad = None
        hess = None
        if order >= _Order.GRADIENT:
            # T = 2·ds/d(r²) = Σ α_n·(4+2n)·u^(1+n) / ρ²
            T = alpha_b.mul(c1).mul(u_e.pow(p1)).sum(dim=-1).div(rho_sq).mul(valid)
            grad_poly = torch.stack((x.mul(T), y.mul(T)), dim=-1)
            grad = conic.gradient.add(grad_poly)

            if order >= _Order.HESSIAN:
                # T' = dT/d(r²) = Σ α_n·(4+2n)(1+n)·u^n / ρ⁴
                Tprime = (
                    alpha_b.mul(c12)
                    .mul(u_e.pow(i))
                    .sum(dim=-1)
                    .div(rho_sq.square())
                    .mul(valid)
                )
                xx = T.add(x.square().mul(Tprime).mul(2.0))
                xy = x.mul(y).mul(Tprime).mul(2.0)
                yy = T.add(y.square().mul(Tprime).mul(2.0))
                hess = conic.hessian.add(_sym2x2(xx, xy, yy))

        return FieldResult(
            _value=conic.value.add(sag_poly),
            _verdict=conic.verdict,
            _gradient=grad,
            _hessian=hess,
        )

    return sag_fn


def flat_sag() -> SagFunction:
    """平面矢高：恒为零，各阶导数亦为零。"""

    def sag_fn(points: RayFloat2D, *, order: _Order) -> FieldResult:
        x = points[..., 0]
        val = torch.zeros_like(x)

        grad = None
        if order >= _Order.GRADIENT:
            grad = torch.zeros_like(points)

        hess = None
        if order >= _Order.HESSIAN:
            hess = torch.zeros(
                *points.shape[:-1], 2, 2, dtype=points.dtype, device=points.device
            )

        return FieldResult(
            _value=val,
            _verdict=Verdict.alive_like(x),
            _gradient=grad,
            _hessian=hess,
        )

    return sag_fn
