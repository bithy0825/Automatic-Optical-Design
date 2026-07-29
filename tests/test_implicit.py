r"""隐式曲面模块全面测试：断言 + 2D 可视化。

覆盖 ``implicit/`` 全部组件：sag → lift → guess → solver → verdict。

坐标系约定（当前版本）
------------------------
* 光轴：**z**
* sag 域：**(x, y)** 平面（垂直于光轴）
* lift：``f(x, y, z) = s(x, y) - z``
* 光线方向：默认沿 +z 传播

与废弃版（E:\workspace\aod\tests\）的关键 API 差异
---------------------------------------------------
* 光轴从 x 变为 z；sag 域从 (y,z) 变为 (x,y)
* ``verdict.hold``（单数）替代 ``verdict.holds``
* ``Verdict.alive_like(ref)`` 替代 ``Verdict.default(ref)``
* ``Verdict.site(hold=, toll=, cause=)`` 替代 ``Verdict(holds=, toll=)``
* ``aspheric_sag(c, k, alpha, norm)`` 不再接受 ``mask`` 参数
* ``lift_raw(sag)`` 简化为 ``f(x,y,z) = s(x,y) - z``，无 nf/tau
* ``Verdict.at()`` 对 toll 取 abs，新增 cause 追踪

用法::

    python tests/test_implicit.py
    python tests/test_implicit.py --no-show   # 仅保存 PNG，不阻塞
    pytest tests/test_implicit.py -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# 把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# 全程使用 float64：非球面系数的数量级与 sag 物理值同级，float32 精度不足
torch.set_default_dtype(torch.float64)

# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

_EPS_FD = 1e-4
_TOL = 1e-4

# ═══════════════════════════════════════════════════════════════════════════
# 调色板 — 色盲友好、高对比度
# ═══════════════════════════════════════════════════════════════════════════

C_NORMAL   = "#1976D2"
C_CONCAVE  = "#7B1FA2"
C_BOUNDARY = "#D32F2F"
C_OUT      = "#757575"
C_ALIVE    = "#2E7D32"
C_DEAD     = "#EF9A9A"
C_TOLL     = "#E65100"
C_GRAD     = "#C62828"
C_HESS     = "#6A1B9A"
C_SURFACE  = "#212121"
C_LIFT     = "#00838F"

_FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    flag = "ok  " if cond else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _FAILED.append(name)


# ═══════════════════════════════════════════════════════════════════════════
# 2D 张量构造辅助
#   sag 域为 (x, y) 平面（垂直于光轴 z），径向扫描取 y=0、x=r
# ═══════════════════════════════════════════════════════════════════════════

def radial_points_2d(r: torch.Tensor) -> torch.Tensor:
    """AOD 2D 点 ``(B=1,F=1,W=1,N,2)`` — sag 域 (x=r, y=0)。"""
    r = r.reshape(1, 1, 1, -1)
    y = torch.zeros_like(r)
    return torch.stack([r, y], dim=-1)


def grid_points_2d(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """构建 (x, y) 网格，返回 ``(X, Y, points)``。"""
    X, Y = torch.meshgrid(x, y, indexing="xy")
    flat_x = X.reshape(1, 1, 1, -1)
    flat_y = Y.reshape(1, 1, 1, -1)
    points = torch.stack([flat_x, flat_y], dim=-1)
    return X, Y, points


# ═══════════════════════════════════════════════════════════════════════════
# 评估辅助（无梯度，numpy 输出）
# ═══════════════════════════════════════════════════════════════════════════

def eval_sag_val(sag_fn, points: torch.Tensor, order: int = 0) -> np.ndarray:
    with torch.no_grad():
        return sag_fn(points, order=order).value.detach().cpu().numpy().reshape(-1)


def eval_sag_grad(sag_fn, points: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        g = sag_fn(points, order=1).gradient
    assert g is not None
    return g.detach().cpu().numpy().reshape(-1, 2)


def eval_sag_hess(sag_fn, points: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        h = sag_fn(points, order=2).hessian
    assert h is not None
    return h.detach().cpu().numpy().reshape(-1, 2, 2)


def eval_sag_hold(sag_fn, points: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return sag_fn(points, order=0).verdict.hold.detach().cpu().numpy().reshape(-1)


def eval_sag_toll(sag_fn, points: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return sag_fn(points, order=0).verdict.toll.detach().cpu().numpy().reshape(-1)


def eval_impl_val(impl_fn, points_3d: torch.Tensor, order: int = 0) -> np.ndarray:
    with torch.no_grad():
        return impl_fn(points_3d, order=order).value.detach().cpu().numpy().reshape(-1)


def eval_impl_grad(impl_fn, points_3d: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        g = impl_fn(points_3d, order=1).gradient
    assert g is not None
    return g.detach().cpu().numpy().reshape(-1, 3)


# ═══════════════════════════════════════════════════════════════════════════
# 绘图辅助
# ═══════════════════════════════════════════════════════════════════════════

def _shade_oob_mask(ax, r_np: np.ndarray, oob: np.ndarray) -> None:
    in_block = False
    start = None
    for i in range(len(oob)):
        if oob[i] and not in_block:
            in_block = True
            start = r_np[i]
        elif not oob[i] and in_block:
            in_block = False
            ax.axvspan(start, r_np[i - 1], color=C_DEAD, alpha=0.18, zorder=0)
    if in_block:
        ax.axvspan(start, float(r_np[-1]), color=C_DEAD, alpha=0.18, zorder=0)


def _shade_oob_by_toll(ax, r_np: np.ndarray, toll: np.ndarray) -> None:
    oob = toll < 0.0
    if oob.any():
        _shade_oob_mask(ax, r_np, oob)


# ═══════════════════════════════════════════════════════════════════════════
# 延迟导入
# ═══════════════════════════════════════════════════════════════════════════

def _import_implicit():
    from core import Verdict, sturdy_div
    from implicit.guess import guess
    from implicit.lift import lift_raw
    from implicit.protocol import (
        FieldResult,
        HalleySolverOptions,
        NewtonSolverOptions,
    )
    from implicit.sag import aspheric_sag, conical_sag, flat_sag, spherical_sag
    from implicit.solver import halley_step, newton_step, solve
    return {
        "Verdict": Verdict, "sturdy_div": sturdy_div,
        "guess": guess, "lift_raw": lift_raw,
        "DerivativeOrder": FieldResult.Order,
        "HalleySolverOptions": HalleySolverOptions,
        "InitMethod": NewtonSolverOptions.Init,
        "NewtonSolverOptions": NewtonSolverOptions,
        "aspheric_sag": aspheric_sag, "conical_sag": conical_sag,
        "flat_sag": flat_sag, "spherical_sag": spherical_sag,
        "halley_step": halley_step, "newton_step": newton_step,
        "solve": solve,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 第一部分：断言测试
# ═══════════════════════════════════════════════════════════════════════════

# ── sag: 球面值 + 梯度有限差分 ──

def test_sag_value_and_gradient(m) -> None:
    print("sag: 球面值 + 梯度（有限差分）")

    c = torch.tensor([0.05])
    sag = m["conical_sag"](c, torch.tensor([0.0]))

    r2 = 0.3 ** 2 + 0.2 ** 2
    radicand = 1.0 - (0.05 ** 2) * r2
    expected = 0.05 * r2 / (1.0 + math.sqrt(radicand))

    r = sag(torch.tensor([[[[[0.3, 0.2]]]]]), order=m["DerivativeOrder"].GRADIENT)
    check("spherical value", abs(r.value.item() - expected) < 1e-6,
          f"got {r.value.item():.6f} want {expected:.6f}")

    gx, gy = r.gradient[0, 0, 0, 0].tolist()

    def val(xv, yv):
        s = m["conical_sag"](torch.tensor([0.05]), torch.tensor([0.0]))
        return s(torch.tensor([[[[[xv, yv]]]]]), order=m["DerivativeOrder"].VALUE).value.item()

    fd_gx = (val(0.3 + _EPS_FD, 0.2) - val(0.3 - _EPS_FD, 0.2)) / (2 * _EPS_FD)
    fd_gy = (val(0.3, 0.2 + _EPS_FD) - val(0.3, 0.2 - _EPS_FD)) / (2 * _EPS_FD)
    check("spherical grad_x", abs(gx - fd_gx) < 1e-4,
          f"analytic {gx:.6f} fd {fd_gx:.6f}")
    check("spherical grad_y", abs(gy - fd_gy) < 1e-4,
          f"analytic {gy:.6f} fd {fd_gy:.6f}")


# ── sag: Hessian 有限差分 ──

def test_sag_hessian_finite_diff(m) -> None:
    print("sag: Hessian（中心有限差分）")

    c, k = torch.tensor([0.1]), torch.tensor([-0.5])
    sag = m["conical_sag"](c, k)
    h = sag(torch.tensor([[[[[0.25, 0.15]]]]]),
            order=m["DerivativeOrder"].HESSIAN).hessian[0, 0, 0, 0]

    def val(xv, yv):
        s = m["conical_sag"](torch.tensor([0.1]), torch.tensor([-0.5]))
        return s(torch.tensor([[[[[xv, yv]]]]]), order=m["DerivativeOrder"].VALUE).value.item()

    e = 1e-2
    fd_xx = (val(0.25 + e, 0.15) - 2 * val(0.25, 0.15) + val(0.25 - e, 0.15)) / (e ** 2)
    fd_xy = (val(0.25 + e, 0.15 + e) - val(0.25 + e, 0.15 - e)
             - val(0.25 - e, 0.15 + e) + val(0.25 - e, 0.15 - e)) / (4 * e ** 2)

    check("hessian xx", abs(h[0, 0].item() - fd_xx) < 1e-3,
          f"analytic {h[0, 0].item():.6f} fd {fd_xx:.6f}")
    check("hessian xy", abs(h[0, 1].item() - fd_xy) < 1e-3,
          f"analytic {h[0, 1].item():.6f} fd {fd_xy:.6f}")


# ── sag: 非球面梯度（新 API 无 mask）──

def test_aspheric_gradient(m) -> None:
    print("sag: 非球面梯度（有限差分，新 API 无 mask）")

    alpha = torch.tensor([[0.3, 0.05]])  # scaled: α_n = A_true * ρ^(4+2n)
    norm = torch.tensor([2.0])
    sag = m["aspheric_sag"](torch.tensor([0.05]), torch.tensor([-0.5]), alpha, norm)
    r = sag(torch.tensor([[[[[0.2, 0.15]]]]]), order=m["DerivativeOrder"].HESSIAN)

    def val(xv, yv):
        s = m["aspheric_sag"](torch.tensor([0.05]), torch.tensor([-0.5]), alpha, norm)
        return s(torch.tensor([[[[[xv, yv]]]]]), order=m["DerivativeOrder"].VALUE).value.item()

    g = r.gradient[0, 0, 0, 0].tolist()
    fd_gx = (val(0.2 + _EPS_FD, 0.15) - val(0.2 - _EPS_FD, 0.15)) / (2 * _EPS_FD)
    fd_gy = (val(0.2, 0.15 + _EPS_FD) - val(0.2, 0.15 - _EPS_FD)) / (2 * _EPS_FD)
    check("aspheric grad_x", abs(g[0] - fd_gx) < 1e-4,
          f"analytic {g[0]:.6f} fd {fd_gx:.6f}")
    check("aspheric grad_y", abs(g[1] - fd_gy) < 1e-4,
          f"analytic {g[1]:.6f} fd {fd_gy:.6f}")


# ── sag: 平面 ──

def test_flat_sag(m) -> None:
    print("sag: 平面（恒零，永在域内）")

    pts = torch.tensor([[[[[0.5, 0.5]]]]])
    r = m["flat_sag"]()(pts, order=m["DerivativeOrder"].HESSIAN)
    check("flat value == 0", abs(r.value.item()) < 1e-9)
    check("flat hold 全 True", bool(r.verdict.hold.all()))
    check("flat toll 为 0（存活无代价）", r.verdict.toll.item() == 0.0)
    check("flat cause 为 NONE", (r.verdict.cause == 0).all().item())


# ── sag: OOB 判死 ──

def test_sag_oob_kills(m) -> None:
    print("sag: 越界 → hold=False, toll=radicand<0")

    sag = m["conical_sag"](torch.tensor([1.0]), torch.tensor([0.0]))
    # (x=2, y=0) → r²=4 → radicand=1-4=-3
    r = sag(torch.tensor([[[[[2.0, 0.0]]]]]), order=m["DerivativeOrder"].HESSIAN)
    check("oob hold=False", not r.verdict.hold.item())
    check("oob toll == radicand (-3)", abs(r.verdict.toll.item() + 3.0) < 1e-6)
    check("oob cause=SAG_DOMAIN",
          (r.verdict.cause == m["Verdict"].Cause.SAG_DOMAIN).item())
    check("oob value=0", abs(r.value.item()) < 1e-9)
    check("oob grad=0", bool(torch.all(r.gradient == 0)))


# ── lift: f = s(x,y) - z ──

def test_lift_implicit(m) -> None:
    print("lift: f(x,y,z) = s(x,y) - z, optical axis=z")

    c = torch.tensor([0.05])
    impl = m["lift_raw"](m["spherical_sag"](c))
    xv, yv = 0.3, 0.2
    s_val = m["spherical_sag"](c)(
        torch.tensor([[[[[xv, yv]]]]]), order=m["DerivativeOrder"].VALUE
    ).value.item()
    # 面上点：(x, y, z=s_val)
    p = torch.tensor([[[[[xv, yv, s_val]]]]])
    r = impl(p, order=m["DerivativeOrder"].GRADIENT)
    check("lift f==0 on surface", abs(r.value.item()) < 1e-6,
          f"got {r.value.item():.3e}")
    # df/dz == -1（光轴为 z）
    check("lift df/dz == -1",
          abs(r.gradient[0, 0, 0, 0].tolist()[2] + 1.0) < 1e-6)


# ── solver 辅助 ──

def _sphere_setup(m) -> tuple:
    """球面 c=0.05，光线从 z=-2 沿 +z 出发，交点在 (x=0.3, y=0.2)。"""
    c = torch.tensor([0.05])
    impl = m["lift_raw"](m["spherical_sag"](c))
    # 光线起点 (x, y, z) = (0.3, 0.2, -2.0)，方向沿 +z
    P = torch.tensor([[[[[0.3, 0.2, -2.0]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    r2 = 0.3 ** 2 + 0.2 ** 2
    radicand = 1.0 - (0.05 ** 2) * r2
    expected_z = 0.05 * r2 / (1.0 + math.sqrt(radicand))
    return impl, P, V, 2.0 + expected_z


# ── solver: Newton & Halley 收敛 ──

def test_solver_convergence(m) -> None:
    print("solver: Newton & Halley 收敛到解析球面交点")

    impl, P, V, expected_d = _sphere_setup(m)
    for name, opt in [("newton", m["NewtonSolverOptions"](num_iter=10)),
                      ("halley", m["HalleySolverOptions"](num_iter=8))]:
        res = m["solve"](opt)(points=P, directions=V, implicit=impl)
        check(f"{name} distance", abs(res.distances.item() - expected_d) < 1e-4,
              f"got {res.distances.item():.6f} want {expected_d:.6f}")
        check(f"{name} hold=True", bool(res.verdict.hold.item()))
        check(f"{name} |value|<tol", abs(res.value.item()) < _TOL,
              f"|value|={abs(res.value.item()):.3e}")


# ── solver: 返回值一致性 ──

def test_solver_value_at_returned_distance(m) -> None:
    print("solver: f(distance) == value（同点一致性）")

    impl, P, V, _ = _sphere_setup(m)
    res = m["solve"](m["HalleySolverOptions"](num_iter=8))(
        points=P, directions=V, implicit=impl)
    hit = P.add(V.mul(res.distances.unsqueeze(-1)))
    f_at_d = impl(hit, order=m["DerivativeOrder"].VALUE).value
    check("value at returned distance",
          torch.allclose(f_at_d, res.value, atol=1e-6))


# ── solver: OOB 光线判死 ──

def test_solver_oob_death(m) -> None:
    print("solver: OOB 光线 → hold=False, toll<0")

    c = torch.tensor([1.0])
    impl = m["lift_raw"](m["spherical_sag"](c))
    # (x=2, y=0, z=-5) → r=2 超出 c=1 的定义域
    P = torch.tensor([[[[[2.0, 0.0, -5.0]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    res = m["solve"](m["HalleySolverOptions"](num_iter=12))(
        points=P, directions=V, implicit=impl)
    check("oob hold=False", not res.verdict.hold.item())
    # Verdict.at() abs() toll → always >= 0; sign lives in cause
    check("oob toll>0 (abs)", res.verdict.toll.item() > 0.0,
          f"toll={res.verdict.toll.item():.4f}")


# ── solver: 不收敛判死 ──

def test_solver_nonconvergence_death(m) -> None:
    print("solver: 迭代不足 → 收敛裁决判死")

    impl, P, V, _ = _sphere_setup(m)
    res = m["solve"](m["HalleySolverOptions"](num_iter=1))(
        points=P, directions=V, implicit=impl)
    check("nonconv hold=False", not res.verdict.hold.item())
    # Verdict.at() abs() → residual still > 0 but toll >= 0
    check("nonconv toll>0 (abs of residual)",
          res.verdict.toll.item() > 0.0,
          f"toll={res.verdict.toll.item():.4f}")


# ── solver: 负距离（中间面）──

def test_solver_negative_distance_middle_face(m) -> None:
    print("solver: 中间面负距离 → 判死, toll=|d|, 梯度链完整")

    c = torch.tensor([0.5], requires_grad=True)
    impl = m["lift_raw"](m["spherical_sag"](c))
    # 光线起点在曲面后方的 z>0 处，沿 +z 传播 → 负根
    P = torch.tensor([[[[[0.4, 0.0, 0.5]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])

    res = m["solve"](m["HalleySolverOptions"](num_iter=8))(
        points=P, directions=V, implicit=impl)
    check("middle-face neg distance", res.distances.item() < 0.0,
          f"distance={res.distances.item():.4f}")
    check("middle-face hold=False", not res.verdict.hold.item())
    check("toll == |distance|",
          abs(res.verdict.toll.item() - abs(res.distances.item())) < 1e-5)

    g = torch.autograd.grad(res.verdict.toll, c)[0]
    check("d(toll)/dc nonzero", abs(g.item()) > 1e-8)


# ── solver: 负距离（首面）──

def test_solver_negative_distance_first_face(m) -> None:
    print("solver: 首面负距离（allow_negative=True）→ 合法，存活")

    c = torch.tensor([0.5])
    impl = m["lift_raw"](m["spherical_sag"](c))
    P = torch.tensor([[[[[0.4, 0.0, 0.5]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])

    res = m["solve"](m["HalleySolverOptions"](num_iter=8, allow_negative=True))(
        points=P, directions=V, implicit=impl)
    check("first-face neg distance", res.distances.item() < 0.0)
    check("first-face hold=True", bool(res.verdict.hold.item()))
    check("first-face toll>=0", res.verdict.toll.item() >= 0.0)


# ── solver: 可微性 ──

def test_solver_differentiability(m) -> None:
    print("solver: distance/value/toll 对曲率 c 可微")

    c = torch.tensor([0.05], requires_grad=True)
    impl = m["lift_raw"](m["spherical_sag"](c))
    P = torch.tensor([[[[[0.3, 0.2, -2.0]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    res = m["solve"](m["HalleySolverOptions"](num_iter=8))(
        points=P, directions=V, implicit=impl)

    g_dist = torch.autograd.grad(res.distances, c, retain_graph=True)[0]
    g_val = torch.autograd.grad(res.value, c, retain_graph=True)[0]
    g_toll = torch.autograd.grad(res.verdict.toll, c)[0]
    check("d distance/dc nonzero", abs(g_dist.item()) > 1e-8)
    check("d value/dc nonzero", abs(g_val.item()) > 1e-8)
    check("d toll/dc nonzero", abs(g_toll.item()) > 1e-8)
    check("gradients finite",
          bool(torch.isfinite(g_dist) and torch.isfinite(g_val) and torch.isfinite(g_toll)))


# ── Verdict 合约 ──

def test_verdict_at_freezes_on_death(m) -> None:
    print("verdict.at: 首次死亡冻结 toll/cause，不复活")

    alive = m["Verdict"].alive_like(torch.zeros(1))
    site_a = m["Verdict"].site(
        hold=torch.tensor([False]), toll=torch.tensor([-3.0]),
        cause=m["Verdict"].Cause.SAG_DOMAIN)
    site_b = m["Verdict"].site(
        hold=torch.tensor([True]), toll=torch.tensor([99.0]),
        cause=m["Verdict"].Cause.NONE)

    d1 = alive.at(site_a)
    check("first death toll pinned", abs(d1.toll.item() - 3.0) < 1e-9)  # at() abs
    check("first death hold=False", not d1.hold.item())

    d2 = d1.at(site_b)
    check("no resurrection", not d2.hold.item())
    check("toll frozen", abs(d2.toll.item() - 3.0) < 1e-9)


def test_verdict_at_survives_passing_sites(m) -> None:
    print("verdict.at: 存活光线经过存活 site，toll 保持 0")

    alive = m["Verdict"].alive_like(torch.zeros(1))
    passing = m["Verdict"].site(
        hold=torch.tensor([True]), toll=torch.tensor([5.0]),
        cause=m["Verdict"].Cause.NONE)
    d = alive.at(passing)
    check("surviving hold=True", bool(d.hold.item()))
    check("surviving toll=0", abs(d.toll.item()) < 1e-9)


def test_sturdy_div_safe(m) -> None:
    print("sturdy_div: 零/无穷除数 → 0，无 NaN")

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([2.0, 0.0, float("inf"), -0.0])
    out = m["sturdy_div"](a, b)
    check("finite everywhere", bool(torch.isfinite(out).all()))
    check("zero on bad divisor", float(out[1]) == 0.0 and float(out[2]) == 0.0)
    check("correct on good divisor", abs(float(out[0]) - 0.5) < 1e-9)


# ── guess ──

def test_guess_closest(m) -> None:
    print("guess: init_closest 投影到光线距原点最近点")

    impl = m["lift_raw"](m["spherical_sag"](torch.tensor([0.05])))
    P = torch.tensor([[[[[0.3, 0.2, -2.0]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    d = m["guess"](P, V, implicit=impl, init_method=m["InitMethod"].CLOSEST)
    # t* = -P·V / |V|² = 2.0 / 1.0
    check("closest distance == 2.0", abs(d.item() - 2.0) < 1e-6)


_ASSERTION_TESTS = [
    test_sag_value_and_gradient,
    test_sag_hessian_finite_diff,
    test_aspheric_gradient,
    test_flat_sag,
    test_sag_oob_kills,
    test_lift_implicit,
    test_solver_convergence,
    test_solver_value_at_returned_distance,
    test_solver_oob_death,
    test_solver_nonconvergence_death,
    test_solver_negative_distance_middle_face,
    test_solver_negative_distance_first_face,
    test_solver_differentiability,
    test_verdict_at_freezes_on_death,
    test_verdict_at_survives_passing_sites,
    test_sturdy_div_safe,
    test_guess_closest,
]


# ═══════════════════════════════════════════════════════════════════════════
# 第二部分：2D 可视化
# ═══════════════════════════════════════════════════════════════════════════

# ── 图 1：矢高值径向截面 ──

def plot_sag_value_sections(m) -> None:
    """sag 在 (x,y) 平面沿径向 r=√(x²+y²) 的截面（y=0 即 x=r）。"""
    r_long = torch.linspace(0.0, 3.5, 400)
    r_short = torch.linspace(0.0, 1.5, 200)
    pts_long = radial_points_2d(r_long)
    pts_short = radial_points_2d(r_short)

    curvature = torch.tensor([0.5])
    c_val = curvature.item()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()
    r_np = r_long.numpy()
    r_s_np = r_short.numpy()

    # 1. spherical_sag
    sph = m["spherical_sag"](curvature)
    sag_sph = eval_sag_val(sph, pts_long)
    toll_sph = eval_sag_toll(sph, pts_long)
    r_b_sph = 1.0 / c_val
    axes[0].plot(r_np, sag_sph, color=C_NORMAL, lw=2.0, label="spherical sag")
    axes[0].axvline(r_b_sph, color=C_BOUNDARY, lw=1.5, ls="--",
                    label=f"r_b={r_b_sph:.2f}")
    _shade_oob_by_toll(axes[0], r_np, toll_sph)
    axes[0].set_xlabel("r = sqrt(x²+y²)")
    axes[0].set_ylabel("sag z")
    axes[0].set_title("spherical_sag (c=0.5)", fontsize=9, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=6.5)
    print(f"[矢高 球面] c={c_val:.2f}, r_b={r_b_sph:.2f}")

    # 2. conical 四种
    for name, kv in [("sphere κ=0", 0.0), ("ellipsoid κ=-0.5", -0.5),
                     ("paraboloid κ=-1", -1.0), ("hyperboloid κ=-2", -2.0)]:
        conic = m["conical_sag"](curvature, torch.tensor([kv]))
        sag_c = eval_sag_val(conic, pts_short)
        axes[1].plot(r_s_np, sag_c, lw=2.0, label=name)
    axes[1].set_xlabel("r"); axes[1].set_ylabel("sag z")
    axes[1].set_title("conical_sag — normal variants", fontsize=9, fontweight="bold")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)
    print("[矢高 圆锥] κ=0, -0.5, -1, -2")

    # 3. ellipsoid 临界/越界
    k_ell = -0.5
    conic_ell = m["conical_sag"](curvature, torch.tensor([k_ell]))
    sag_ell = eval_sag_val(conic_ell, pts_long)
    toll_ell = eval_sag_toll(conic_ell, pts_long)
    r_b_ell = 1.0 / (c_val * np.sqrt(1.0 + k_ell))
    axes[2].plot(r_np, sag_ell, color=C_NORMAL, lw=2.0)
    axes[2].axvline(r_b_ell, color=C_BOUNDARY, lw=1.5, ls="--",
                    label=f"r_b={r_b_ell:.2f}")
    _shade_oob_by_toll(axes[2], r_np, toll_ell)
    axes[2].set_xlabel("r"); axes[2].set_ylabel("sag z")
    axes[2].set_title(f"conical ellipsoid κ={k_ell} critical/OOB",
                      fontsize=9, fontweight="bold")
    axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=6.5)
    print(f"[矢高 椭球临界] r_b={r_b_ell:.2f}")

    # 4. aspheric normal
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    alpha_n = torch.tensor([[0.5, 0.1]])
    asp_n = m["aspheric_sag"](curvature, k0, alpha_n, norm1)
    sag_asp_n = eval_sag_val(asp_n, pts_long)
    toll_asp_n = eval_sag_toll(asp_n, pts_long)
    axes[3].plot(r_np, sag_asp_n, color=C_NORMAL, lw=2.0)
    axes[3].axvline(r_b_sph, color=C_BOUNDARY, lw=1.5, ls="--")
    _shade_oob_by_toll(axes[3], r_np, toll_asp_n)
    axes[3].set_xlabel("r"); axes[3].set_ylabel("sag z")
    axes[3].set_title("aspheric_sag normal (small α)", fontsize=9, fontweight="bold")
    axes[3].grid(True, alpha=0.3)
    print("[矢高 非球面正常] 球面基底 + 小偶次修正")

    # 5. aspheric critical
    alpha_c = torch.tensor([[2.0, -0.5]])
    asp_c = m["aspheric_sag"](curvature, k0, alpha_c, norm1)
    sag_asp_c = eval_sag_val(asp_c, pts_long)
    toll_asp_c = eval_sag_toll(asp_c, pts_long)
    axes[4].plot(r_np, sag_asp_c, color=C_CONCAVE, lw=2.0)
    axes[4].axvline(r_b_sph, color=C_BOUNDARY, lw=1.5, ls="--")
    _shade_oob_by_toll(axes[4], r_np, toll_asp_c)
    axes[4].set_xlabel("r"); axes[4].set_ylabel("sag z")
    axes[4].set_title("aspheric_sag critical (large α)", fontsize=9, fontweight="bold")
    axes[4].grid(True, alpha=0.3)
    print("[矢高 非球面临界] 大系数使 sag 显著变形")

    # 6. spherical concave
    c_neg = torch.tensor([-0.5])
    sph_neg = m["spherical_sag"](c_neg)
    sag_neg = eval_sag_val(sph_neg, pts_long)
    toll_neg = eval_sag_toll(sph_neg, pts_long)
    r_b_neg = 1.0 / abs(c_neg.item())
    axes[5].plot(r_np, sag_neg, color=C_CONCAVE, lw=2.0)
    axes[5].axvline(r_b_neg, color=C_BOUNDARY, lw=1.5, ls="--")
    _shade_oob_by_toll(axes[5], r_np, toll_neg)
    axes[5].set_xlabel("r"); axes[5].set_ylabel("sag z")
    axes[5].set_title("spherical_sag concave (c=-0.5)", fontsize=9, fontweight="bold")
    axes[5].grid(True, alpha=0.3)
    print(f"[矢高 凹球面] c={c_neg.item():.2f}, r_b={r_b_neg:.2f}")

    # 7. flat
    sag_flat = eval_sag_val(m["flat_sag"](), pts_long)
    axes[6].plot(r_np, sag_flat, color="#424242", lw=2.0)
    axes[6].set_xlabel("r"); axes[6].set_ylabel("sag z")
    axes[6].set_title("flat_sag — zero everywhere", fontsize=9, fontweight="bold")
    axes[6].grid(True, alpha=0.3)
    print("[矢高 平面] 恒为零")

    axes[7].axis("off")

    fig.suptitle("Sag Functions — Radial Cross-Sections (z vs r)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_value_sections.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 2：矢高 (x,y) 热力图 ──

def plot_sag_value_heatmaps(m) -> None:
    """sag 值在 (x,y) 平面上的 2D 热力图。"""
    x = torch.linspace(-3.0, 3.0, 200)
    y = torch.linspace(-3.0, 3.0, 200)
    X, Y, pts_grid = grid_points_2d(x, y)

    curvature = torch.tensor([0.5])
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    cases = [
        ("spherical_sag (c=0.5)", m["spherical_sag"](curvature)),
        ("spherical concave (c=-0.5)", m["spherical_sag"](torch.tensor([-0.5]))),
        ("conical ellipsoid κ=-0.5", m["conical_sag"](curvature, torch.tensor([-0.5]))),
        ("conical paraboloid κ=-1", m["conical_sag"](curvature, torch.tensor([-1.0]))),
        ("conical hyperboloid κ=-2", m["conical_sag"](curvature, torch.tensor([-2.0]))),
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1)),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1)),
        ("flat_sag", m["flat_sag"]()),
    ]

    for ax, (title, sag_fn) in zip(axes, cases):
        sag = eval_sag_val(sag_fn, pts_grid).reshape(X.shape)
        cnt = ax.contourf(X.numpy(), Y.numpy(), sag, levels=20, cmap="viridis")
        plt.colorbar(cnt, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Sag Functions — (x, y) Value Heatmaps",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_value_heatmaps.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 3：sturdy 算子效应 ──

def plot_sturdy_effects(m) -> None:
    r_long = torch.linspace(0.0, 3.5, 400)
    pts_long = radial_points_2d(r_long)
    curvature = torch.tensor([0.5])
    c_val = curvature.item()
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    def _plot_sturdy(ax, r, c, k, sag_np, title, r_boundary=None):
        r_np = r.numpy()
        radicand = 1.0 - (1.0 + k) * c * c * r_np * r_np
        sqrt_safe = np.where(radicand >= 0.0,
                             np.sqrt(np.clip(radicand, 0.0, None)), 0.0)
        numerator = c * r_np * r_np

        ax2 = ax.twinx()
        l_num = ax.plot(r_np, numerator, "--", color="#90CAF9", lw=1.5,
                        label="numerator c·r²")[0]
        l_sag = ax.plot(r_np, sag_np, "-", color=C_NORMAL, lw=2.5,
                        label="sturdy sag")[0]
        l_rad = ax2.plot(r_np, radicand, ":", color="#757575", lw=1.5,
                         label="radicand")[0]
        l_sqrt = ax2.plot(r_np, sqrt_safe, "-", color="#F57C00", lw=1.5,
                          label="sturdy_sqrt")[0]
        lines = [l_num, l_sag, l_rad, l_sqrt]

        if r_boundary is not None and r_boundary > 0:
            ax.axvline(r_boundary, color=C_BOUNDARY, lw=1.5, ls="--")
            ax.axvspan(r_boundary, float(r_np.max()), color=C_OUT, alpha=0.12)

        ax.set_xlabel("r"); ax.set_ylabel("sag / numerator", color=C_NORMAL)
        ax2.set_ylabel("radicand / sqrt", color="#F57C00")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6,
                  loc="upper left")

    r_b_sph = 1.0 / c_val

    # spherical
    sag_sph = eval_sag_val(m["spherical_sag"](curvature), pts_long)
    _plot_sturdy(axes[0], r_long, c_val, 0.0, sag_sph,
                 "sturdy: spherical_sag", r_boundary=r_b_sph)

    # ellipsoid
    k_ell = -0.5
    sag_ell = eval_sag_val(m["conical_sag"](curvature, torch.tensor([k_ell])), pts_long)
    r_b_ell = 1.0 / (c_val * np.sqrt(1.0 + k_ell))
    _plot_sturdy(axes[1], r_long, c_val, k_ell, sag_ell,
                 "sturdy: conical ellipsoid", r_boundary=r_b_ell)

    # paraboloid
    sag_par = eval_sag_val(m["conical_sag"](curvature, torch.tensor([-1.0])), pts_long)
    _plot_sturdy(axes[2], r_long, c_val, -1.0, sag_par,
                 "sturdy: conical paraboloid")

    # hyperboloid
    sag_hyp = eval_sag_val(m["conical_sag"](curvature, torch.tensor([-2.0])), pts_long)
    _plot_sturdy(axes[3], r_long, c_val, -2.0, sag_hyp,
                 "sturdy: conical hyperboloid")

    # aspheric normal
    alpha_n = torch.tensor([[0.5, 0.1]])
    sag_asp_n = eval_sag_val(
        m["aspheric_sag"](curvature, k0, alpha_n, norm1), pts_long)
    _plot_sturdy(axes[4], r_long, c_val, 0.0, sag_asp_n,
                 "sturdy: aspheric normal", r_boundary=r_b_sph)

    # aspheric critical
    alpha_c = torch.tensor([[2.0, -0.5]])
    sag_asp_c = eval_sag_val(
        m["aspheric_sag"](curvature, k0, alpha_c, norm1), pts_long)
    _plot_sturdy(axes[5], r_long, c_val, 0.0, sag_asp_c,
                 "sturdy: aspheric critical", r_boundary=r_b_sph)

    # concave spherical
    c_neg = -0.5
    sag_neg = eval_sag_val(m["spherical_sag"](torch.tensor([c_neg])), pts_long)
    _plot_sturdy(axes[6], r_long, c_neg, 0.0, sag_neg,
                 "sturdy: spherical concave",
                 r_boundary=1.0 / abs(c_neg))

    # flat
    sag_f = eval_sag_val(m["flat_sag"](), pts_long)
    axes[7].plot(r_long.numpy(), sag_f, color="#424242", lw=2.0)
    axes[7].set_xlabel("r"); axes[7].set_ylabel("sag z")
    axes[7].set_title("sturdy: flat_sag", fontsize=9, fontweight="bold")
    axes[7].grid(True, alpha=0.3)

    fig.suptitle("Sturdy-Operator Effect on Sag",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_sturdy.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 4：梯度径向截面 ──

def plot_sag_gradient_sections(m) -> None:
    """dsag/dx 沿 y=0 径向的截面。"""
    r_long = torch.linspace(0.0, 3.5, 400)
    r_short = torch.linspace(0.0, 1.5, 200)
    pts_long = radial_points_2d(r_long)
    pts_short = radial_points_2d(r_short)

    curvature = torch.tensor([0.5])
    c_val = curvature.item()
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    r_np = r_long.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    # 1. spherical gradient
    grad_sph = eval_sag_grad(m["spherical_sag"](curvature), pts_long)[:, 0]
    toll_sph = eval_sag_toll(m["spherical_sag"](curvature), pts_long)
    axes[0].plot(r_np, np.clip(grad_sph, -20, 20), color=C_GRAD, lw=2.0)
    axes[0].axvline(1.0 / c_val, color=C_BOUNDARY, lw=1.5, ls="--")
    _shade_oob_by_toll(axes[0], r_np, toll_sph)
    axes[0].set_xlabel("r"); axes[0].set_ylabel("dsag/dx")
    axes[0].set_title("gradient: spherical_sag", fontsize=9, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # 2. conical variants
    r_s_np = r_short.numpy()
    for name, kv in [("sphere κ=0", 0.0), ("ellipsoid κ=-0.5", -0.5),
                     ("paraboloid κ=-1", -1.0), ("hyperboloid κ=-2", -2.0)]:
        g = eval_sag_grad(m["conical_sag"](curvature, torch.tensor([kv])),
                          pts_short)[:, 0]
        axes[1].plot(r_s_np, np.clip(g, -10, 10), lw=2.0, label=name)
    axes[1].set_xlabel("r"); axes[1].set_ylabel("dsag/dx")
    axes[1].set_title("gradient: conical_sag variants", fontsize=9, fontweight="bold")
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=7)

    # 3. ellipsoid critical
    k_ell = -0.5
    grad_ell = eval_sag_grad(
        m["conical_sag"](curvature, torch.tensor([k_ell])), pts_long)[:, 0]
    toll_ell = eval_sag_toll(
        m["conical_sag"](curvature, torch.tensor([k_ell])), pts_long)
    r_b_ell = 1.0 / (c_val * np.sqrt(1.0 + k_ell))
    axes[2].plot(r_np, np.clip(grad_ell, -20, 20), color=C_GRAD, lw=2.0)
    axes[2].axvline(r_b_ell, color=C_BOUNDARY, lw=1.5, ls="--")
    _shade_oob_by_toll(axes[2], r_np, toll_ell)
    axes[2].set_xlabel("r"); axes[2].set_ylabel("dsag/dx")
    axes[2].set_title("gradient: conical ellipsoid critical",
                      fontsize=9, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # 4-7: aspheric normal, aspheric critical, concave, flat
    extra_cases = [
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1)),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1)),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5]))),
        ("flat_sag", m["flat_sag"]()),
    ]
    for idx, (title, sag_fn) in enumerate(extra_cases):
        ax = axes[4 + idx]
        grad = eval_sag_grad(sag_fn, pts_long)[:, 0]
        toll = eval_sag_toll(sag_fn, pts_long)
        ax.plot(r_np, np.clip(grad, -20, 20), color=C_GRAD, lw=2.0)
        _shade_oob_by_toll(ax, r_np, toll)
        ax.set_xlabel("r"); ax.set_ylabel("dsag/dx")
        ax.set_title(f"gradient: {title}", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[3].axis("off")

    fig.suptitle("Sag Gradients — Radial Cross-Sections (dsag/dx along y=0)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_gradient_sections.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 5：梯度幅度 (x,y) 热力图 ──

def plot_sag_gradient_heatmaps(m) -> None:
    x = torch.linspace(-3.0, 3.0, 200)
    y = torch.linspace(-3.0, 3.0, 200)
    X, Y, pts_grid = grid_points_2d(x, y)

    curvature = torch.tensor([0.5])
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    cases = [
        ("spherical_sag", m["spherical_sag"](curvature)),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5]))),
        ("conical ellipsoid κ=-0.5",
         m["conical_sag"](curvature, torch.tensor([-0.5]))),
        ("conical paraboloid κ=-1",
         m["conical_sag"](curvature, torch.tensor([-1.0]))),
        ("conical hyperboloid κ=-2",
         m["conical_sag"](curvature, torch.tensor([-2.0]))),
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1)),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1)),
        ("flat_sag", m["flat_sag"]()),
    ]

    for ax, (title, sag_fn) in zip(axes, cases):
        grad = eval_sag_grad(sag_fn, pts_grid)
        gmag = np.linalg.norm(grad, axis=-1).reshape(X.shape)
        cnt = ax.contourf(X.numpy(), Y.numpy(), gmag, levels=20, cmap="magma")
        plt.colorbar(cnt, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle("Sag Gradient Magnitudes — (x, y) Heatmaps",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_gradient_heatmaps.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 6：Hessian 对角分量截面 ──

def plot_sag_hessian_sections(m) -> None:
    r_long = torch.linspace(0.0, 3.5, 400)
    pts_long = radial_points_2d(r_long)

    curvature = torch.tensor([0.5])
    c_val = curvature.item()
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    r_np = r_long.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    cases = [
        ("spherical_sag", m["spherical_sag"](curvature), 1.0 / c_val),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5])), 2.0),
        ("conical ellipsoid κ=-0.5",
         m["conical_sag"](curvature, torch.tensor([-0.5])),
         1.0 / (c_val * np.sqrt(0.5))),
        ("conical paraboloid κ=-1",
         m["conical_sag"](curvature, torch.tensor([-1.0])), None),
        ("conical hyperboloid κ=-2",
         m["conical_sag"](curvature, torch.tensor([-2.0])), None),
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1),
         1.0 / c_val),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1),
         1.0 / c_val),
        ("flat_sag", m["flat_sag"](), None),
    ]

    for ax, (title, sag_fn, r_boundary) in zip(axes, cases):
        hess = eval_sag_hess(sag_fn, pts_long)
        toll = eval_sag_toll(sag_fn, pts_long)

        ax.plot(r_np, np.clip(hess[:, 0, 0], -10, 10), color=C_HESS, lw=2.0,
                label="∂²s/∂x²")
        ax.plot(r_np, np.clip(hess[:, 1, 1], -10, 10), color="#CE93D8", lw=1.5,
                ls="--", label="∂²s/∂y²")
        _shade_oob_by_toll(ax, r_np, toll)
        if r_boundary is not None:
            ax.axvline(r_boundary, color=C_BOUNDARY, lw=1.5, ls="--")

        ax.set_xlabel("r"); ax.set_ylabel("Hessian diag")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=6)

    fig.suptitle("Sag Hessians — Diagonal Components (clipped ±10)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_hessian_sections.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 7：Verdict toll 径向截面 ──

def plot_verdict_toll_sections(m) -> None:
    r_long = torch.linspace(0.0, 3.5, 400)
    pts_long = radial_points_2d(r_long)

    curvature = torch.tensor([0.5])
    c_val = curvature.item()
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    r_np = r_long.numpy()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    cases = [
        ("spherical_sag", m["spherical_sag"](curvature), 1.0 / c_val),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5])), 2.0),
        ("conical ellipsoid κ=-0.5",
         m["conical_sag"](curvature, torch.tensor([-0.5])),
         1.0 / (c_val * np.sqrt(0.5))),
        ("conical paraboloid κ=-1",
         m["conical_sag"](curvature, torch.tensor([-1.0])), None),
        ("conical hyperboloid κ=-2",
         m["conical_sag"](curvature, torch.tensor([-2.0])), None),
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1),
         1.0 / c_val),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1),
         1.0 / c_val),
        ("flat_sag", m["flat_sag"](), None),
    ]

    for ax, (name, sag_fn, r_boundary) in zip(axes, cases):
        sag = eval_sag_val(sag_fn, pts_long)
        toll = eval_sag_toll(sag_fn, pts_long)
        hold = eval_sag_hold(sag_fn, pts_long)

        ax.plot(r_np, sag, color=C_NORMAL, lw=2.0, label="sag")
        ax2 = ax.twinx()
        ax2.plot(r_np, np.clip(toll, -3, 3), color=C_TOLL, lw=1.8, ls="--",
                 label="toll")
        ax2.axhline(0, color=C_BOUNDARY, lw=1.0, ls=":", alpha=0.6)
        _shade_oob_by_toll(ax, r_np, toll)

        if r_boundary is not None:
            ax.axvline(r_boundary, color=C_BOUNDARY, lw=1.5, ls="--")

        n_alive = int(hold.sum())
        ax.annotate(
            f"alive: {n_alive}/{len(hold)}",
            xy=(0.98, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=7,
            color=C_ALIVE if n_alive == len(hold) else C_BOUNDARY,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

        ax.set_xlabel("r"); ax.set_ylabel("sag z", color=C_NORMAL)
        ax2.set_ylabel("toll", color=C_TOLL)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=6,
                  loc="upper right")

    fig.suptitle("Sag Verdict (toll) — Radial Cross-Sections",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_sag_toll_sections.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 8：Lift 2D 截面 — f(z, y) at x=0 ──

def plot_lift_cross_sections(m) -> None:
    """隐式函数 f(x,y,z) 在 x=0 处的 (z, y) 截面（光轴为 z）。"""
    curvature = torch.tensor([0.5])

    z_vals = torch.linspace(-1.0, 3.0, 200)
    y_vals = torch.linspace(-2.5, 2.5, 200)
    Z, Y = torch.meshgrid(z_vals, y_vals, indexing="xy")
    # 3D 点: (x=0, y, z) → (B=1,F=1,W=1,N,3)
    flat_x = torch.zeros(Z.numel()).reshape(1, 1, 1, -1)
    flat_y = Y.reshape(1, 1, 1, -1)
    flat_z = Z.reshape(1, 1, 1, -1)
    pts_3d = torch.stack([flat_x, flat_y, flat_z], dim=-1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=150)
    axes = axes.flatten()

    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    cases = [
        ("lift(spherical)", m["lift_raw"](m["spherical_sag"](curvature))),
        ("lift(ellipsoid κ=-0.5)",
         m["lift_raw"](m["conical_sag"](curvature, torch.tensor([-0.5])))),
        ("lift(paraboloid κ=-1)",
         m["lift_raw"](m["conical_sag"](curvature, torch.tensor([-1.0])))),
        ("lift(hyperboloid κ=-2)",
         m["lift_raw"](m["conical_sag"](curvature, torch.tensor([-2.0])))),
        ("lift(aspheric)",
         m["lift_raw"](m["aspheric_sag"](
             curvature, k0, torch.tensor([[0.5, 0.1]]), norm1))),
        ("lift(flat_sag)", m["lift_raw"](m["flat_sag"]())),
    ]

    for ax, (title, impl_fn) in zip(axes, cases):
        f_vals = eval_impl_val(impl_fn, pts_3d).reshape(Z.shape)
        vmax = max(float(np.abs(f_vals).max()), 0.5)
        im = ax.imshow(
            f_vals,
            extent=[z_vals.min().item(), z_vals.max().item(),
                    y_vals.min().item(), y_vals.max().item()],
            origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="f(0, y, z)")
        ax.contour(Z.numpy(), Y.numpy(), f_vals, levels=[0.0],
                   colors="black", linewidths=1.5)
        ax.set_xlabel("z (optical axis)")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig.suptitle("Lift — 2D Cross-Sections f(x=0, y, z)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_lift_cross_sections.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 9：Solver 光线-曲面交点图（2D：z-y 平面）──

def plot_solver_ray_diagrams(m) -> None:
    """在 (z, y) 平面上绘制光线与曲面的交点（光轴=z, x=0 截面）。"""
    curv = 0.2
    z0 = -5.0

    def _rays_from_offsets(y_offsets):
        y = torch.tensor(y_offsets, dtype=torch.float32)
        N = len(y_offsets)
        P = torch.zeros(1, 1, 1, N, 3)
        P[..., 2] = z0   # z = z0
        P[..., 1] = y    # y = offset
        V = torch.zeros(1, 1, 1, N, 3)
        V[..., 2] = 1.0  # 沿 +z
        return P, V

    def _solve_rays(impl_fn, P, V):
        return m["solve"](m["NewtonSolverOptions"](num_iter=15, damping=0.95))(
            points=P, directions=V, implicit=impl_fn)

    def _surface_profile(sag_fn, y_lim=(-6, 6), n=400):
        y = torch.linspace(y_lim[0], y_lim[1], n)
        pts = (torch.stack([torch.zeros_like(y), y], dim=-1)
               .unsqueeze(0).unsqueeze(0).unsqueeze(0))
        with torch.no_grad():
            r = sag_fn(pts, order=0)
            z = r.value.squeeze().numpy()
            hold = r.verdict.hold.squeeze().numpy()
        return y.numpy(), z, hold

    def _plot_ray_diagram(ax, sag_fn, P, V, res, title,
                          xlim=(-6, 4), ylim=(-6, 6)):
        y_surf, z_surf, hold_surf = _surface_profile(sag_fn, y_lim=ylim)
        # 曲面: z vs y
        ax.plot(z_surf[hold_surf], y_surf[hold_surf], color=C_SURFACE, lw=2.5,
                label="surface")
        ax.plot(z_surf[~hold_surf], y_surf[~hold_surf], color=C_SURFACE,
                lw=1.5, ls="--", alpha=0.4, label="surface (OOB)")

        P_np = P.squeeze().numpy()
        d_np = res.distances.squeeze().detach().numpy()
        hold_solver = res.verdict.hold.squeeze().numpy()
        val_np = res.value.squeeze().detach().numpy()

        for i in range(P_np.shape[0]):
            # P: (x, y, z); V: (dx, dy, dz) = (0, 0, 1)
            pz = P_np[i, 2]; py = P_np[i, 1]
            d_i = d_np[i]
            if hold_solver[i]:
                hz = pz + d_i
                color, style = "#2E7D32", "-"
            else:
                hz = pz + 2.0
                color, style = "#D32F2F", "--"
            ax.plot([pz, hz], [py, py], color=color, lw=1.8, ls=style)
            ax.scatter([hz], [py], color=color, s=40, zorder=5)
            if hold_solver[i]:
                ax.annotate(f"|f|={abs(val_np[i]):.2e}",
                            xy=(hz, py), xytext=(5, 5),
                            textcoords="offset points", fontsize=6, color=color)

        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.set_xlabel("z (optical axis)"); ax.set_ylabel("y")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=6.5, loc="upper right")

    k0 = torch.zeros(1)
    norm1 = torch.tensor([2.0])
    sag_sphere = m["spherical_sag"](torch.tensor([curv]))
    sag_parabola = m["conical_sag"](torch.tensor([curv]), torch.tensor([-1.0]))
    sag_hyperbola = m["conical_sag"](torch.tensor([curv]), torch.tensor([-2.0]))
    sag_asphere = m["aspheric_sag"](torch.tensor([curv]), k0,
                                    torch.tensor([[0.5, 0.1]]), norm1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

    P_sph, V_sph = _rays_from_offsets([-3.0, -1.5, 0.0, 1.5, 3.0, 4.9, 5.5])
    res_sph = _solve_rays(m["lift_raw"](sag_sphere), P_sph, V_sph)
    _plot_ray_diagram(axes[0, 0], sag_sphere, P_sph, V_sph, res_sph,
                      f"sphere (c={curv}) — normal/off-axis/grazing/OOB")

    P_para, V_para = _rays_from_offsets([-3.0, -1.5, 0.0, 1.5, 3.0])
    res_para = _solve_rays(m["lift_raw"](sag_parabola), P_para, V_para)
    _plot_ray_diagram(axes[0, 1], sag_parabola, P_para, V_para, res_para,
                      f"paraboloid (κ=-1, c={curv})")

    P_hyp, V_hyp = _rays_from_offsets([-3.0, -1.5, 0.0, 1.5, 3.0])
    res_hyp = _solve_rays(m["lift_raw"](sag_hyperbola), P_hyp, V_hyp)
    _plot_ray_diagram(axes[1, 0], sag_hyperbola, P_hyp, V_hyp, res_hyp,
                      f"hyperboloid (κ=-2, c={curv})")

    P_asp, V_asp = _rays_from_offsets([-3.0, -1.5, 0.0, 1.5, 3.0, 4.9, 5.5])
    res_asp = _solve_rays(m["lift_raw"](sag_asphere), P_asp, V_asp)
    _plot_ray_diagram(axes[1, 1], sag_asphere, P_asp, V_asp, res_asp,
                      "aspheric (sphere base + small α)")

    fig.suptitle("Newton Solver — Ray-Surface Intersection (z-y plane, x=0)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_solver_rays.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 10：收敛曲线 ──

def plot_solver_convergence(m) -> None:
    curv = 0.2; damping = 0.95
    sag = m["spherical_sag"](torch.tensor([curv]))
    impl = m["lift_raw"](sag)

    P = torch.tensor([[[[[0.0, 1.5, -5.0]]]]])  # (x=0, y=1.5, z=-5)
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    max_iter = 12

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for ax, solver_name, step_fn in [
        (axes[0], "Newton (order=1)", m["newton_step"]),
        (axes[1], "Halley (order=2)", m["halley_step"]),
    ]:
        distance = m["guess"](P, V, implicit=impl,
                              init_method=m["InitMethod"].CLOSEST)
        residuals = []
        with torch.no_grad():
            for _ in range(max_iter + 1):
                step = step_fn(distance, P, V, impl)
                hit = P.add(V.mul(distance.unsqueeze(-1)))
                residuals.append(abs(impl(hit, order=0).value.item()))
                distance = distance.sub(step.delta.mul(damping))

        iters = np.arange(len(residuals))
        ax.semilogy(iters, np.maximum(residuals, 1e-16), "o-",
                    color=C_NORMAL, lw=2, markersize=5)
        ax.axhline(1e-4, color=C_BOUNDARY, ls="--", lw=1.2, label="tol=1e-4")
        ax.set_xlabel("iteration"); ax.set_ylabel("|f(P + tV)|")
        ax.set_title(f"{solver_name} convergence (y=1.5)",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=7)

    fig.suptitle("Solver Convergence — Newton vs Halley",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_solver_convergence.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 11：初值方法对比 ──

def plot_solver_init_methods(m) -> None:
    curv = 0.2; z0 = -5.0
    sag = m["spherical_sag"](torch.tensor([curv]))
    impl = m["lift_raw"](sag)

    y_offsets = [-3.0, -1.5, 0.0, 1.5, 3.0]
    y = torch.tensor(y_offsets, dtype=torch.float32)
    N = len(y_offsets)
    P = torch.zeros(1, 1, 1, N, 3)
    P[..., 2] = z0; P[..., 1] = y
    V = torch.zeros(1, 1, 1, N, 3)
    V[..., 2] = 1.0

    torch.manual_seed(42)
    init_methods = [m["InitMethod"].CLOSEST, m["InitMethod"].ZERO,
                    m["InitMethod"].RANDOM]
    distances = {}; values = {}

    for method in init_methods:
        opt = m["NewtonSolverOptions"](num_iter=15, init_method=method)
        res = m["solve"](opt)(points=P, directions=V, implicit=impl)
        distances[method] = res.distances.squeeze().detach().numpy()
        values[method] = np.abs(res.value.squeeze().detach().numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    x_pos = np.arange(N); width = 0.25
    colors = [C_NORMAL, C_TOLL, C_ALIVE]

    for i, method in enumerate(init_methods):
        axes[0].bar(x_pos + i * width, distances[method], width,
                    label=method.value, alpha=0.85, color=colors[i])
    axes[0].set_xticks(x_pos + width)
    axes[0].set_xticklabels([f"y={v}" for v in y_offsets])
    axes[0].set_ylabel("distance t")
    axes[0].set_title("sphere: intersection distance by init method",
                      fontsize=9, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y"); axes[0].legend(fontsize=7)

    for i, method in enumerate(init_methods):
        axes[1].bar(x_pos + i * width, np.maximum(values[method], 1e-16), width,
                    label=method.value, alpha=0.85, color=colors[i])
    axes[1].set_xticks(x_pos + width)
    axes[1].set_xticklabels([f"y={v}" for v in y_offsets])
    axes[1].set_ylabel("|f|"); axes[1].set_yscale("log")
    axes[1].set_title("sphere: final residual by init method",
                      fontsize=9, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y"); axes[1].legend(fontsize=7)

    fig.suptitle("Solver — Init Method Comparison (Newton, sphere c=0.2)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_solver_init_methods.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 12：allow_negative 效应 ──

def plot_solver_allow_negative_effect(m) -> None:
    curv = 0.2
    sag = m["spherical_sag"](torch.tensor([curv]))
    impl = m["lift_raw"](sag)

    # 光线起点在曲面后的 z>0 处，沿 +z
    P = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    V = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])

    res_clamp = m["solve"](m["NewtonSolverOptions"](
        num_iter=15, allow_negative=False))(points=P, directions=V, implicit=impl)
    res_noclamp = m["solve"](m["NewtonSolverOptions"](
        num_iter=15, allow_negative=True))(points=P, directions=V, implicit=impl)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    labels = ["allow_negative=False", "allow_negative=True"]
    distances = [res_clamp.distances.item(), res_noclamp.distances.item()]
    values = [abs(res_clamp.value.item()), abs(res_noclamp.value.item())]
    bar_hold = [res_clamp.verdict.hold.item(), res_noclamp.verdict.hold.item()]
    bar_colors = [C_ALIVE if h else C_BOUNDARY for h in bar_hold]

    ax.bar(labels, distances, color=bar_colors, alpha=0.8)
    ax.set_ylabel("distance t")
    ax.set_title("starting inside sphere (z=1, +z direction)",
                 fontsize=10, fontweight="bold")

    for i, (d, v, h) in enumerate(zip(distances, values, bar_hold)):
        va = "bottom" if d >= 0 else "top"
        ax.annotate(f"t={d:.3f}\n|f|={v:.2e}\nhold={h}",
                    xy=(i, d), ha="center", va=va, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_solver_allow_negative.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 13：Solver verdict 优先级 ──

def plot_solver_verdict_priority(m) -> None:
    def _scenario(ax, title, P, V, impl, num_iter, description):
        res = m["solve"](m["HalleySolverOptions"](num_iter=num_iter))(
            points=P, directions=V, implicit=impl)
        hit = P.add(V.mul(res.distances.unsqueeze(-1)))

        surface_v = impl(hit, order=0).verdict
        f_val = impl(hit, order=0).value
        neg_toll = res.distances
        conv_toll = torch.full_like(f_val, 1e-4).sub(f_val.abs())

        labels = ["surface toll", "negative toll\n(distance)",
                  "convergence toll\n(tol-|f|)", "recorded toll"]
        vals = [surface_v.toll.item(), neg_toll.item(),
                conv_toll.item(), res.verdict.toll.item()]
        bar_colors = [C_BOUNDARY, C_GRAD, C_TOLL, C_NORMAL]
        bars = ax.bar(labels, vals, color=bar_colors, alpha=0.8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("toll"); ax.set_title(title, fontsize=9, fontweight="bold")
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            ax.annotate(f"{val:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3 if h >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
        ax.annotate(description, xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.9))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # A: OOB-first
    c_oob = torch.tensor([1.0])
    impl_oob = m["lift_raw"](m["spherical_sag"](c_oob))
    _scenario(axes[0], "surface dies first (OOB)",
              torch.tensor([[[[[2.0, 0.0, -5.0]]]]]),
              torch.tensor([[[[[0.0, 0.0, 1.0]]]]]),
              impl_oob, num_iter=2,
              description="surface dead at first evaluation")

    # B: convergence-first
    c_conv = torch.tensor([0.05])
    impl_conv = m["lift_raw"](m["spherical_sag"](c_conv))
    _scenario(axes[1], "convergence dies first",
              torch.tensor([[[[[0.3, 0.2, -2.0]]]]]),
              torch.tensor([[[[[0.0, 0.0, 1.0]]]]]),
              impl_conv, num_iter=1,
              description="surface alive, solver not converged")

    fig.suptitle("Solver Verdict Priority — First Death Freezes the Toll",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_solver_verdict_priority.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 14：梯度精度验证 ──

def plot_gradient_accuracy(m) -> None:
    curvature = torch.tensor([0.5])
    k0 = torch.zeros_like(curvature)
    norm1 = torch.tensor([2.0])
    eps = 1e-4
    r_vals = torch.linspace(0.05, 2.8, 200)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)
    axes = axes.flatten()

    cases = [
        ("spherical_sag", m["spherical_sag"](curvature)),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5]))),
        ("conical ellipsoid κ=-0.5",
         m["conical_sag"](curvature, torch.tensor([-0.5]))),
        ("conical paraboloid κ=-1",
         m["conical_sag"](curvature, torch.tensor([-1.0]))),
        ("conical hyperboloid κ=-2",
         m["conical_sag"](curvature, torch.tensor([-2.0]))),
        ("aspheric normal",
         m["aspheric_sag"](curvature, k0, torch.tensor([[0.5, 0.1]]), norm1)),
        ("aspheric critical",
         m["aspheric_sag"](curvature, k0, torch.tensor([[2.0, -0.5]]), norm1)),
        ("flat_sag", m["flat_sag"]()),
    ]

    for ax, (name, sag_fn) in zip(axes, cases):
        grad_analytic = []; grad_fd = []
        for r_val in r_vals:
            r_t = r_val.reshape(1, 1, 1, 1)
            pt = torch.stack([r_t, torch.zeros_like(r_t)], dim=-1)
            g = sag_fn(pt, order=1).gradient
            grad_analytic.append(g[0, 0, 0, 0, 0].item())
            pt_p = torch.stack([r_t + eps, torch.zeros_like(r_t)], dim=-1)
            pt_m = torch.stack([r_t - eps, torch.zeros_like(r_t)], dim=-1)
            vp = sag_fn(pt_p, order=0).value.item()
            vm = sag_fn(pt_m, order=0).value.item()
            grad_fd.append((vp - vm) / (2 * eps))

        grad_analytic = np.array(grad_analytic); grad_fd = np.array(grad_fd)
        err = np.abs(grad_analytic - grad_fd)
        r_np = r_vals.numpy()

        ax.plot(r_np, grad_analytic, color=C_NORMAL, lw=2.0,
                label="analytic ∂s/∂x")
        ax.plot(r_np, grad_fd, color=C_TOLL, lw=1.5, ls="--",
                label="FD ∂s/∂x")
        ax2 = ax.twinx()
        ax2.plot(r_np, np.maximum(err, 1e-16), color=C_BOUNDARY, lw=1.0,
                 ls=":", alpha=0.7)
        ax2.set_yscale("log")
        ax.set_xlabel("r"); ax.set_ylabel("∂sag/∂x")
        ax2.set_ylabel("|error| (log)", color=C_BOUNDARY, fontsize=7)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=6, loc="upper left")

    fig.suptitle("Gradient Accuracy — Analytic vs FD (eps=1e-4)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_gradient_accuracy.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ── 图 15：Batch 广播 ──

def plot_batch_broadcast(m) -> None:
    curvatures = torch.tensor([0.1, 0.3, 0.5, 0.8])
    sag = m["spherical_sag"](curvatures)
    r = torch.linspace(0.0, 3.5, 400)
    r_b = r.reshape(1, 1, 1, -1).expand(len(curvatures), 1, 1, -1)
    y_b = torch.zeros_like(r_b)
    pts = torch.stack([r_b, y_b], dim=-1)

    with torch.no_grad():
        result = sag(pts, order=0)
        sag_vals = result.value.detach().cpu().numpy()
        hold_vals = result.verdict.hold.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()
    batch_colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    for i, (ax, c) in enumerate(zip(axes, curvatures.tolist())):
        r_b_c = 1.0 / abs(c)
        s = sag_vals[i, 0, 0, :]
        h = hold_vals[i, 0, 0, :]
        ax.plot(r.numpy(), s, color=batch_colors[i], lw=2.0,
                label=f"c={c:.2f}")
        _shade_oob_mask(ax, r.numpy(), ~h)
        ax.axvline(r_b_c, color=C_BOUNDARY, lw=1.5, ls="--",
                   label=f"r_b={r_b_c:.2f}")
        ax.set_xlabel("r"); ax.set_ylabel("sag z")
        ax.set_title(f"batch item {i}: c={c:.2f}",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("Batch Broadcasting — Spherical Sag for B=4 Curvatures",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "implicit_batch_broadcast.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[OK] -> {out.name}")


# ═══════════════════════════════════════════════════════════════════════════
# 第三部分：综合 sanity 检查
# ═══════════════════════════════════════════════════════════════════════════

def run_sanity_checks(m) -> None:
    print("\n" + "=" * 60)
    print("Sanity Checks")
    print("=" * 60)

    curv_test = torch.tensor([0.5])
    norm1 = torch.tensor([2.0])

    # 1. sag at origin
    origin_2d = radial_points_2d(torch.zeros(1))
    for name, sag_fn in [
        ("spherical", m["spherical_sag"](curv_test)),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5]))),
        ("conical ellipsoid",
         m["conical_sag"](curv_test, torch.tensor([-0.5]))),
        ("conical paraboloid",
         m["conical_sag"](curv_test, torch.tensor([-1.0]))),
        ("aspheric",
         m["aspheric_sag"](curv_test, torch.zeros(1),
                           torch.tensor([[0.5, 0.1]]), norm1)),
        ("flat", m["flat_sag"]()),
    ]:
        val = eval_sag_val(sag_fn, origin_2d)[0]
        check(f"sanity: {name} sag at r=0 == 0", abs(val) < 1e-9,
              f"got {val}")

    # 2. hold == (toll >= 0)
    r_test = torch.linspace(0.0, 3.5, 200)
    pts_test = radial_points_2d(r_test)
    k0 = torch.zeros(1)

    for name, sag_fn in [
        ("spherical", m["spherical_sag"](curv_test)),
        ("spherical concave", m["spherical_sag"](torch.tensor([-0.5]))),
        ("conical κ=-0.5", m["conical_sag"](curv_test, torch.tensor([-0.5]))),
        ("conical κ=-1", m["conical_sag"](curv_test, torch.tensor([-1.0]))),
        ("conical κ=-2", m["conical_sag"](curv_test, torch.tensor([-2.0]))),
        ("aspheric",
         m["aspheric_sag"](curv_test, k0,
                           torch.tensor([[0.5, 0.1]]), norm1)),
    ]:
        r = sag_fn(pts_test, order=2)
        hold_arr = r.verdict.hold.detach().cpu().numpy()
        toll_arr = r.verdict.toll.detach().cpu().numpy()
        ok = bool((hold_arr == (toll_arr >= 0.0)).all())
        check(f"sanity: {name} hold == (toll >= 0)", ok)

    # 3. flat_sag
    r_flat = m["flat_sag"]()(pts_test, order=2)
    check("sanity: flat hold all True", bool(r_flat.verdict.hold.all()))
    check("sanity: flat toll all 0（存活无代价）",
          bool((r_flat.verdict.toll == 0.0).all()))
    check("sanity: flat cause=NONE",
          (r_flat.verdict.cause == 0).all().item())

    # 4. OOB sag
    oob_pt = torch.tensor([[[[[2.0, 0.0]]]]])
    r_oob = m["spherical_sag"](torch.tensor([1.0]))(oob_pt, order=2)
    check("sanity: OOB hold=False", not r_oob.verdict.hold.item())
    check("sanity: OOB toll<0", r_oob.verdict.toll.item() < 0)
    check("sanity: OOB value=0", abs(r_oob.value.item()) < 1e-9)
    check("sanity: OOB grad=0", bool(torch.all(r_oob.gradient == 0)))
    check("sanity: OOB cause=SAG_DOMAIN",
          (r_oob.verdict.cause == m["Verdict"].Cause.SAG_DOMAIN).item())

    # 5. concave
    r_neg = torch.linspace(0.0, 1.99, 100)
    pts_neg = radial_points_2d(r_neg)
    r_nc = m["spherical_sag"](torch.tensor([-0.5]))(pts_neg, order=0)
    check("sanity: concave sag <= 0", bool((r_nc.value <= 0).all()))
    check("sanity: concave r<2 in-domain", bool(r_nc.verdict.hold.all()))

    # 6. Hessian symmetry
    for name, sag_fn in [
        ("spherical", m["spherical_sag"](curv_test)),
        ("conical κ=-0.5", m["conical_sag"](curv_test, torch.tensor([-0.5]))),
        ("conical κ=-1", m["conical_sag"](curv_test, torch.tensor([-1.0]))),
        ("conical κ=-2", m["conical_sag"](curv_test, torch.tensor([-2.0]))),
    ]:
        r = sag_fn(pts_test, order=2)
        h = r.hessian
        sym_err = (h[..., 0, 1] - h[..., 1, 0]).abs().max().item()
        check(f"sanity: {name} Hessian symmetry",
              sym_err < 1e-6, f"asym={sym_err:.2e}")

    # 7. lift: df/dz == -1
    impl_test = m["lift_raw"](m["spherical_sag"](torch.tensor([0.5])))
    r_lift = impl_test(torch.tensor([[[[[0.5, 0.3, 1.0]]]]]), order=1)
    dfdz = r_lift.gradient[0, 0, 0, 0, 2].item()
    check("sanity: lift df/dz == -1", abs(dfdz + 1.0) < 1e-6,
          f"got {dfdz}")

    # 8. Newton convergence
    c_sph = 0.05
    impl_sph = m["lift_raw"](m["spherical_sag"](torch.tensor([c_sph])))
    P_sph = torch.tensor([[[[[0.3, 0.2, -2.0]]]]])
    V_sph = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
    r2 = 0.3 ** 2 + 0.2 ** 2
    radicand = 1.0 - (c_sph ** 2) * r2
    expected_z = c_sph * r2 / (1.0 + np.sqrt(radicand))
    expected_d = 2.0 + expected_z

    res_newton = m["solve"](m["NewtonSolverOptions"](num_iter=10))(
        points=P_sph, directions=V_sph, implicit=impl_sph)
    err_d = abs(res_newton.distances.item() - expected_d)
    check("sanity: Newton converges", err_d < 1e-4, f"err={err_d:.2e}")
    check("sanity: Newton hold=True", bool(res_newton.verdict.hold.item()))
    check("sanity: Newton |value|<tol",
          abs(res_newton.value.item()) < 1e-4)

    # 9. Halley convergence
    res_halley = m["solve"](m["HalleySolverOptions"](num_iter=6))(
        points=P_sph, directions=V_sph, implicit=impl_sph)
    err_h = abs(res_halley.distances.item() - expected_d)
    check("sanity: Halley converges", err_h < 1e-4, f"err={err_h:.2e}")

    # 10. Gradient FD spot checks
    eps_fd = 1e-4
    for name, sag_spot in [
        ("convex sphere", m["spherical_sag"](torch.tensor([0.5]))),
        ("concave sphere", m["spherical_sag"](torch.tensor([-0.5]))),
    ]:
        pt_spot = torch.tensor([[[[[0.3, 0.2]]]]])
        g_analytic = sag_spot(pt_spot, order=1).gradient[0, 0, 0, 0].numpy()

        def _fd_grad(xv, yv, fn):
            ptxp = torch.tensor([[[[[xv + eps_fd, yv]]]]])
            ptxm = torch.tensor([[[[[xv - eps_fd, yv]]]]])
            ptyp = torch.tensor([[[[[xv, yv + eps_fd]]]]])
            ptym = torch.tensor([[[[[xv, yv - eps_fd]]]]])
            gx = ((fn(ptxp, order=0).value.item()
                   - fn(ptxm, order=0).value.item()) / (2 * eps_fd))
            gy = ((fn(ptyp, order=0).value.item()
                   - fn(ptym, order=0).value.item()) / (2 * eps_fd))
            return gx, gy

        gx_fd, gy_fd = _fd_grad(0.3, 0.2, sag_spot)
        check(f"sanity: {name} grad_x FD",
              abs(g_analytic[0] - gx_fd) < 1e-4)
        check(f"sanity: {name} grad_y FD",
              abs(g_analytic[1] - gy_fd) < 1e-4)

    # 11. Away ray → negative verdict
    P_away = torch.tensor([[[[[0.0, 0.0, -5.0]]]]])
    V_away = torch.tensor([[[[[0.0, 0.0, -1.0]]]]])
    res_away = m["solve"](m["NewtonSolverOptions"](num_iter=15))(
        points=P_away, directions=V_away, implicit=impl_sph)
    check("sanity: away ray hold=False",
          not res_away.verdict.hold.item())
    check("sanity: away ray distance<0",
          res_away.distances.item() < 0)

    # 12. Verdict priority
    c_prio = torch.tensor([1.0])
    impl_prio = m["lift_raw"](m["spherical_sag"](c_prio))
    res_prio = m["solve"](m["HalleySolverOptions"](num_iter=2))(
        points=torch.tensor([[[[[2.0, 0.0, -5.0]]]]]),
        directions=torch.tensor([[[[[0.0, 0.0, 1.0]]]]]),
        implicit=impl_prio)
    check("sanity: OOB-first ray dead",
          not res_prio.verdict.hold.item())
    hit_prio = (torch.tensor([[[[[2.0, 0.0, -5.0]]]]])
                .add(torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
                     .mul(res_prio.distances.unsqueeze(-1))))
    surface_toll = impl_prio(hit_prio, order=0).verdict.toll.item()
    check("sanity: OOB-first records |surface toll|",
          abs(res_prio.verdict.toll.item() - abs(surface_toll)) < 1e-3)

    # 13. Batch broadcasting
    curvs_batch = torch.tensor([0.1, 0.3, 0.5, 0.8])
    sag_batch = m["spherical_sag"](curvs_batch)
    r_vals = torch.linspace(0.0, 3.5, 200)
    r_bb = r_vals.reshape(1, 1, 1, -1).expand(len(curvs_batch), 1, 1, -1)
    z_bb = torch.zeros_like(r_bb)
    pts_batch = torch.stack([r_bb, z_bb], dim=-1)
    res_batch = sag_batch(pts_batch, order=0)
    check("sanity: batch output B dim",
          res_batch.value.shape[0] == len(curvs_batch))
    for i, c in enumerate(curvs_batch.tolist()):
        in_domain = (res_batch.verdict.hold[i, 0, 0, :]
                     .detach().cpu().numpy())
        in_idx = np.where(in_domain)[0]
        oob_idx = np.where(~in_domain)[0]
        if len(in_idx) > 0 and len(oob_idx) > 0:
            approx_rb = ((r_vals.numpy()[in_idx[-1]]
                          + r_vals.numpy()[oob_idx[0]]) / 2.0)
            check(f"sanity: batch c={c} boundary",
                  abs(approx_rb - 1.0 / c) < 0.05)

    print("\n" + "=" * 60)
    print("Sanity checks complete.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

_VISUAL_TESTS = [
    plot_sag_value_sections,
    plot_sag_value_heatmaps,
    plot_sturdy_effects,
    plot_sag_gradient_sections,
    plot_sag_gradient_heatmaps,
    plot_sag_hessian_sections,
    plot_verdict_toll_sections,
    plot_lift_cross_sections,
    plot_solver_ray_diagrams,
    plot_solver_convergence,
    plot_solver_init_methods,
    plot_solver_allow_negative_effect,
    plot_solver_verdict_priority,
    plot_gradient_accuracy,
    plot_batch_broadcast,
]


def main() -> int:
    matplotlib.use("Agg")
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9})
    plt.rcParams["figure.max_open_warning"] = 0

    no_show = "--no-show" in sys.argv

    m = _import_implicit()

    # ── 断言测试 ──
    print("=" * 60)
    print("Part 1: Assertion Tests")
    print("=" * 60)
    for t in _ASSERTION_TESTS:
        print(f"\n== {t.__name__} ==")
        t(m)

    if _FAILED:
        print("\n" + "=" * 60)
        print(f"ASSERTION FAILURES: {len(_FAILED)}")
        for f in _FAILED:
            print(f"  - {f}")
        print("=" * 60)

    # ── 可视化 ──
    print("\n" + "=" * 60)
    print(f"Part 2: Visualizations ({len(_VISUAL_TESTS)} figures)")
    print("=" * 60)
    for vt in _VISUAL_TESTS:
        print(f"\n-- {vt.__name__} --")
        vt(m)

    # ── sanity 检查 ──
    run_sanity_checks(m)

    # ── 汇总 ──
    print("\n" + "=" * 60)
    if _FAILED:
        print(f"FAILED {len(_FAILED)} assertions: {_FAILED}")
        return 1
    print(f"ALL PASS ({len(_ASSERTION_TESTS)} assertion groups"
          f" + all sanity checks"
          f" + {len(_VISUAL_TESTS)} figures)")
    print("=" * 60)

    if not no_show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
