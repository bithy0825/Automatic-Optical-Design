"""迭代求解器：Newton / Halley 沿光线求隐式曲面交点距离。

两阶段结构：前 N−1 步在 ``no_grad`` 下把 distances 推近终值（无裁决意义，
不建图省显存）；末步带梯度推进并对最终命中点纯评估一次，使
``distances`` / ``value`` / ``verdict`` 严格同点。
"""

from collections.abc import Callable
from functools import partial
from typing import Final, cast

import torch

from core import RayFloat3D, RayFloatScalar, Verdict, sturdy_div, term
from implicit.guess import guess
from implicit.protocol import (
    FieldResult,
    HalleySolverOptions,
    ImplicitFunction,
    NewtonSolverOptions,
    SolverFunction,
    SolverResult,
)

type _StepFunction = Callable[
    [RayFloatScalar, RayFloat3D, RayFloat3D, ImplicitFunction], RayFloatScalar
]


def _step(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
    *,
    order: FieldResult.Order,
) -> RayFloatScalar:
    """沿光线推进 *distances*，在命中点评估隐式函数并返回单步位移。

    ``order=GRADIENT`` 走 Newton（一阶），``order=HESSIAN`` 走 Halley（三阶）；
    两者共用本函数，仅在步长公式上分叉。
    """
    r = implicit(points.add(directions.mul(distances.unsqueeze(-1))), order=order)

    f = r.value
    f_prime = r.grad().mul(directions).sum(dim=-1)

    if order >= FieldResult.Order.HESSIAN:
        hess_dot_dir = torch.einsum("...ij,...j->...i", r.hess(), directions)
        f_double_prime = hess_dot_dir.mul(directions).sum(dim=-1)
        # 真 Halley：三阶收敛，每步仅需求一个 Hessian。
        return sturdy_div(
            f_prime.mul(f).mul(2.0),
            f_prime.square().mul(2.0).sub(f.mul(f_double_prime)),
        )
    return sturdy_div(f, f_prime)


def newton_step(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
) -> RayFloatScalar:
    """Newton 单步位移（一阶收敛，``order=GRADIENT``）。"""
    return _step(distances, points, directions, implicit, order=FieldResult.Order.GRADIENT)


def halley_step(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
) -> RayFloatScalar:
    """Halley 单步位移（三阶收敛，``order=HESSIAN``，每步需求一个 Hessian）。"""
    return _step(distances, points, directions, implicit, order=FieldResult.Order.HESSIAN)


def _solve(
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
    *,
    options: NewtonSolverOptions,
    step_fn: _StepFunction,
) -> SolverResult:
    distances = guess(points, directions, implicit=implicit, init_method=options.init_method)

    # 推进阶段：前 N-1 步无裁决意义，全程 no_grad 只把 distances 推近终值。
    # 推进链不夹负值——distances 的正负与梯度都诚实保留，clamp/判死交给裁决阶段。
    with torch.no_grad():
        for _ in range(options.num_iter - 1):
            delta = step_fn(distances, points, directions, implicit)
            distances = distances.sub(delta.mul(options.damping))

    # 最后一步带梯度推进：distances 在此建立对曲面参数（经 implicit 的 c/κ/α）
    # 的梯度链——像差损失/死亡损失反向传播所必需。仅展开最后一步既稳定又够用。
    delta = step_fn(distances, points, directions, implicit)
    distances = distances.sub(delta.mul(options.damping))

    # 裁决阶段：在最终 distances 上纯评估（带梯度），使 distances/value/verdict
    # 严格同点。step 评估于更新前的点，不可直接拿来裁决。
    hit = points.add(directions.mul(distances.unsqueeze(-1)))
    r = implicit(hit, order=FieldResult.Order.VALUE)

    # 三裁决（界内 ≥0、越界 <0 的可微 toll），按 at 链组合——首次判死的站点钉死
    # toll，后续站点不改（已死 need_update=False），契合"死亡即冻结、绝不复活"：
    #   shape 域   —— 面型 sag 域裁决，base。toll=radicand，罚落点 y/z 越界。
    #   negative   —— 负 distances 裁决。holds = allow_negative OR d≥0。中间面
    #                (allow_negative=False) 负 distances = 边缘 X 型打架，物理不可能
    #                → 判死，toll=d，罚不合理物理、驱动交点回前方；首面
    #                (allow_negative=True) holds 恒真 → 负根合法存活，不判死。
    #   convergence —— 收敛裁决。toll=tol−|f|，不收敛判死，罚收敛程度、驱动残差→0。
    # 链序 negative→convergence：同时多病时负 distances（物理根因）先于残差（数值
    # 次生）记录——拉回前方后收敛自然解决。allow_negative=True 时 negative 永不
    # 判死，链退化为 shape.at(convergence)。
    residual_toll = torch.full_like(r.value, options.tol).sub(r.value.abs())
    negative = Verdict.site(
        hold=distances.ge(0.0).logical_or(
            torch.full_like(distances, options.allow_negative, dtype=torch.bool)
        ),
        toll=distances,
        cause=Verdict.Cause.SOLVER_NEGATIVE,
    )
    convergence = Verdict.site(
        hold=residual_toll.ge(0.0),
        toll=residual_toll,
        cause=Verdict.Cause.SOLVER_CONVERGENCE,
    )

    return SolverResult(
        distances=distances,
        value=r.value,
        verdict=r.verdict.at(negative).at(convergence),
    )


# 求解方法 → 单步位移函数。options 由 _solve 整体读取，避免对 slots dataclass
# 做脆弱的字段拆解；分派按枚举值而非选项类型，子类化 NewtonSolverOptions
# 不会破坏分派。
_STEP_OF: Final[dict[NewtonSolverOptions.Method, _StepFunction]] = {
    NewtonSolverOptions.Method.NEWTON: newton_step,
    NewtonSolverOptions.Method.HALLEY: halley_step,
}

_OPTIONS_OF: Final[dict[NewtonSolverOptions.Method, type[NewtonSolverOptions]]] = {
    NewtonSolverOptions.Method.NEWTON: NewtonSolverOptions,
    NewtonSolverOptions.Method.HALLEY: HalleySolverOptions,
}


def make_solver_options(**kwargs) -> NewtonSolverOptions:
    """按 ``method`` 键分派选项类型，其余键经 ``update`` 透传字段更新。"""
    method = NewtonSolverOptions.Method(term.METHOD.resolve(kwargs, default="newton"))
    kwargs = {k: v for k, v in kwargs.items() if k not in term.METHOD}
    return _OPTIONS_OF[method]().update(**kwargs)


def solve(options: NewtonSolverOptions) -> SolverFunction:
    """按 ``options.method`` 分派单步格式，返回整体求解闭包。"""
    step_fn = _STEP_OF[NewtonSolverOptions.Method(options.method)]
    return cast(SolverFunction, partial(_solve, options=options, step_fn=step_fn))
