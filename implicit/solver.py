from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol, cast

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


@dataclass(frozen=True, slots=True)
class _StepResult:
    _delta: RayFloatScalar
    _value: RayFloatScalar
    _verdict: Verdict

    @property
    def delta(self) -> RayFloatScalar:
        return self._delta

    @property
    def value(self) -> RayFloatScalar:
        return self._value

    @property
    def verdict(self) -> Verdict:
        return self._verdict


class _StepFunction(Protocol):
    def __call__(
        self,
        distances: RayFloatScalar,
        points: RayFloat3D,
        directions: RayFloat3D,
        implicit: ImplicitFunction,
    ) -> _StepResult: ...


def _evaluate(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
    *,
    order: FieldResult.Order,
) -> _StepResult:
    """沿光线推进 *distances*，在命中点评估隐式函数并装配一步的结果。

    ``order=GRADIENT`` 走 Newton（一阶），``order=HESSIAN`` 走 Halley（三阶）；
    两者共用本函数，仅在步长公式上分叉，``verdict`` 一律取该命中点的隐式裁决。
    """
    points_at_t = points.add(directions.mul(distances.unsqueeze(-1)))
    r = implicit(points_at_t, order=order)

    f = r.value
    f_prime = r.gradient.mul(directions).sum(dim=-1)

    if order >= FieldResult.Order.HESSIAN:
        hess_dot_dir = torch.einsum("...ij,...j->...i", r.hessian, directions)
        f_double_prime = hess_dot_dir.mul(directions).sum(dim=-1)
        # 真 Halley：三阶收敛，每步仅需求一个 Hessian。
        delta = sturdy_div(
            f_prime.mul(f).mul(2.0),
            f_prime.square().mul(2.0).sub(f.mul(f_double_prime)),
        )
    else:
        delta = sturdy_div(f, f_prime)

    return _StepResult(_delta=delta, _value=f, _verdict=r.verdict)


def newton_step(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
) -> _StepResult:
    """Newton 单步（一阶收敛，``order=GRADIENT``）。"""
    return _evaluate(
        distances, points, directions, implicit, order=FieldResult.Order.GRADIENT
    )


def halley_step(
    distances: RayFloatScalar,
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
) -> _StepResult:
    """Halley 单步（三阶收敛，``order=HESSIAN``，每步需求一个 Hessian）。"""
    return _evaluate(
        distances, points, directions, implicit, order=FieldResult.Order.HESSIAN
    )


def _solve(
    points: RayFloat3D,
    directions: RayFloat3D,
    implicit: ImplicitFunction,
    *,
    options: NewtonSolverOptions,
    step_fn: _StepFunction,
) -> SolverResult:
    distances = guess(
        points, directions, implicit=implicit, init_method=options.init_method
    )

    # 推进阶段：前 N-1 步无裁决意义，全程 no_grad 只把 distances 推近终值。
    # 推进链不夹负值——distances 的正负与梯度都诚实保留，clamp/判死交给裁决阶段。
    with torch.no_grad():
        for _ in range(options.num_iter - 1):
            step = step_fn(distances, points, directions, implicit)
            distances = distances.sub(step.delta.mul(options.damping))

    # 最后一步带梯度推进：distances 在此建立对曲面参数（经 implicit 的 c/κ/α）
    # 的梯度链——像差损失/死亡损失反向传播所必需。仅展开最后一步既稳定又够用。
    step = step_fn(distances, points, directions, implicit)
    distances = distances.sub(step.delta.mul(options.damping))

    # 裁决阶段：在最终 distances 上纯评估（带梯度），使 _distances/_value/_verdict
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
        _value=r.value,
        _distances=distances,
        _verdict=r.verdict.at(negative).at(convergence),
    )


# 求解方法 → 单步步长函数。options 由 _solve 整体读取，
# 避免对 slots dataclass 做脆弱的字段拆解；分派按枚举值而非选项类型，
# 子类化 NewtonSolverOptions 不会破坏分派。
_STEP_OF: Final[dict[NewtonSolverOptions.Method, _StepFunction]] = {
    NewtonSolverOptions.Method.NEWTON: newton_step,
    NewtonSolverOptions.Method.HALLEY: halley_step,
}


def make_solver_options(**kwargs) -> NewtonSolverOptions:
    """按 ``method`` 分派求解器选项类型，返回一个 dataclass 实例。"""
    method = term.METHOD.resolve(kwargs, default="newton")
    kwargs = {k: v for k, v in kwargs.items() if k not in term.METHOD}
    if method == NewtonSolverOptions.Method.NEWTON:
        return NewtonSolverOptions().update(**kwargs)
    elif method == NewtonSolverOptions.Method.HALLEY:
        return HalleySolverOptions().update(**kwargs)
    else:
        raise ValueError(f"Unknown solve method: {method}") from None


def solve(options: NewtonSolverOptions) -> SolverFunction:
    """按 ``options.method`` 分派单步格式，返回整体求解闭包。"""
    try:
        step_fn = _STEP_OF[options.method]
    except KeyError:
        raise ValueError(f"Unknown solve method: {options.method}") from None
    return cast(SolverFunction, partial(_solve, options=options, step_fn=step_fn))
