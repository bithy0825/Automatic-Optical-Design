"""归一化区域（line / disk / rect）上的点集采样。

所有采样器输出归一化坐标：line → ``(N,)`` ∈ [-1, 1]；disk → ``(N, 2)``
单位圆盘；rect → ``(N, 2)`` ∈ [-1, 1]²。物理尺度（光瞳半径、视场角、
波长范围）的映射由调用方完成。

合法 (method, region) 组合由 :data:`_SAMPLERS` 分派表唯一界定：
:meth:`SampleOptions.__post_init__` 的校验与 :func:`sample` 的分派都
从同一张表派生，不存在第二处事实源。
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, Self, cast, final, get_args

import torch
from jaxtyping import Float
from torch import Tensor

from core import term

type SampleMethod = Literal["uniform", "random", "fibonacci"]
type SampleRegion = Literal["line", "disk", "rect"]
type SampleCount = int | tuple[int, int]

type LineSamples = Float[Tensor, "N"]  # [-1, 1] 上 N 个采样位置
type PlaneSamples = Float[Tensor, "N 2"]  # 单位区域上 N 个 (x, y) 采样点

_VALID_METHODS: Final = frozenset(get_args(SampleMethod.__value__))
_VALID_REGIONS: Final = frozenset(get_args(SampleRegion.__value__))

_GOLDEN_ANGLE: Final[float] = torch.pi * (3.0 - 5.0**0.5)  # π·(3 − √5)


# ── 采样器 ──


def _axis_coords(n: int) -> LineSamples:
    """单轴坐标：``n=1`` → ``[0]``，否则 [-1, 1] 上 *n* 等分。"""
    return torch.zeros(1) if n == 1 else torch.linspace(-1.0, 1.0, n)


def _line_uniform(count: int) -> LineSamples:
    """[-1, 1] 等距采样 *count* 个点。"""
    return _axis_coords(count)


def _line_random(count: int) -> LineSamples:
    """[-1, 1] 均匀随机采样 *count* 个点。"""
    return torch.rand(count).mul_(2.0).sub_(1.0)


def _disk_uniform(count: tuple[int, int]) -> PlaneSamples:
    """单位圆盘网格采样（同心环 + 等分圆周），首点为圆心。

    *count* 为 ``(rings, sectors)``（*rings* 含圆心）；半径开平方加密
    使环带等面积。总点数 = rings×sectors − sectors + 1。
    """
    rings, sectors = count
    radii = torch.linspace(0.0, 1.0, rings)[1:]
    angles = torch.arange(sectors, dtype=torch.get_default_dtype()).div_(sectors).mul_(2 * torch.pi)

    r, a = torch.meshgrid(radii.sqrt(), angles, indexing="ij")
    x, y = r.mul(a.cos()).ravel(), r.mul(a.sin()).ravel()
    ring_pts = torch.stack((x, y), dim=-1)  # ((rings-1)*sectors, 2)
    return torch.cat((torch.zeros(1, 2), ring_pts), dim=0)


def _disk_random(count: int) -> PlaneSamples:
    """单位圆盘均匀随机采样 *count* 个点（半径开平方 = 等面积）。"""
    rho = torch.rand(count).sqrt_()
    theta = torch.rand(count).mul_(2 * torch.pi)
    return torch.stack((rho * theta.cos(), rho * theta.sin()), dim=-1)


def _disk_fibonacci(count: int) -> PlaneSamples:
    """单位圆盘 Fibonacci 螺旋采样 *count* 个点（等面积，首点为圆心）。"""
    if count == 1:
        return torch.zeros(1, 2)

    i = torch.arange(1, count, dtype=torch.get_default_dtype())
    rho = i.div(count - 1).sqrt()
    theta = i.mul(_GOLDEN_ANGLE)

    ring = torch.stack((rho * theta.cos(), rho * theta.sin()), dim=-1)
    return torch.cat((torch.zeros(1, 2), ring))


def _rect_uniform(count: tuple[int, int]) -> PlaneSamples:
    """[-1, 1]² 矩形网格采样：*count* 为 ``(na, nb)`` 两轴网格数。

    输出列 0 沿第一轴取 *na* 个值、列 1 沿第二轴取 *nb* 个值，
    总点数 = na × nb。
    """
    na, nb = count
    aa, bb = torch.meshgrid(_axis_coords(na), _axis_coords(nb), indexing="ij")
    return torch.stack((aa.ravel(), bb.ravel()), dim=-1)


def _rect_random(count: int) -> PlaneSamples:
    """[-1, 1]² 矩形均匀随机采样 *count* 个点。"""
    return torch.rand(count, 2).mul_(2.0).sub_(1.0)


# ── 分派表（组合合法性的唯一事实源） ──

_SAMPLERS: Final[dict[tuple[SampleMethod, SampleRegion], Callable[[Any], Tensor]]] = {
    ("uniform", "line"): _line_uniform,
    ("uniform", "disk"): _disk_uniform,
    ("uniform", "rect"): _rect_uniform,
    ("random", "line"): _line_random,
    ("random", "disk"): _disk_random,
    ("random", "rect"): _rect_random,
    ("fibonacci", "disk"): _disk_fibonacci,
}

# count 取 (na, nb) 二元组的组合（uniform 二维网格）；其余组合均为 int
_GRID_COUNTS: Final = {("uniform", "disk"), ("uniform", "rect")}


# ── 配置与入口 ──


@final
@dataclass(frozen=True, slots=True)
class SampleOptions:
    """采样配置，构造时自动兼容性校验。

    合法 (method, region) 组合与 count 约定：

    * ``("uniform", "line")``   —— int：等距点数
    * ``("uniform", "disk")``   —— ``(rings, sectors)``：环数含圆心
    * ``("uniform", "rect")``   —— ``(na, nb)``：两轴网格数
    * ``("random", ...)``       —— int：随机点数
    * ``("fibonacci", "disk")`` —— int：总点数，首点为圆心
    """

    method: SampleMethod
    region: SampleRegion
    count: SampleCount

    def __post_init__(self) -> None:
        combo = (self.method, self.region)
        if combo not in _SAMPLERS:
            if self.method == "fibonacci":
                raise ValueError(
                    f"fibonacci sampling requires region='disk', got region={self.region!r}"
                )
            raise ValueError(f"unsupported (method, region): {combo!r}")

        if combo in _GRID_COUNTS:
            if not isinstance(self.count, tuple):
                raise TypeError(
                    f"({self.method}, {self.region}) expects count: tuple[int, int], "
                    f"got {type(self.count).__name__}"
                )
            if len(self.count) != 2 or not all(isinstance(c, int) for c in self.count):
                raise TypeError(f"count must be a pair of ints, got {self.count!r}")
            if self.count[0] <= 0 or self.count[1] <= 0:
                raise ValueError(f"count elements must be positive, got {self.count}")
        else:
            if not isinstance(self.count, int):
                raise TypeError(
                    f"({self.method}, {self.region}) expects count: int, "
                    f"got {type(self.count).__name__}"
                )
            if self.count <= 0:
                raise ValueError(f"count must be positive, got {self.count}")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        """从原始配置映射构造，非法输入 fail-fast。

        键名经 :class:`~core.noun.Noun` 别名机制解析（``"method"`` /
        ``"Method"`` 等写法等价）；TOML 数组自动 list → tuple。
        """
        method_raw = term.METHOD.resolve(options)
        region_raw = term.REGION.resolve(options)
        count_raw = term.COUNT.resolve(options)

        if not isinstance(method_raw, str) or method_raw not in _VALID_METHODS:
            raise ValueError(
                f"unknown sampling method: {method_raw!r} "
                f"(expected one of: {', '.join(sorted(_VALID_METHODS))})"
            )
        if not isinstance(region_raw, str) or region_raw not in _VALID_REGIONS:
            raise ValueError(
                f"unknown sampling region: {region_raw!r} "
                f"(expected one of: {', '.join(sorted(_VALID_REGIONS))})"
            )

        count = tuple(count_raw) if isinstance(count_raw, list) else count_raw
        return cls(
            method=cast(SampleMethod, method_raw),
            region=cast(SampleRegion, region_raw),
            count=cast(SampleCount, count),
        )


@torch.no_grad()
def sample(options: SampleOptions) -> Tensor:
    """按配置生成归一化点集：line → ``(N,)``；disk / rect → ``(N, 2)``。

    组合合法性由 :class:`SampleOptions` 构造时保证，此处直接查表分派。
    """
    return _SAMPLERS[(options.method, options.region)](options.count)
