from dataclasses import dataclass
from typing import Literal, Self, cast, final, get_args as _get_args

import torch
from jaxtyping import Float
from torch import Tensor

from core import term

type SampleMethod = Literal["uniform", "random", "fibonacci"]
type SampleRegion = Literal["line", "disk", "rect"]
type SampleCount = int | tuple[int, int]

_VALID_METHODS = frozenset(_get_args(SampleMethod.__value__))
_VALID_REGIONS = frozenset(_get_args(SampleRegion.__value__))

_GOLDEN_ANGLE: float = torch.pi * (3.0 - 5.0**0.5)  # π·(3 − √5)


def _validate_fibonacci(method: SampleMethod, region: SampleRegion) -> None:
    """Fibonacci 采样仅适用于 disk 区域。"""
    if method == "fibonacci" and region != "disk":
        raise ValueError(
            f"Fibonacci sampling requires region='disk', got region={region!r}"
        )


def _validate_count_type(
    method: SampleMethod,
    region: SampleRegion,
    count: SampleCount,
) -> None:
    """校验 count 类型是否与 (method, region) 组合匹配。"""
    match (method, region):
        case ("fibonacci", "disk") | ("random", _) | ("uniform", "line"):
            if not isinstance(count, int):
                raise TypeError(
                    f"({method}, {region}) expects count: int, "
                    f"got {type(count).__name__}"
                )
        case ("uniform", "disk" | "rect"):
            if not isinstance(count, tuple):
                raise TypeError(
                    f"({method}, {region}) expects count: tuple[int, int], "
                    f"got {type(count).__name__}"
                )
        case _:
            raise ValueError(f"Unknown (method, region): ({method!r}, {region!r})")


def _validate_count_positive(count: SampleCount) -> None:
    """校验 count 值为正（int > 0 或二元正 int 元组）。"""
    match count:
        case int(n) if n <= 0:
            raise ValueError(f"count must be positive, got {n}")
        case (int(a), int(b)) if a <= 0 or b <= 0:
            raise ValueError(f"Count tuple elements must be positive, got ({a}, {b})")
        case int() | (int(), int()):
            return
        case _:
            raise TypeError(f"Count must be int or tuple[int, int], got {count!r}")


@final
@dataclass(frozen=True, slots=True)
class SampleOptions:
    """采样配置，构造时自动兼容性校验。

    Attributes:
        method: ``"uniform"`` / ``"random"`` / ``"fibonacci"``。
        region: ``"line"`` / ``"disk"`` / ``"rect"``。
        count:  采样点数，类型由 (method, region) 决定。

    Raises:
        TypeError: count 类型不匹配。
        ValueError: (method, region) 不支持或 count 值非法。
    """

    method: SampleMethod
    region: SampleRegion
    count: SampleCount

    def __post_init__(self) -> None:
        _validate_fibonacci(self.method, self.region)
        _validate_count_type(self.method, self.region, self.count)
        _validate_count_positive(self.count)

    @classmethod
    def from_options(cls, options: dict[str, object]) -> Self:
        """从原始配置映射构造，非法输入直接抛出异常。

        *options* 的键名支持多种写法（由 :class:`~core.noun.Noun` 别名机制解析）：
        ``"method"`` / ``"sample method"`` / ``"Sample Method"`` 等均可识别。

        Args:
            options: 包含采样方法、区域、点数配置的字典。

        Returns:
            合法的 :class:`SampleOptions` 实例。

        Raises:
            KeyError: 缺少必需的键。
            ValueError: method / region 值不合法。
            TypeError: count 类型与 (method, region) 不兼容。
        """
        method_raw = term.METHOD.resolve(options)
        region_raw = term.REGION.resolve(options)
        count_raw = term.COUNT.resolve(options)

        if not isinstance(method_raw, str) or method_raw not in _VALID_METHODS:
            raise ValueError(
                f"Unknown {term.METHOD}: {method_raw!r}. Expected one of {_VALID_METHODS}"
            )
        if not isinstance(region_raw, str) or region_raw not in _VALID_REGIONS:
            raise ValueError(
                f"Unknown {term.REGION}: {region_raw!r}. Expected one of {_VALID_REGIONS}"
            )

        match count_raw:
            case list(items):
                count: SampleCount = tuple(items)
            case _:
                count = cast(SampleCount, count_raw)

        return cls(
            method=cast(SampleMethod, method_raw),
            region=cast(SampleRegion, region_raw),
            count=count,
        )


def _line_uniform(count: int) -> Float[Tensor, "N"]:
    """[-1, 1] 等距采样 *count* 个点。"""
    return _axis_coords(count)


def _line_random(count: int) -> Float[Tensor, "N"]:
    """[-1, 1] 均匀随机采样 *count* 个点。"""
    return torch.rand(count).mul_(2.0).sub_(1.0)


def _disk_uniform(count: tuple[int, int]) -> Float[Tensor, "N 2"]:
    """单位圆盘网格采样（同心环 + 等分圆周）。

    Args:
        count: ``(rings, sectors)``，*rings* 含圆心。
               总点数 = rings×sectors − sectors + 1。
    """
    rings, sectors = count
    radii = torch.linspace(0.0, 1.0, rings)[1:]
    angles = torch.arange(sectors, dtype=torch.float32).div_(sectors).mul_(2 * torch.pi)

    r, a = torch.meshgrid(radii.sqrt(), angles, indexing="ij")
    x, y = r.mul(a.cos()).ravel(), r.mul(a.sin()).ravel()
    ring_pts = torch.stack((x, y), dim=-1)  # shape: [(rings-1)*sectors, 2]
    return torch.cat(
        (torch.zeros(1, 2), ring_pts), dim=0
    )  # shape: [rings*sectors - sectors + 1, 2]


def _disk_random(count: int) -> Float[Tensor, "N 2"]:
    """单位圆盘均匀随机采样 *count* 个点。"""
    rho = torch.rand(count).sqrt_()
    theta = torch.rand(count).mul_(2 * torch.pi)
    return torch.stack((rho * theta.cos(), rho * theta.sin()), dim=-1)


def _disk_fibonacci(count: int) -> Float[Tensor, "N 2"]:
    """单位圆盘 Fibonacci 螺旋采样 *count* 个点。"""
    if count == 1:
        return torch.zeros(1, 2)

    i = torch.arange(1, count, dtype=torch.float32)
    rho = i.div(count - 1).sqrt()
    theta = i.mul(_GOLDEN_ANGLE)

    ring = torch.stack((rho * theta.cos(), rho * theta.sin()), dim=-1)
    return torch.cat((torch.zeros(1, 2), ring))


def _rect_uniform(count: tuple[int, int]) -> Float[Tensor, "N 2"]:
    """[-1, 1]² 矩形网格均匀采样。

    Args:
        count: ``(na, nb)`` —— 输出列 0 沿第一轴取 *na* 个值、列 1 沿
               第二轴取 *nb* 个值，总点数 = na × nb。
    """
    na, nb = count
    a = _axis_coords(na)
    b = _axis_coords(nb)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    return torch.stack((aa.ravel(), bb.ravel()), dim=-1)


def _rect_random(count: int) -> Float[Tensor, "N 2"]:
    """[-1, 1]² 矩形均匀随机采样 *count* 个点。"""
    return torch.rand(count, 2).mul_(2.0).sub_(1.0)


def _axis_coords(n: int) -> Tensor:
    """单轴坐标：*n=1* → ``[0]``，否则 [-1, 1] 上 *n* 等分。"""
    return torch.zeros(1) if n == 1 else torch.linspace(-1.0, 1.0, n)


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def sample(options: SampleOptions) -> Tensor:
    """根据采样配置生成点集。

    Args:
        options: :class:`SampleOptions` 实例。

    Returns:
        采样点张量，形状取决于 (method, region)。
    """
    match (options.method, options.region):
        case ("uniform", "line"):
            return _line_uniform(cast(int, options.count))
        case ("uniform", "disk"):
            return _disk_uniform(cast(tuple[int, int], options.count))
        case ("uniform", "rect"):
            return _rect_uniform(cast(tuple[int, int], options.count))
        case ("random", "line"):
            return _line_random(cast(int, options.count))
        case ("random", "disk"):
            return _disk_random(cast(int, options.count))
        case ("random", "rect"):
            return _rect_random(cast(int, options.count))
        case ("fibonacci", "disk"):
            return _disk_fibonacci(cast(int, options.count))
        case _:
            raise ValueError(
                f"Unsupported (method, region): ({options.method!r}, {options.region!r})"
            )
