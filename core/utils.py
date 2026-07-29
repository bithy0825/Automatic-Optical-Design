from typing import Final, Any
from collections.abc import Mapping

import torch

from core.noun import Noun
from core import term


_LEADING_AXES: Final[int] = 4  # (P, F, W, N) for every ray tensor; per core.aliases.


def broadcast_system_to_ray(
    system_tensor: torch.Tensor, ray_tensor: torch.Tensor
) -> torch.Tensor:
    """把逐个体张量 ``(P, ...)`` 广播对齐到逐光线张量 ``(P, F, W, N, ...)``。"""
    n_leading = ray_tensor.ndim - system_tensor.ndim
    if n_leading != _LEADING_AXES - 1:
        raise ValueError(
            f"Expected system tensor to have {ray_tensor.ndim - _LEADING_AXES + 1} leading axes, "
            f"but got {n_leading} leading axes instead."
        )
    new_shape = (
        (system_tensor.shape[0],) + (1,) * n_leading + tuple(system_tensor.shape[1:])
    )
    return system_tensor.reshape(new_shape).expand_as(ray_tensor).to(ray_tensor.dtype)


def fmt_param(
    x: torch.Tensor | list | tuple, *, precision: int = 4, max_items: int = 8
) -> str:
    def fmt_item(v: object) -> str:
        return f"{v:.{precision}g}" if type(v) in (int, float) else str(v)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 0:
            return fmt_item(x.item())
        vals = x.tolist()
    elif isinstance(x, (list, tuple)):
        vals = list(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

    head = ", ".join(fmt_item(v) for v in vals[:max_items])
    return f"[{head}]" if len(vals) <= max_items else f"[{head}, …]"


def parse_param(
    options: Mapping[str, Any],
    key: Noun | str,
    population: int,
) -> torch.Tensor:
    """按配置规格生成 ``(P,)`` 浮点张量。

    规格形式：``{method: "raw", value: x}``、``{method: "normal", mean, std}``
    或 ``{method: "uniform", low, high}``。设备与精度取运行时默认（模块
    整体的迁移由构造后的 ``.to()`` 完成）。
    """
    if isinstance(key, str):
        param = options.get(key)
        if param is None:
            raise KeyError(f"Missing parameter: {key}")
    else:
        param = key.resolve(options)

    method = term.METHOD.resolve(param)

    if term.RAW.match(method):
        value = term.VALUE.resolve(param)
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected {term.VALUE} to be a number, got {type(value)}")
        return torch.full((population,), float(value))

    if term.NORMAL.match(method):
        mean = term.MEAN.resolve(param)
        std = term.STD.resolve(param)
        if not isinstance(mean, (int, float)):
            raise TypeError(f"Expected {term.MEAN} to be a number, got {type(mean)}")
        if not isinstance(std, (int, float)):
            raise TypeError(f"Expected {term.STD} to be a number, got {type(std)}")
        return torch.normal(mean=mean, std=std, size=(population,))

    if term.UNIFORM.match(method):
        low = term.LOW.resolve(param)
        high = term.HIGH.resolve(param)
        if not isinstance(low, (int, float)):
            raise TypeError(f"Expected {term.LOW} to be a number, got {type(low)}")
        if not isinstance(high, (int, float)):
            raise TypeError(f"Expected {term.HIGH} to be a number, got {type(high)}")
        if low >= high:
            raise ValueError(f"Expected {term.LOW} < {term.HIGH}, got {low} >= {high}")
        return torch.rand((population,)).mul(high - low).add(low)

    raise ValueError(
        f"Unknown {term.METHOD}: {method!r}. Expected one of {term.METHOD.aliases}"
    )
