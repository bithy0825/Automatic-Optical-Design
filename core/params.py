"""参数的注册与配置规格解析。

* :func:`init_param` —— 把初值注册到模块名下（``Parameter`` 或 buffer）。
* :func:`parse_param` —— 把分布配置规格物化为 ``(P,)`` 浮点张量。
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from core import term
from core.noun import Noun


def init_param(
    parent: nn.Module,
    name: str | Noun,
    value: float | Sequence[float] | torch.Tensor,
    trainable: bool = False,
) -> torch.Tensor:
    """把 *value* 注册到 *parent* 名下：*trainable* 注册 ``Parameter``，否则注册 buffer。

    *name* 为 :class:`Noun` 时取其规范形式。张量原样接管（detach）；标量 /
    序列按 *parent* 已有参数的设备建张量（无参数时取进程缺省设备）。
    """
    key = name.canonical if isinstance(name, Noun) else name

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        first_param = next(parent.parameters(), None)
        device = first_param.device if first_param is not None else torch.get_default_device()
        tensor = torch.tensor(value, device=device)

    if trainable:
        param = nn.Parameter(tensor)
        parent.register_parameter(key, param)
        return param
    parent.register_buffer(key, tensor)
    return tensor


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
            raise KeyError(f"missing parameter: {key}")
    else:
        param = key.resolve(options)

    method = term.METHOD.resolve(param)

    if term.RAW.match(method):
        return torch.full((population,), _real(term.VALUE.resolve(param), term.VALUE))
    if term.NORMAL.match(method):
        mean = _real(term.MEAN.resolve(param), term.MEAN)
        std = _real(term.STD.resolve(param), term.STD)
        return torch.normal(mean=mean, std=std, size=(population,))
    if term.UNIFORM.match(method):
        low = _real(term.LOW.resolve(param), term.LOW)
        high = _real(term.HIGH.resolve(param), term.HIGH)
        if low >= high:
            raise ValueError(f"uniform spec expects low < high, got {low} >= {high}")
        return torch.rand((population,)).mul(high - low).add(low)
    raise ValueError(f"unknown sampling method: {method!r} (expected one of: raw, normal, uniform)")


def _real(value: Any, key: Noun) -> float:
    """规格字段 → Python 浮点数；非数值类型 fail-fast。"""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} spec expects a number, got {type(value).__name__}")
    return float(value)
