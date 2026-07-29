"""张量容器的公共基类。

由张量字段（或嵌套的容器字段）组成的 dataclass 继承本类，即免费获得
``apply`` / ``to`` / ``clone`` / ``detach`` 一组变换，无需逐类手写。

约定：子类必须是 dataclass；字段要么是老张量，要么同为容器（递归处理）。
"""

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Self

import torch

from core.repr import render_line, styled


@dataclass(slots=True, eq=False)
class TensorContainer:
    def __repr__(self) -> str:
        def show(v: object) -> object:
            return tuple(v.shape) if isinstance(v, torch.Tensor) else type(v).__name__

        info = ", ".join(
            f"{f.name}={show(getattr(self, f.name))}" for f in fields(self)
        )
        return render_line(styled(type(self).__name__, info))

    def apply(self, func: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """对每个张量字段应用 *func*，返回同类型的新容器（嵌套容器递归）。"""
        kwargs = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, TensorContainer):
                kwargs[f.name] = value.apply(func)
            else:
                kwargs[f.name] = func(value)
        return replace(self, **kwargs)

    def to(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> Self:
        """移动设备/精度。*dtype* 只作用于浮点张量（bool/long 仅迁移设备）。"""
        return self.apply(
            lambda t: t.to(
                device=device, dtype=dtype if t.is_floating_point() else None
            )
        )

    def clone(self) -> Self:
        return self.apply(torch.Tensor.clone)

    def detach(self) -> Self:
        return self.apply(torch.Tensor.detach)

    def replace(self, **kwargs: object) -> Self:
        return replace(self, **kwargs)

    @property
    def device(self) -> torch.device:
        for f in fields(self):
            value = getattr(self, f.name)
            return value.device
        raise RuntimeError(f"{type(self).__name__} has no tensor fields")

    @property
    def dtype(self) -> torch.dtype:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                return value.dtype
        raise RuntimeError(f"{type(self).__name__} has no floating-point tensor fields")
