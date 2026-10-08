"""采样模块：归一化区域（line / disk / rect）上的点集生成。

模块组织
--------
* :mod:`sampling._core` —— :class:`SampleOptions`（构造即校验的采样配置）
  与 :func:`sample`（查表分派的点集生成）。
"""

from sampling._core import (
    SampleCount,
    SampleMethod,
    SampleOptions,
    SampleRegion,
    sample,
)

__all__ = [
    "SampleCount",
    "SampleMethod",
    "SampleOptions",
    "SampleRegion",
    "sample",
]
