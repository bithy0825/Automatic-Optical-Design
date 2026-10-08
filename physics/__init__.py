"""物理模块：光线与介质界面的相互作用（镜面反射 / Snell 折射）。

模块组织
--------
* :mod:`physics._core` —— :class:`InteractionResult` 与 :func:`reflect` /
  :func:`refract` 两个相互作用算子。
"""

from physics._core import InteractionResult, reflect, refract

__all__ = [
    "InteractionResult",
    "reflect",
    "refract",
]
