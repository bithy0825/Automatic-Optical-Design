"""光学面形模块：具体曲面类型及其光线追迹。

模块组织
--------
* :mod:`shape.protocol` —— 抽象基类 :class:`Shape`（含 ``_train_flags``
  可训练标记解析）。
* :mod:`shape.trace` —— 光线-曲面相交结果与追迹函数。
* :mod:`shape.sphere` —— 球面。
* :mod:`shape.conic` —— 圆锥曲面（椭球 / 抛物面 / 双曲面）。
* :mod:`shape.asphere` —— 偶次非球面（圆锥基底 + 多项式）。
* :mod:`shape.disk` —— 平面（无限薄孔径光阑 / 像面）。
"""

from shape.asphere import Asphere
from shape.conic import Conic
from shape.disk import Disk
from shape.protocol import Shape
from shape.sphere import Sphere
from shape.trace import (
    ApertureFunction,
    TraceResult,
    circle_aperture,
    intersect,
    no_aperture,
)

__all__ = [
    # 基类
    "Shape",
    # 追迹
    "TraceResult",
    "ApertureFunction",
    "no_aperture",
    "circle_aperture",
    "intersect",
    # 面形
    "Asphere",
    "Conic",
    "Disk",
    "Sphere",
]
