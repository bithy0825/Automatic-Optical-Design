"""核心模块：追迹数据模型、光学模块基类、配置词表与底层张量原语。

模块组织
--------
* :mod:`core.aliases` —— 张量形状约定（jaxtyping 别名 + System→Ray 对齐原语）。
* :mod:`core.noun` / :mod:`core.term` —— 规范名词与全库唯一词表。
* :mod:`core.container` —— 张量容器基类（``apply`` / ``to`` / ``clone`` 免费获得）。
* :mod:`core.flow` —— 追迹数据流：``Verdict`` / ``RayBundle`` / ``TraceFlow``。
* :mod:`core.transformer` —— 逐个体 4×4 齐次变换（正 / 逆矩阵成对维护）。
* :mod:`core.module` —— 光学模块基类 :class:`OpticalModule`（种群语义 + GA 演化）。
* :mod:`core.params` —— 参数注册与分布规格解析。
* :mod:`core.sturdy_math` —— NaN 安全的数值原语。
* :mod:`core.repr` —— rich 树形打印与参数格式化。
"""

from core import term
from core.aliases import (
    HomMatrix,
    RayBoolScalar,
    RayFloat2D,
    RayFloat3D,
    RayFloatMatrix2D,
    RayFloatMatrix3D,
    RayFloatMatrix4D,
    RayFloatScalar,
    RayLongScalar,
    SystemBoolND,
    SystemBoolScalar,
    SystemFloat2D,
    SystemFloat3D,
    SystemFloatMatrix2D,
    SystemFloatMatrix3D,
    SystemFloatMatrix4D,
    SystemFloatND,
    SystemFloatScalar,
    SystemLongScalar,
    broadcast_system_to_ray,
)
from core.container import TensorContainer
from core.flow import RayBundle, TraceFlow, Verdict
from core.module import OpticalModule
from core.noun import Noun
from core.params import init_param, parse_param
from core.repr import fmt_param
from core.sturdy_math import sturdy_div, sturdy_inv, sturdy_sqrt
from core.transformer import Transformer

__all__ = [
    # 形状约定
    "HomMatrix",
    "RayBoolScalar",
    "RayFloat2D",
    "RayFloat3D",
    "RayFloatMatrix2D",
    "RayFloatMatrix3D",
    "RayFloatMatrix4D",
    "RayFloatScalar",
    "RayLongScalar",
    "SystemBoolND",
    "SystemBoolScalar",
    "SystemFloat2D",
    "SystemFloat3D",
    "SystemFloatMatrix2D",
    "SystemFloatMatrix3D",
    "SystemFloatMatrix4D",
    "SystemFloatND",
    "SystemFloatScalar",
    "SystemLongScalar",
    "broadcast_system_to_ray",
    # 容器与基类
    "TensorContainer",
    "OpticalModule",
    # 词表
    "Noun",
    "term",
    # 追迹数据流
    "RayBundle",
    "TraceFlow",
    "Transformer",
    "Verdict",
    # 参数构造
    "init_param",
    "parse_param",
    # 数值原语与打印
    "sturdy_div",
    "sturdy_inv",
    "sturdy_sqrt",
    "fmt_param",
]
